from difflib import restore
from functools import partial

from requests import PreparedRequest

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from coperception.models.transformers.mae_base import *
from coperception.utils.maeutil.pos_embed import get_2d_sincos_pos_embed
import math

from coperception.utils.softmax_focal_loss import SoftmaxFocalLoss

class VQSTARViT(MultiAgentMaskedAutoencoderViT):
    """
    STAR + Vector Quantizer
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_head="mlp", norm_pix_loss=False, time_stamp=1, 
                 mask_method="complement", encode_partial=False, no_temp_emb=False, decode_singletemp=False, 
                 decay=0., commitment_cost=0.25, num_vq_embeddings=512, vq_embedding_dim=64):
        super(VQSTARViT, self).__init__(
                img_size=img_size, 
                patch_size=patch_size, 
                in_chans=in_chans,
                embed_dim=embed_dim, 
                depth=depth, 
                num_heads=num_heads,
                decoder_embed_dim=decoder_embed_dim, 
                decoder_depth=decoder_depth, 
                decoder_num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio, 
                norm_layer=norm_layer, 
                decoder_head = decoder_head,
                norm_pix_loss=norm_pix_loss,
                time_stamp = time_stamp,
                mask_method = mask_method,
                inter_emb_dim = vq_embedding_dim)
        # for neighbor agents' features
        self.patch_h = 0
        self.patch_w = 0
        # class_weights = torch.FloatTensor([1.0, 20.0]) # [free, occ]
        # print("Using weighted cross entropy loss with weights", class_weights)
        # self.cls_loss = nn.CrossEntropyLoss(weight = class_weights)
        self.cls_loss = nn.CrossEntropyLoss()
        # self.focal_loss = SoftmaxFocalLoss(alpha=0.75, gamma=3)

        # quantizer part
        if decay > 0.0:
            self._vq_star = STARVectorQuantizerEMA(num_vq_embeddings, vq_embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_star = STARVectorQuantizer(num_vq_embeddings, vq_embedding_dim,
                                           commitment_cost)

        
        if mask_method == "random":
            print("do random masking")
            self.masking_handle = self.amortized_random_masking
            self.unmasking_handle = self.amortized_random_unmasking
        elif mask_method == "complement":
            print("do complement masking")
            self.masking_handle = self.amortized_complement_masking
            self.unmasking_handle = self.amortized_complement_unmasking
        else:
            raise NotImplementedError(mask_method)

        # ---- ablation study for encoder ------
        if encode_partial:
            print("encode after masking")
            self.forward_encoder = self.forward_encoder_partial
        else:
            print("encoder BEFORE masking")
            self.forward_encoder = self.forward_encoder_all

        # ---- ablation study for decoder ------
        self.decode_singletemp = decode_singletemp
        self.no_temp_emb = no_temp_emb

        

    def amortized_random_masking(self, x_ts, mask_ratio):
        """
        Random masking for each data of each timestamp.
        Maintain indices for decoder to fuse and possibly fill in mask tokens
        x_ts: [bxa x time_stamp, L, D]
        """
        xts_masked, mask, ids_restore = self.random_masking(x_ts, mask_ratio)
        return xts_masked, mask, ids_restore

    def amortized_random_unmasking(self, x_ts, mask, ids_restore):
        """
        Reverse masking with fillings from other timestamp, using mask and torch where
        x_ts: [BxAxts, L, D]
        mask: [BxAxts, L] -> full length L
        ids_restore: [BxAxts, L] -> full length L
        """
        # print(x_ts.size(), mask.size(), ids_restore.size())
        # fill with mask tokens
        mask_tokens = self.mask_token.repeat(x_ts.shape[0], ids_restore.shape[1] + 1 - x_ts.shape[1], 1)
        x_ts_ = torch.cat([x_ts, mask_tokens], dim=1)  # no cls token
        x_ts_ = torch.gather(x_ts_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_ts.shape[2]))  # unshuffle
        BAT, L, D = x_ts_.size()
        assert BAT%self.time_stamp == 0, (BAT, self.time_stamp)
        BA = BAT // self.time_stamp
        # add temporal embedding
        x_ts_ = x_ts_.reshape(BA, self.time_stamp, L, D)

        # ---- ablation: use time emb -----
        if not self.no_temp_emb:
            # print("use temporal embedding")
            x_ts_ = x_ts_ + self.decoder_temp_embed #check dim
        # ----------------------------------

        # figure out how to fill in the other timestamp
        x_ts_unp = x_ts_.reshape(BA, self.time_stamp, self.patch_h, self.patch_w, x_ts_.shape[-1])
        # print("x_ts_unp size", x_ts_unp.size())
        mask_unp = mask.reshape(mask.shape[0], self.patch_h, self.patch_w) #self.unpatchify(mask.unsqueeze(-1))
        # assert mask_unp.size(1) == 1, (mask_unp.size())
        mask_unp = mask_unp.reshape(BA, self.time_stamp, self.patch_h, self.patch_w) # [BxA, ts, H, W], 0 is kept, 1 is removed
        reverse_mask = 1. - mask_unp # 0 is removed, >0 is kept
        mask_filled = reverse_mask[:,0,:,:] # current timestamp, 
        x_curr = x_ts_unp[:,0,:,:,:] #[BxA, h, w, D]
        # print("x_curr size", x_curr.size())
        if self.decode_singletemp:
            print("decode single time stamp")
            x_curr = x_curr.permute(0,3,1,2)
            return x_curr
        else:
            for ti in range(self.time_stamp-1):
                # not filled previously but can be filled by ti+1
                mask_curr = reverse_mask[:,ti+1, :, :]
                mask_select = ((mask_filled==0.) & (mask_curr>0.)).unsqueeze(-1)
                x_curr = torch.where(mask_select, x_ts_unp[:, ti+1, :, :, :], x_curr)
                mask_filled = mask_filled + mask_curr # update the mask
            x_curr = x_curr.permute(0,3,1,2) #[BxA, C, H, W]
            return x_curr

    def amortized_complement_masking(self, x_ts, mask_ratio):
        """
        x_ts: [bxa x time_stamp, L, D]
        NOTE: the returned mask and ids_restore has a different dimensionality from the random masking
        """
        mask_ratio = 1.0 - 1.0 / self.time_stamp # e.g. times_stamp = 1, mask ratio ==0, 2, 0.5
        BAT, L, D = x_ts.shape
        assert BAT % self.time_stamp == 0
        BA = BAT // self.time_stamp
        # len_keep_each = int(L * (1 - mask_ratio))
        len_keep_each = math.ceil(L * (1 - mask_ratio)) # make sure every patch is covered
        noise = torch.rand(BA, L, device=x_ts.device)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        x_ts = x_ts.reshape(BA, self.time_stamp, L, D)

        mask = torch.ones([BA, self.time_stamp, L], device=x_ts.device)
        x_masked = []
        # keep the subset for each timestamp
        for ti in range(self.time_stamp):
            ids_keep = ids_shuffle[:, len_keep_each*ti : min(len_keep_each*(ti+1), L)]
            x_ts_masked = torch.gather(x_ts[:,ti,:,:], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
            if ids_keep.size(1) < len_keep_each:
                # padding
                x_padded = torch.zeros(ids_keep.size(0), len_keep_each - ids_keep.size(1), D, device=x_ts.device)
                x_ts_masked = torch.cat((x_ts_masked, x_padded), dim=1)
            x_masked.append(x_ts_masked)
            mask[:, ti, len_keep_each*ti : min(len_keep_each*(ti+1), L)] = 0
            mask[:, ti, :] = torch.gather(mask[:, ti, :], dim=1, index=ids_restore)
        x_masked = torch.stack(x_masked, dim=0).permute(1,0,2,3) #[BA, timestamp, L, D]
        x_masked = x_masked.reshape(BAT, -1, D) # because the seq len is not L after masking

        return x_masked, mask, ids_restore

    def amortized_complement_unmasking(self, x_ts, mask, ids_restore):
        """
        x_ts: [BxAxts, L, D]
        mask: [BxA, ts, L] -> this is real L
        ids_restore: [BxA, L] -> this is real L
        restore, add temporal embedding
        NOTE: subtlety: because of complementary masking, this L can be padded at the end thus >= real L
        """
        x_ts_ = x_ts
        BAT, L, D = x_ts_.shape
        realL = ids_restore.size(1)
        BA = BAT // self.time_stamp
        x_ts_ = x_ts_.reshape(BA, self.time_stamp, L, D)
        if not self.no_temp_emb:
            # print("use temporal embedding")
            x_ts_ = x_ts_ + self.decoder_temp_embed #check dim
        if self.decode_singletemp:
            # print("decode single timestamp")
            x_curr = x_ts_[:,0, :realL, :] # [BA, realL, D]
            mask_tokens = self.mask_token.repeat(x_curr.shape[0], ids_restore.shape[1] + 1 - x_curr.shape[1], 1)
            x_curr = torch.cat([x_curr, mask_tokens], dim=1)  # no cls token
            x_restore = torch.gather(x_curr, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_curr.shape[2]))  # unshuffle
        else:
            x_restore = []
            for ti in range(self.time_stamp):
                x_restore.append(x_ts_[:,ti, :, :]) #[BA, L, D]
            x_restore = torch.cat(x_restore, dim=1)
            x_restore = x_restore[:, :realL, :]
            x_restore = torch.gather(x_restore, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_ts.shape[2]))

        feature_maps = x_restore.reshape(BA, self.patch_h, self.patch_w, D)
        return feature_maps


    def late_fusion(self, pred, trans_matrices, num_agent_tensor, batch_size):
        """ reshape the model's predictions back to 256x256x13, do aggregation on this"""
        device = pred.device
        # indiv_imgs = self.unpatchify(pred) # [B C H W]
        indiv_imgs = pred.type(torch.FloatTensor).to(device)
        ## --- do fusion ---
        size = self.get_feature_maps_size(indiv_imgs)
        # print(size)
        assert indiv_imgs.size(0) % batch_size == 0, (indiv_imgs.size(), batch_size)
        self.num_agent = indiv_imgs.size(0) // batch_size
        feat_list = self.build_feature_list(batch_size, indiv_imgs)
        # [[1,1,256,32,32]x5] NOTE should it be [[B, 1, 256, 32, 32]x5]?
        # print(feat_list)
        local_com_mat = self.build_local_communication_matrix(feat_list)  # [2 5 13 256 256] [batch, agent, channel, height, width]
        local_com_mat_update = self.build_local_communication_matrix(feat_list)  # to avoid the inplace operation

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                self.tg_agent = local_com_mat[b, i]
                # print("tg agent shape", self.tg_agent.shape) #[13,256,256] 
                self.neighbor_feat_list = []
                self.neighbor_feat_list.append(self.tg_agent)
                all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]
                # print(all_warp.shape)[5,4,4]
                self.build_neighbors_feature_list(b, i, all_warp, num_agent, local_com_mat,
                                                    device, size)

                # feature update
                # torch.save(torch.stack(self.neighbor_feat_list).detach().cpu(), "/mnt/NAS/home/zjx/Masked-Multiagent-Autoencoder/debug/nbf-{}-{}.pt".format(b, i))
                local_com_mat_update[b, i] = self.fusion(mode="union") 
        
        # weighted feature maps is passed to decoder
        fused_images = self.agents_to_batch(local_com_mat_update)
        # print('feat fuse mat size', feat_fuse_mat.size())
        return fused_images

    def ego_late_fusion(self, pred, gt, trans_matrices, num_agent_tensor, batch_size):
        """ aggregation, ego use ground truth input, used specifically in inference time"""
        device = pred.device
        # indiv_imgs = self.unpatchify(pred) # [B C H W]
        indiv_imgs = pred.type(torch.FloatTensor).to(device)
        ## --- do fusion ---
        size = self.get_feature_maps_size(indiv_imgs)
        # print(size)
        assert indiv_imgs.size(0) % batch_size == 0, (indiv_imgs.size(), batch_size)
        self.num_agent = indiv_imgs.size(0) // batch_size
        feat_list = self.build_feature_list(batch_size, indiv_imgs)
        gt_list = self.build_feature_list(batch_size, gt)
        # print(feat_list)
        local_com_mat = self.build_local_communication_matrix(feat_list)  # [2 5 13 256 256] [batch, agent, channel, height, width]
        local_com_mat_update = self.build_local_communication_matrix(feat_list)  # to avoid the inplace operation
        local_com_mat_gt = self.build_local_communication_matrix(gt_list)

        for b in range(batch_size):
            num_agent = num_agent_tensor[b, 0]
            for i in range(num_agent):
                # use gt for the ego
                self.tg_agent = local_com_mat_gt[b, i]
                # print("tg agent shape", self.tg_agent.shape) #[13,256,256] 
                self.neighbor_feat_list = []
                self.neighbor_feat_list.append(self.tg_agent)
                all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]
                # print(all_warp.shape)[5,4,4]
                self.build_neighbors_feature_list(b, i, all_warp, num_agent, local_com_mat,
                                                    device, size)

                # feature update
                # torch.save(torch.stack(self.neighbor_feat_list).detach().cpu(), "/mnt/NAS/home/zjx/Masked-Multiagent-Autoencoder/debug/nbf-{}-{}.pt".format(b, i))
                local_com_mat_update[b, i] = self.fusion(mode="union") 
        
        # weighted feature maps is passed to decoder
        fused_images = self.agents_to_batch(local_com_mat_update)
        # print('feat fuse mat size', feat_fuse_mat.size())
        return fused_images

    def forward_encoder_all(self, x1, x_next, mask_ratio):
        """
        x1: [bxa, C, H, W]
        x_next: [bxa, ts-1, C, H, W] beq_next_frames
        """
        # cat x1 and x_next to encoder independently
        BA, C, H, W = x1.size()
        x1 = x1.unsqueeze(1)
        if self.time_stamp>1:
            x_ind = torch.cat((x1, x_next), dim=1) # [bxa, ts, C, H, W]
        else:
            x_ind = x1
        # print(x_ind.size())
        assert x_ind.size(1) == self.time_stamp
        x_ind = x_ind.reshape(BA*self.time_stamp, C, H, W)
        # embed patches
        x_ind = self.patch_embed(x_ind)
        x_ind = x_ind + self.pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.blocks:
            x_ind = blk(x_ind)
        x = self.norm(x_ind)
        # compress for communication
        # TODO: mask ONLY for transmission, encode the complete sequence
        x_masked, mask, ids_restore = self.masking_handle(x, mask_ratio)
        x = self.compressor(x_masked)

        return x, mask, ids_restore

    def forward_encoder_partial(self, x1, x_next, mask_ratio):
        """
        x1: [bxa, C, H, W]
        x_next: [bxa, ts-1, C, H, W] beq_next_frames
        """
        # cat x1 and x_next to encoder independently
        BA, C, H, W = x1.size()
        x1 = x1.unsqueeze(1)
        if self.time_stamp>1:
            x_ind = torch.cat((x1, x_next), dim=1) # [bxa, ts, C, H, W]
        else:
            x_ind = x1
        # print(x_ind.size())
        assert x_ind.size(1) == self.time_stamp
        x_ind = x_ind.reshape(BA*self.time_stamp, C, H, W)
        # embed patches
        x_ind = self.patch_embed(x_ind)
        x_ind = x_ind + self.pos_embed[:, 1:, :]

        # mask before transformer encoding
        # amortized masking, complement and random
        x_masked, mask, ids_restore = self.masking_handle(x_ind, mask_ratio)
        # print(x_masked.size())
        # print(mask.size())
        for blk in self.blocks:
            x_masked = blk(x_masked)
        x = self.norm(x_masked)
        x = self.compressor(x)

        # # -------- encoder then mask ----------
        # # apply Transformer blocks
        # for blk in self.blocks:
        #     x_ind = blk(x_ind)
        # x = self.norm(x_ind)
        # # compress for communication
        # # mask ONLY for transmission, encode the complete sequence
        # x_masked, mask, ids_restore = self.masking_handle(x, mask_ratio)
        # x = self.compressor(x_masked)
        # # --------------------------------------

        return x, mask, ids_restore

    def forward_decoder(self, latent, mask, ids_restore):
        """
        overwrite the original forward_decoder, now input latent are fused
        """
        # print("to decoder, latent", latent.size())
        latent = self.decompressor(latent)
        # # embed tokens
        latent = self.decoder_embed(latent)

        restored_latent = self.unmasking_handle(latent, mask, ids_restore)
        restored_latent = restored_latent.reshape(restored_latent.size(0), self.patch_h*self.patch_w, -1)
        # x = torch.cat([latent_cls, restored_latent], dim=1)

        # add pos embed
        x = restored_latent + self.decoder_pos_embed[:, 1:, :]

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # VERSION mlp -----
        x_occ = self.decoder_pred_occ(x) 
        x_free = self.decoder_pred_free(x)
        x_occ = self.unpatchify(x_occ)
        x_free = self.unpatchify(x_free)
        # remove cls token
        # x = x[:, 1:, :]
        # print("pred size", x.size())
        # --------------------
        x_pred = torch.stack((x_free, x_occ), dim=1) # [B, class, C, H, W]
        return x_pred

    def forward_loss(self, teacher, pred):
        """
        overwrite the original forward_loss, now calculate loss on the entire image

        """
        target = self.patchify(teacher)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2 ## L2 loss
        loss = loss.sum(dim=-1) # [N, L], sum loss per patch
        loss = loss.mean()
        return loss

    def forward_bce_loss(self, target, pred):
        target = target.type(torch.LongTensor).to(pred.device)
        loss = self.cls_loss(pred, target)
        return loss

    def forward_focal_loss(self, target, pred):
        return self.focal_loss(pred, target)

    def forward(self, imgs1, imgs_next, teacher, trans_matrices, num_agent_tensor, batch_size, mask_ratio=0.75):
        """
        Encoder encodes each timestamp alone
        Decoder fuses multi timestamp together
        """
        p = self.patch_embed.patch_size[0]
        self.patch_h = self.patch_w = imgs1.shape[2]//p
        # print("image, patch h, w", imgs1.shape, self.patch_h, self.patch_w)
        latent, mask, ids_restore = self.forward_encoder(imgs1, imgs_next, mask_ratio)
        # latent: [BAT, L, D]
        # print("to vq, latent", latent.size())
        vq_loss, quantized, perplexity, encodings = self._vq_star(latent)
        pred = self.forward_decoder(quantized, mask, ids_restore)
        # loss = self.forward_loss(imgs1, pred)
        recon_loss = self.forward_bce_loss(imgs1, pred)
        # recon_loss = self.forward_focal_loss(imgs1, pred)
        # result = self.unpatchify(pred)
        ind_result = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        result = self.late_fusion(ind_result, trans_matrices, num_agent_tensor, batch_size)
        # print(result.size())
        # ind_result = self.unpatchify(pred)
        loss = vq_loss + recon_loss
        return loss, result, mask, ind_result, perplexity

    def inference(self, imgs1, imgs_next, teacher, trans_matrices, num_agent_tensor, batch_size, mask_ratio=0.75):
        """
        Encoder encodes each timestamp alone
        Decoder fuses multi timestamp together
        """
        p = self.patch_embed.patch_size[0]
        self.patch_h = self.patch_w = imgs1.shape[2]//p
        # print("image, patch h, w", imgs1.shape, self.patch_h, self.patch_w)
        latent, mask, ids_restore = self.forward_encoder(imgs1, imgs_next, mask_ratio)
        # latent: [BAT, L, D]
        vq_loss, quantized, perplexity, encodings = self._vq_star(latent)
        pred = self.forward_decoder(quantized, mask, ids_restore)
        # loss = self.forward_loss(imgs1, pred)
        # loss = self.forward_bce_loss(imgs1, pred)
        recon_loss = self.forward_focal_loss(imgs1, pred)
        # print(imgs1.size(), imgs1.type())
        ind_result = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        result = self.ego_late_fusion(ind_result, imgs1, trans_matrices, num_agent_tensor, batch_size)
        # result = self.ego_late_fusion(imgs1, imgs1, trans_matrices, num_agent_tensor, batch_size)
        # print(result.size())
        # ind_result = self.unpatchify(pred)
        loss = recon_loss + vq_loss
        return loss, result, mask, ind_result, perplexity


def vq_amo_individual_star_patch8_dec256d4b(**kwargs):
    model = VQSTARViT(
        img_size=256, patch_size=8, in_chans=13, embed_dim=384, depth=6, num_heads=12,
        decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_head="mlp", **kwargs)
    return model

vqstar = vq_amo_individual_star_patch8_dec256d4b
