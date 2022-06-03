# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------


# modified MAE to be multiagent MAE
# Compared to the MAE and spatial temporal MAE, three modifications are made:
# 1) a compression auto encoder module is added at the end of the encoder (compress) and 
#    at the begnining of the decoder (decompress)
# 2) enable multiagent style
# 3) Loss changed from mean to sum
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block
from coperception.models.mae_base import *
from coperception.utils.maeutil.pos_embed import get_2d_sincos_pos_embed



class FusionMultiAgentMAEViT(MultiAgentMaskedAutoencoderViT):
    """
    joint reconstruction
    Serves as the DetModel, handles fusion/communication between encoder and decoder
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_head="conv3", norm_pix_loss=False, time_stamp=1, mask_method="random"):
        super(FusionMultiAgentMAEViT, self).__init__(
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
                mask_method = mask_method
                )
        # for neighbor agents' features
        self.patch_h = 0
        self.patch_w = 0
        # self.num_agent = 0
        # self.neighbor_feat_list = []
        # self.tg_agent = None    

    def forward_fusion(self, x, ids_restore, trans_matrices, num_agent_tensor, batch_size):
        """
        as the pre process of the decoder, handles:
        1) add mask, restore feature maps shape like (b x agent x channel x h x w)
        2) communication/fushion/aggregation
        3) ready for decoder
        """
        device = x.device
        # # decompress
        x = self.decompressor(x)
        # # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # print("x_", x_.size())
        ## --- check fusion (communication) --- 
        # x_ = self.patchify(x) # here x is input image, directly
        
        # x_: (B, seq, chns)
        ## -- reshape back to 256x256 ---
        # feature_maps = self.unpatchify(x_)
        ## -------------
        ## --- reshape into B C H W ---
        feature_maps = x_.reshape(x_.shape[0], self.patch_h, self.patch_w, x_.shape[-1]) # (B, h, w, chns)
        feature_maps = feature_maps.permute(0, 3, 1, 2) # (B, chns, h, w)
        # print("feature map", feature_maps.size())
        # -------
        ## --- do fusion ---
        size = self.get_feature_maps_size(feature_maps)
        # print(size)
        assert feature_maps.size(0) % batch_size == 0, (feature_maps.size(), batch_size)
        self.num_agent = feature_maps.size(0) // batch_size
        feat_list = self.build_feature_list(batch_size, feature_maps)
        # [[1,1,256,32,32]x5] NOTE should it be [[B, 1, 256, 32, 32]x5]?
        # print(feat_list)
        local_com_mat = self.build_local_communication_matrix(
            feat_list)  # [2 5 512 32 32] [batch, agent, channel, height, width]
        # # FIXME: check size
        # print("local com mat size", local_com_mat.shape) #[2,5,256,32,32]
        local_com_mat_update = self.build_local_communication_matrix(feat_list)  # to avoid the inplace operation

        for b in range(batch_size):
            self.num_agent = num_agent_tensor[b, 0]
            for i in range(self.num_agent):
                self.tg_agent = local_com_mat[b, i]
                # print("tg agent shape", self.tg_agent.shape) #[256,32,32] 
                self.neighbor_feat_list = []
                self.neighbor_feat_list.append(self.tg_agent)
                all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]
                # print(all_warp.shape)[5,4,4]
                self.build_neighbors_feature_list(b, i, all_warp, self.num_agent, local_com_mat,
                                                    device, size)

                # feature update
                # torch.save(torch.stack(self.neighbor_feat_list).detach().cpu(), "/mnt/NAS/home/zjx/Masked-Multiagent-Autoencoder/debug/nbf-{}-{}.pt".format(b, i))
                local_com_mat_update[b, i] = self.fusion() 
        
        # weighted feature maps is passed to decoder
        feat_fuse_mat = self.agents_to_batch(local_com_mat_update)
        # print('feat fuse mat size', feat_fuse_mat.size())
        
        
        # shape: [B, chns, h, w], B = batch_size x num_agent
        # reshape back to NxLxD
        real_bs = feat_fuse_mat.size(0)
        chns = feat_fuse_mat.size(1)
        fused_latent = feat_fuse_mat.permute(0,2,3,1).reshape(real_bs, self.patch_h*self.patch_w, chns)
        # --- re patchify ---
        # fused_latent = self.patchify(feat_fuse_mat)
        # --------------------
        # print("fused latent size", fused_latent.size()) 
        # NxLxD
        fused_latent = torch.cat([x[:, :1, :], fused_latent], dim=1)  # append cls token
        return fused_latent

    def forward_encoder(self, x1, x_next, mask_ratio):
        """
        x_next: [bxa, ts-1, C, H, W] beq_next_frames
        """
        # embed patches
        x1 = self.patch_embed(x1)      
        # add pos embed w/o cls token
        x1 = x1 + self.pos_embed[:, 1:, :] + self.temp_embed[:, 0, :]

        # handles other time stamps
        xs = []
        for ts in range(self.time_stamp-1):
            xt = x_next[:, ts, :, :, :] # [Bxa, C, H, W]
            # print("xt size", xt.size())
            xt = self.patch_embed(xt) + self.pos_embed[:, 1:, :] + self.temp_embed[:, ts+1, :]
            xs.append(xt)

        # masking: length -> length * mask_ratio
        # x_masked, x1len, mask1, ids_restore1 = self.more_random_masking(x1, xs, mask_ratio)
        # complement masking
        # x_masked, x1len, mask1, ids_restore1 = self.complement_masking(x1, xs, mask_ratio)
        x_masked, x1len, mask1, ids_restore1 = self.masking_handle(x1, xs, mask_ratio)
        # print(x_masked.size())

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # compress for communication
        x = self.compressor(x)

        return x, mask1, ids_restore1, x1len

    def forward_decoder(self, x):
        """
        overwrite the original forward_decoder, now input latent are fused
        """
        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # # predictor projection
        # x = self.decoder_pred(x)

        # # remove cls token
        # x = x[:, 1:, :]

        # # VERSION immediate input ---
        # x = x + self.decoder_pos_embed
        # x = self.decoder_pred(x)
        # x = x[:, 1:, :]
        # # ---------------------------

        # VERSION conv -----
        x = x[:, 1:, :]
        x = x.reshape(x.shape[0], self.patch_h, self.patch_w, x.shape[-1]) # (B, h, w, chns)
        x = x.permute(0, 3, 1, 2).contiguous() # (B, chns, h, w)
        # print("before pred size", x.size())
        x = self.decoder_pred(x)
        # patchify back to accomodate
        x = self.patchify(x)
        # print("pred size", x.size())
        # --------------------
        return x

    def forward_loss(self, teacher, pred, mask):
        """
        overwrite the original forward_loss, now calculate loss on the entire image

        """
        target = self.patchify(teacher)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2 ## L2 loss
        # loss = self.L1_Loss(pred, target) ## L1 loss
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.sum(dim=-1) # [N, L], sum loss per patch
        loss = loss.mean()
        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss


    def forward(self, imgs1, imgs_next, teacher, trans_matrices, num_agent_tensor, batch_size, mask_ratio=0.75):
        """
        Modified from the original forward, make fusion happen
        """
        p = self.patch_embed.patch_size[0]
        self.patch_h = self.patch_w = imgs1.shape[2]//p
        
        latent, mask1, ids_restore, size1 = self.forward_encoder(imgs1, imgs_next, mask_ratio)
        # latent: [Bxa, L, D]
        # now we only consider reconstruct the first frame
        # this can be extended to reconstruct both
        latent_to_decode = latent[:, :1+size1, :] # CLS(0) + timesamp t (size1)
        fused_latent = self.forward_fusion(latent_to_decode, ids_restore, trans_matrices, num_agent_tensor, batch_size)
        pred = self.forward_decoder(fused_latent)  # [N, L, p*p*3]

        # --- check communication ---
        # fused_latent = self.forward_fusion(imgs1, ids_restore, trans_matrices, num_agent_tensor, batch_size)
        # pred = fused_latent # cls token is not included in forward fusion
        #----------------------------
        loss = self.forward_loss(teacher, pred, mask1)
        # loss = self.forward_loss(imgs1, pred, mask1) # use single view as supervision
        # what to do with the masking? now it is fused. So not exactly mask1
        # ONE solution: just calculate the loss over the entire input
        return loss, pred, mask1, fused_latent[:,1:,:] # remove cls
    

class IndivMultiAgentMAEViT(MultiAgentMaskedAutoencoderViT):
    """
    individual reconstruction
    Serves as the DetModel, handles fusion/communication after encoder and decoder
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_head="mlp", norm_pix_loss=False, time_stamp=1, mask_method="random"):
        super(IndivMultiAgentMAEViT, self).__init__(
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
                mask_method = mask_method)
        # for neighbor agents' features
        self.patch_h = 0
        self.patch_w = 0

    def late_fusion(self, pred, trans_matrices, num_agent_tensor, batch_size):
        """ reshape the model's predictions back to 256x256x13, do aggregation on this"""
        device = pred.device
        indiv_imgs = self.unpatchify(pred) # [B C H W]
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
                local_com_mat_update[b, i] = self.fusion() 
        
        # weighted feature maps is passed to decoder
        fused_images = self.agents_to_batch(local_com_mat_update)
        # print('feat fuse mat size', feat_fuse_mat.size())
        return fused_images


    def forward(self, imgs1, imgs_next, teacher, trans_matrices, num_agent_tensor, batch_size, mask_ratio=0.7):
        latent, mask1, ids_restore, size1 = self.forward_encoder(imgs1, imgs_next, mask_ratio)
        # now we only consider reconstruct the first frame
        # this can be extended to reconstruct both
        latent_to_decode = latent[:, :1+size1, :]
        pred = self.forward_decoder(latent_to_decode, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs1, pred, mask1)
        # result = self.unpatchify(pred)
        result = self.late_fusion(pred, trans_matrices, num_agent_tensor, batch_size)
        return loss, result, mask1, pred

class AmortizedFushionMMAEViT(MultiAgentMaskedAutoencoderViT):
    """
    Temporal amorized version
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_head="conv3", norm_pix_loss=False, time_stamp=1, mask_method="random"):
        super(AmortizedFushionMMAEViT, self).__init__(
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
                mask_method = mask_method)
        # for neighbor agents' features
        self.patch_h = 0
        self.patch_w = 0

    def amortized_random_masking(self, x_ts, )

    def forward_encoder(self, x1, x_next, mask_ratio):
        """
        x1: [bxa, C, H, W]
        x_next: [bxa, ts-1, C, H, W] beq_next_frames
        """
        # cat x1 and x_next to encoder independently
        BA, C, H, W = x1.size()
        x1 = x1.unsqueeze(1)
        x_ind = torch.cat((x1, x_next), dim=1) # [bxa, ts, C, H, W]
        # print(x_ind.size())
        assert x_ind.size(1) == self.time_stamp
        x_ind = x_ind.reshape(BA*self.time_stamp, C, H, W)
        # embed patches
        x_ind = self.patch_embed(x_ind)
        x_ind = x_ind + self.pos_embed[:, 1:, :]

        # amortized masking, complement and random

        # masking: length -> length * mask_ratio
        # x_masked, x1len, mask1, ids_restore1 = self.more_random_masking(x1, xs, mask_ratio)
        # complement masking
        # x_masked, x1len, mask1, ids_restore1 = self.complement_masking(x1, xs, mask_ratio)
        x_masked, x1len, mask1, ids_restore1 = self.masking_handle(x1, xs, mask_ratio)
        # print(x_masked.size())

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x_masked), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # compress for communication
        x = self.compressor(x)

        return x, mask1, ids_restore1, x1len

    def forward(self, imgs1, imgs_next, teacher, trans_matrices, num_agent_tensor, batch_size, mask_ratio=0.75):
        """
        Encoder encodes each timestamp alone
        Decoder fuses multi timestamp together
        """
        p = self.patch_embed.patch_size[0]
        self.patch_h = self.patch_w = imgs1.shape[2]//p
        # TODO: -----new model----
        # encoder handles each timestamp seperately, decoder fuse them together. 
        # temporal embedding add at the decoder side
        # special take care of the mask ids
        # implement a new model to do so.
        # a) complement masking: no mask tokens
        # b) random masking: fuse timestamps, [mask] fills the rest
        # ------------------------
        latent, mask1, ids_restore, size1 = self.forward_encoder(imgs1, imgs_next, mask_ratio)

    
def individual_bev_multi_mae_vit_base_patch8_dec512d8b(**kwargs):
    model = IndivMultiAgentMAEViT(
        img_size=256, patch_size=8, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_head="mlp", **kwargs)
    return model

def fusion_bev_multi_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = FusionMultiAgentMAEViT(
        img_size=256, patch_size=16, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_head="conv3", **kwargs)
    return model

def fusion_bev_multi_mae_vit_base_patch8_dec512d8b(**kwargs):
    model = FusionMultiAgentMAEViT(
        img_size=256, patch_size=8, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_head="conv3", **kwargs)
    return model

def fusion_bev_multi_mae_vit_base_patch1_dec512d8b(**kwargs):
    model = FusionMultiAgentMAEViT(
        img_size=256, patch_size=1, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def fusion_bev_multi_mae_vit_base_patch32_dec512d8b(**kwargs):
    model = FusionMultiAgentMAEViT(
        img_size=256, patch_size=32, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_head="conv3", **kwargs)
    return model

def bev_multi_mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MultiAgentMaskedAutoencoderViT(
        img_size=256, patch_size=16, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def fusion_bev_multi_mae_vit_base_patch16_dec1024d8b(**kwargs):
    model = FusionMultiAgentMAEViT(
        img_size=256, patch_size=16, in_chans=13, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=1024, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), decoder_head="conv3", **kwargs)
    return model


# Juexiao add
bev_mae_vit_base_patch16 = bev_multi_mae_vit_base_patch16_dec512d8b # decoder: 512 dim, 8 blocks
# joint reconstruction
joint_bev_mae_vit_base_patch16 = fusion_bev_multi_mae_vit_base_patch16_dec512d8b
joint_bev_mae_vit_base_patch8 = fusion_bev_multi_mae_vit_base_patch8_dec512d8b
joint_bev_mae_vit_base_patch1 = fusion_bev_multi_mae_vit_base_patch1_dec512d8b
joint_bev_mae_vit_base_patch32 = fusion_bev_multi_mae_vit_base_patch32_dec512d8b
joint_bev_mae_vit_base_patch16_dec1024 = fusion_bev_multi_mae_vit_base_patch16_dec1024d8b
# individual reconstruction
ind_bev_mae_vit_base_patch8 = individual_bev_multi_mae_vit_base_patch8_dec512d8b