#mae base code

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from coperception.utils.maeutil.pos_embed import get_2d_sincos_pos_embed

class SegmentationHead(nn.Module):
  '''
  3D Segmentation heads to retrieve semantic segmentation at each scale.
  Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
  '''
  def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
    super().__init__()

    # First convolution
    self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

    # ASPP Block
    self.conv_list = dilations_conv_list
    self.conv1 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn1 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.conv2 = nn.ModuleList(
      [nn.Conv3d(planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False) for dil in dilations_conv_list])
    self.bn2 = nn.ModuleList([nn.BatchNorm3d(planes) for dil in dilations_conv_list])
    self.relu = nn.ReLU(inplace=True)

    # Convolution for output
    self.conv_classes = nn.Conv3d(planes, nbr_classes, kernel_size=3, padding=1, stride=1)

  def forward(self, x_in):

    # Dimension exapension
    x_in = x_in[:, None, :, :, :]

    # Convolution to go from inplanes to planes features...
    x_in = self.relu(self.conv0(x_in))

    y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
    for i in range(1, len(self.conv_list)):
      y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
    x_in = self.relu(y + x_in)  # modified

    x_in = self.conv_classes(x_in)

    return x_in

class ConvPred(nn.Module):
    """
    a upsampling and deconve module in repalce of the original final linear projection,
    to obtain the prediction of the original image.
    for patch size 8x8, feature map size 32x32x512
    """
    def __init__(self, input_size=32, output_size=256, input_chans=512, output_chans=13):
        super().__init__()
        # 64x64x512 -> 64x64x256
        chans1 = input_chans//2
        self.conv1 = nn.Conv2d(input_chans, chans1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(chans1)
        # 128x128x256 -> 128x128x128
        chans2 = chans1//2
        self.conv2 = nn.Conv2d(chans1, chans2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(chans2)
        # 256x256x128 -> 2256x256x13
        self.conv3 = nn.Conv2d(chans2, output_chans, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d(chans2, output_chans, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(output_chans)

        # self.cls_head = SegmentationHead(inplanes=1, planes=8, nbr_classes=2, dilations_conv_list=[1, 2, 3])

    def forward(self, x):
        # ---- patch size 8x8 --- will have 32x32x512 ----
        # upsampling 32x32x512 -> 64x64x512
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # channel compression 512 -> 256
        x = F.relu(self.bn1(self.conv1(x)))
        # upsamping 64x64x256 -> 128x128x256
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # channel compression 256 -> 128
        x = F.relu(self.bn2(self.conv2(x)))
        # upsamping 128x128x128 -> 256x256x128
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # channel compression 128 -> 13
        x = self.bn3(self.conv3(x))
        # x = torch.sigmoid(self.bn3(self.conv3(x)))
        # x = self.cls_head(x) #[B, 2, C, H, W]
        return x

class ConvPred16(nn.Module):
    """
    a upsampling and deconve module in repalce of the original final linear projection,
    to obtain the prediction of the original image.
    for patch size 16x16, feature map size 16x16x512
    """
    def __init__(self, input_size=16, output_size=256, input_chans=512, output_chans=13):
        super().__init__()
        # 32x32x512 -> 32x32x256
        chans1 = input_chans//2
        self.conv1 = nn.Conv2d(input_chans, chans1, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(chans1)
        # 64x64x256 -> 64x64x128
        chans2 = chans1//2
        self.conv2 = nn.Conv2d(chans1, chans2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(chans2)
        # 128x128x128 -> 128x128x64
        chans3 = chans2//2
        self.conv3 = nn.Conv2d(chans2, chans3, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(chans3)
        # 256x256x64 -> 256x256x13
        self.conv4 = nn.Conv2d(chans3, output_chans, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(output_chans)

    def forward(self, x):
        # ---- 16x16x512 ------
        # upsampling 16x16x512 -> 32x32x512
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # print("up 1", x.size())
        # channel compression 512 -> 256
        x = F.relu(self.bn1(self.conv1(x)))
        # upsamping 32x32x256 -> 64x64x256
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # print("up 2", x.size())
        # channel compression 256->128
        x = F.relu(self.bn2(self.conv2(x)))
        # upsamping 64x64x128 -> 128x128x128
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # print("up 3", x.size())
        # channel compression 128->64
        x = F.relu(self.bn3(self.conv3(x)))
        # upsamping 128x128x64 -> 256x256x64
        x = F.interpolate(x, scale_factor=2, mode='bilinear') #, align_corners=True)
        # print("up 4", x.size())
        # channel compression 64->13
        # x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn4(self.conv4(x))
        # ----------------------------
        return x

class STARVectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(STARVectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        # print("vq inputs", inputs.size())
        input_shape = inputs.shape
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        # print("encoding_indices", encoding_indices)
        # print("encoding_indices", encoding_indices.size())
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        # print(avg_probs)
        # print("avg_probs", avg_probs.size())
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings


class STARVectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(STARVectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        # inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings



class MultiAgentMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, decoder_head="mlp", norm_pix_loss=False, time_stamp=1, mask_method="random",
                 inter_emb_dim=32):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.time_stamp = time_stamp
        # temporal embeddings
        self.temp_embed = nn.Parameter(torch.zeros(1, self.time_stamp, embed_dim)) # learnable temporal embeddings for 2 time stamp
        self.decoder_temp_embed = nn.Parameter(torch.zeros(1, self.time_stamp, 1, decoder_embed_dim)) # learnable temporal embeddings for 2 time stamp

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        self.compressor = nn.Sequential(
            nn.Linear(embed_dim, inter_emb_dim),
            nn.ReLU(),
        )
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decompressor = nn.Linear(inter_emb_dim, embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        # try option1:
        # self.decoder_embed = nn.Linear(inter_emb_dim, decoder_embed_dim, bias=True)
        # try option2:
        # self.decoder_embed = nn.Sequential(
        #     nn.Linear(inter_emb_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        if decoder_head == "mlp":
            # self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
            self.decoder_pred_occ = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
            self.decoder_pred_free = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)
        # conv decoder pred
        elif decoder_head == "conv3":
            if patch_size == 8:
                self.decoder_pred_occ = ConvPred(input_size=32, output_size=256, input_chans=512, output_chans=13)
                self.decoder_pred_free = ConvPred(input_size=32, output_size=256, input_chans=512, output_chans=13)
            elif patch_size == 16:
                self.decoder_pred_occ = ConvPred16(input_size=16, output_size=256, input_chans=512, output_chans=13)
                self.decoder_pred_free = ConvPred16(input_size=16, output_size=256, input_chans=512, output_chans=13)
            else:
                raise NotImplementedError("not supported conv head for patch size", patch_size)
        else:
            raise NotImplementedError("decoder head", decoder_head)
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.in_chans = in_chans
        self.L1_Loss = nn.L1Loss(reduction="none")
        

        self.initialize_weights()

        # ----- fushion support ------
        self.num_agent = 0
        self.neighbor_feat_list = []
        self.tg_agent = None

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # temporal embedding initialized likewise
        torch.nn.init.normal_(self.temp_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, chs, H, W)
        x: (N, L, patch_size**2 *chns)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        num_chans = imgs.size(1)

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], num_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * num_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *chans)
        imgs: (N, chans, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        num_chans = x.size(-1)//(p**2)
        assert p**2 * num_chans == x.size(-1)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, num_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], num_chans, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    # x_masked, x1len, mask1, ids_restore1 = self.more_random_masking(x1, xs, mask_ratio)
    def more_random_masking(self, x1, xs, mask_ratio):
        """
        random masking each time stamp's data and cat
        """
        assert isinstance(xs, list)
        xnum = len(xs)+1
        x_masked = []

        x1_masked, mask1, ids_restore1 = self.random_masking(x1, mask_ratio)
        x1len = x1_masked.shape[1]
        for idx in range(xnum-1):
            xt_masked, _, _ = self.random_masking(xs[idx], mask_ratio)
            x_masked.append(xt_masked)

        # x2, mask2, ids_restore2 = self.random_masking(x2, mask_ratio)
        x_masked = torch.cat(x_masked, dim=1)
        return x_masked, x1len, mask1, ids_restore1

    def complement_masking(self, x1, x2, mask_ratio):
        """
        mask x1 with mask_ratio
        mask x2 with 1 - mask ratio
        so that they are complementary
        x1, x2: same size [N, D, L] 
        """
        N, L, D = x1.shape  # batch, length, dim
        len_keep1 = int(L * (1 - mask_ratio))
        len_keep2 = L - len_keep1
        noise = torch.rand(N, L, device=x1.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep1 = ids_shuffle[:, :len_keep1]
        ids_keep2 = ids_shuffle[:, len_keep1:]
        assert ids_keep2.size(1) == len_keep2

        x1_masked = torch.gather(x1, dim=1, index=ids_keep1.unsqueeze(-1).repeat(1, 1, D))
        x2_masked = torch.gather(x2, dim=1, index=ids_keep2.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask1 = torch.ones([N, L], device=x1.device)
        mask1[:, :len_keep1] = 0
        mask2 = torch.ones([N, L], device=x2.device)
        mask2[:, len_keep1:] = 0
        # unshuffle to get the binary mask
        mask1 = torch.gather(mask1, dim=1, index=ids_restore)
        mask2 = torch.gather(mask2, dim=1, index=ids_restore)

        return x1_masked, x2_masked, mask1, mask2, ids_restore

    def more_complement_masking(self, x1, xs, mask_ratio):
        """
        masking is conducted complementary
        x1 is the current time stamp image
        xs is a list of images for the other time stamps
        x1: [N, D, L] 
        xs: a list (ts-1) x [N, D, L]
        """
        assert isinstance(xs, list)
        xnum = len(xs)+1
        N, L, D = x1.shape  # batch, length, dim
        device = x1.device
        
        noise = torch.rand(N, L, device=x1.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the corresponding subset
        len_keep1 = int(L * (1 - mask_ratio))
        ids_keep1 = ids_shuffle[:, :len_keep1]
        mask1 = torch.ones([N, L], device=device)
        mask1[:, :len_keep1] = 0
        mask1 = torch.gather(mask1, dim=1, index=ids_restore)
        # for the rest time stamps
        ids_keeps = []
        masks = [] # binray masks, 0 is keep, 1 is remove
        if xnum==2:
            len_keep2 = L - len_keep1
            ids_keep2 = ids_shuffle[:, len_keep1:]
            assert ids_keep2.size(1) == len_keep2
            # ids_keeps.append(ids_keep1)
            ids_keeps.append(ids_keep2)
            # # generate the binary mask: 0 is keep, 1 is remove
            # mask2 = torch.ones([N, L], device=device)
            # mask2[:, len_keep1:] = 0
            # # unshuffle to get the binary mask
            # mask2 = torch.gather(mask2, dim=1, index=ids_restore)
            # masks.append(mask2)
        elif xnum == 3:
            len_keep2 = int(L * (1 - mask_ratio))
            len_keep3 = L - len_keep1 - len_keep2
            ids_keep2 = ids_shuffle[:, len_keep1: len_keep1+len_keep2]
            ids_keep3 = ids_shuffle[:, len_keep1+len_keep2 :]
            assert ids_keep2.size(1) == len_keep2
            assert ids_keep3.size(1) == len_keep3
            ids_keeps.append(ids_keep2)
            ids_keeps.append(ids_keep3)
            # generate the binary mask: 0 is keep, 1 is remove
            # mask2 = torch.ones([N, L], device=device)
            # mask2[:, len_keep1: len_keep1+len_keep2] = 0
            # mask3 = torch.ones([N, L], device=device)
            # mask3[:, len_keep1+len_keep2 :] = 0
            # # unshuffle to get the binary mask
            # mask2 = torch.gather(mask2, dim=1, index=ids_restore)
            # mask3 = torch.gather(mask3, dim=1, index=ids_restore)
            # masks.append(mask2)
            # masks.append(mask3)
        else:
            raise NotImplementedError

        x_masked = []
        x1_masked = torch.gather(x1, dim=1, index=ids_keep1.unsqueeze(-1).repeat(1, 1, D))
        x1_len = x1_masked.size(1)
        x_masked.append(x1_masked)
        # masks = []
        for idx in range(xnum-1):
            x_masked.append(torch.gather(xs[idx], dim=1, index=ids_keeps[idx].unsqueeze(-1).repeat(1, 1, D)))
            # mask_per = torch.ones([N, L], device=x1.device)
            # mask_per[:, :len_keep1] = 0

        x_masked = torch.cat(x_masked, dim=1)
        return x_masked, x1_len, mask1, ids_restore

    def get_feature_maps_size(self, feature_maps: tuple):
        size = list(feature_maps.shape)
        # NOTE: batch size will change the shape[0]. We need to manually set it to 1.
        size[0] = 1
        size = tuple(size)
        return size

    # get feat maps for each agent [10 512 32 32] -> [2 5 512 32 32]
    def build_feature_list(self, batch_size: int, feat_maps: dict) -> list:
        feature_map = {}
        # [5,256,32,32]
        feature_list = []

        for i in range(self.num_agent):
            feature_map[i] = torch.unsqueeze(feat_maps[batch_size * i:batch_size * (i + 1)], 1)
            # feature_map[i]: [B,1,256,32,32]
            feature_list.append(feature_map[i])

        return feature_list

    # [2 5 512 16 16] [batch, agent, channel, height, width]
    @staticmethod
    def build_local_communication_matrix(feature_list: list):
        return torch.cat(tuple(feature_list), 1)

    @staticmethod
    # FIXME: rename 'j'
    def feature_transformation(b, nb_agent_idx, local_com_mat, all_warp, device, size):
        nb_agent = torch.unsqueeze(local_com_mat[b, nb_agent_idx], 0)  # [1 512 16 16]
        nb_warp = all_warp[nb_agent_idx]  # [4 4]
        # normalize the translation vector
        x_trans = (4 * nb_warp[0, 3]) / 128
        y_trans = -(4 * nb_warp[1, 3]) / 128

        theta_rot = torch.tensor(
            [[nb_warp[0, 0], nb_warp[0, 1], 0.0], [nb_warp[1, 0], nb_warp[1, 1], 0.0]]).type(
            dtype=torch.float).to(device)
        theta_rot = torch.unsqueeze(theta_rot, 0)
        grid_rot = F.affine_grid(theta_rot, size=torch.Size(size))  # get grid for grid sample

        theta_trans = torch.tensor([[1.0, 0.0, x_trans], [0.0, 1.0, y_trans]]).type(dtype=torch.float).to(
            device)
        theta_trans = torch.unsqueeze(theta_trans, 0)
        grid_trans = F.affine_grid(theta_trans, size=torch.Size(size))  # get grid for grid sample

        # first rotate the feature map, then translate it
        warp_feat_rot = F.grid_sample(nb_agent, grid_rot, mode='nearest')
        warp_feat_trans = F.grid_sample(warp_feat_rot, grid_trans, mode='nearest')
        # FIXME: check dim
        # print("warp feature size", warp_feat_trans.size())
        return torch.squeeze(warp_feat_trans, dim=0) # [512, 16, 16]


    def build_neighbors_feature_list(self, b, agent_idx, all_warp, num_agent, local_com_mat, device,
                                     size) -> None:
        for j in range(num_agent):
            if j != agent_idx:
                warp_feat = self.feature_transformation(b, j, local_com_mat, all_warp, device, size)
                self.neighbor_feat_list.append(warp_feat)
        
    def fusion(self, mode="sum"):
        # add all the features point wise
        # print("len of nb feat list", len(self.neighbor_feat_list))
        fused = torch.sum(torch.stack(self.neighbor_feat_list), dim=0)
        if mode == "union":
            fused[fused>0.] = 1.0 
        return fused 

    # Question: this can be done by a view or reshape?
    # shaped like: [ (batch_agent1, batch_agent2, ...)  , channel, h, w]
    @staticmethod
    def agents_to_batch(feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

    def forward_encoder(self, x1, x_next, mask_ratio):
        # embed patches
        x1 = self.patch_embed(x1)
        # x2 = self.patch_embed(x2)
        

        # add pos embed w/o cls token
        x1 = x1 + self.pos_embed[:, 1:, :] + self.temp_embed[:, 0, :]

        # handles other time stamps
        xs = []
        for ts in range(self.time_stamp-1):
            xt = x_next[:, ts, :, :, :] # [Bxa, C, H, W]
            # print("xt size", xt.size())
            xt = self.patch_embed(xt) + self.pos_embed[:, 1:, :] + self.temp_embed[:, ts+1, :]
            xs.append(xt)
        # x2 = x2 + self.pos_embed[:, 1:, :] + self.temp_embed[:, 1, :]

        # masking: length -> length * mask_ratio
        # x1, mask1, ids_restore1 = self.random_masking(x1, mask_ratio)
        # x2, mask2, ids_restore2 = self.random_masking(x2, mask_ratio)
        # x_masked, x1len, mask1, ids_restore1 = self.more_random_masking(x1, xs, mask_ratio)
        # complement masking
        # x_masked, x1len, mask1, ids_restore1 = self.complement_masking(x1, xs, mask_ratio)
        x_masked, x1len, mask1, ids_restore1 = self.masking_handle(x1, xs, mask_ratio)
        # print(x_masked.size())
        # print(x1len)

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

    def forward_decoder(self, x, ids_restore):
        # decompress
        x = self.decompressor(x)
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # if want to reconstruct the second time stemp
        # cat mask_tokens to the front and then unshuffle

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # # binarize pred, because target is binary
        # pred[pred>=0.5] = 1.0
        # pred[pred<0.5] = 0.0
        loss = (pred - target) ** 2 ## L2 loss
        # loss = self.L1_Loss(pred, target) ## L1 loss
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.sum(dim=-1) # [N, L], sum loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs1, imgs2, mask_ratio=0.75):
        latent, mask1, ids_restore, size1 = self.forward_encoder(imgs1, imgs2, mask_ratio)
        # now we only consider reconstruct the first frame
        # this can be extended to reconstruct both
        latent_to_decode = latent[:, :1+size1, :]
        pred = self.forward_decoder(latent_to_decode, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs1, pred, mask1)
        result = self.unpatchify(pred)
        return loss, pred, mask1, result