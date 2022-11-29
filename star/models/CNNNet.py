import torch
import torch.nn.functional as F
import torch.nn as nn

class Backbone(nn.Module):
    """The backbone class that contains encode and decode function"""

    def __init__(self, height_feat_size, compress_level=0, train_completion=False):
        super().__init__()
        self.conv_pre_1 = nn.Conv2d(
            height_feat_size, 32, kernel_size=3, stride=1, padding=1
        )
        self.conv_pre_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn_pre_1 = nn.BatchNorm2d(32)
        self.bn_pre_2 = nn.BatchNorm2d(32)

        self.conv3d_1 = Conv3D(
            64, 64, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)
        )
        self.conv3d_2 = Conv3D(
            128, 128, kernel_size=(1, 1, 1), stride=1, padding=(0, 0, 0)
        )

        self.conv1_1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv6_1 = nn.Conv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv7_1 = nn.Conv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.conv7_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv8_1 = nn.Conv2d(64 + 32, 32, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.bn1_1 = nn.BatchNorm2d(64)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.bn2_1 = nn.BatchNorm2d(128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.bn3_1 = nn.BatchNorm2d(256)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.bn4_1 = nn.BatchNorm2d(512)
        self.bn4_2 = nn.BatchNorm2d(512)

        self.bn5_1 = nn.BatchNorm2d(256)
        self.bn5_2 = nn.BatchNorm2d(256)

        self.bn6_1 = nn.BatchNorm2d(128)
        self.bn6_2 = nn.BatchNorm2d(128)

        self.bn7_1 = nn.BatchNorm2d(64)
        self.bn7_2 = nn.BatchNorm2d(64)

        self.bn8_1 = nn.BatchNorm2d(32)
        self.bn8_2 = nn.BatchNorm2d(32)

        self.train_completion = train_completion
        if train_completion:
            self.conv9_1 = nn.Conv2d(32, 13, kernel_size=1, stride=1, padding=0)
            self.bn9_1 = nn.BatchNorm2d(13)

            self.conv9_2 = nn.Conv2d(13, 13, kernel_size=1, stride=1, padding=0)
            self.bn9_2 = nn.BatchNorm2d(13)

            self.conv9_3 = nn.Conv2d(32, 13, kernel_size=1, stride=1, padding=0)
            self.bn9_3 = nn.BatchNorm2d(13)

            self.conv9_4 = nn.Conv2d(13, 13, kernel_size=1, stride=1, padding=0)
            self.bn9_4 = nn.BatchNorm2d(13)

        self.compress_level = compress_level
        if compress_level > 0:
            assert compress_level <= 8
            compress_channel_num = 256 // (2**compress_level)

            # currently only support compress/decompress at layer x_3
            self.com_compresser = nn.Conv2d(
                256, compress_channel_num, kernel_size=1, stride=1
            )
            self.bn_compress = nn.BatchNorm2d(compress_channel_num)

            self.com_decompresser = nn.Conv2d(
                compress_channel_num, 256, kernel_size=1, stride=1
            )
            self.bn_decompress = nn.BatchNorm2d(256)

    def encode(self, x):
        """Encode the input BEV features.

        Args:
            x (tensor): the input BEV features.

        Returns:
            A list that contains all the encoded layers.
        """
        batch, seq, z, h, w = x.size()

        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x = x.to(torch.float)
        x = F.relu(self.bn_pre_1(self.conv_pre_1(x)))
        x = F.relu(self.bn_pre_2(self.conv_pre_2(x)))

        # -------------------------------- Encoder Path --------------------------------
        # -- STC block 1
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))

        x_1 = x_1.view(
            batch, -1, x_1.size(1), x_1.size(2), x_1.size(3)
        ).contiguous()  # (batch, seq, c, h, w)
        x_1 = self.conv3d_1(x_1)
        x_1 = x_1.view(
            -1, x_1.size(2), x_1.size(3), x_1.size(4)
        ).contiguous()  # (batch * seq, c, h, w)

        # -- STC block 2
        x_2 = F.relu(self.bn2_1(self.conv2_1(x_1)))
        x_2 = F.relu(self.bn2_2(self.conv2_2(x_2)))

        x_2 = x_2.view(
            batch, -1, x_2.size(1), x_2.size(2), x_2.size(3)
        ).contiguous()  # (batch, seq, c, h, w)
        x_2 = self.conv3d_2(x_2)
        x_2 = x_2.view(
            -1, x_2.size(2), x_2.size(3), x_2.size(4)
        ).contiguous()  # (batch * seq, c, h, w), seq = 1

        # -- STC block 3
        x_3 = F.relu(self.bn3_1(self.conv3_1(x_2)))
        x_3 = F.relu(self.bn3_2(self.conv3_2(x_3)))

        # -- STC block 4
        x_4 = F.relu(self.bn4_1(self.conv4_1(x_3)))
        x_4 = F.relu(self.bn4_2(self.conv4_2(x_4)))

        # compress x_3 (the layer that agents communicates on)
        if self.compress_level > 0:
            x_3 = F.relu(self.bn_compress(self.com_compresser(x_3)))
            x_3 = F.relu(self.bn_decompress(self.com_decompresser(x_3)))

        return [x, x_1, x_2, x_3, x_4]

    def decode(
        self,
        x,
        x_1,
        x_2,
        x_3,
        x_4,
        batch,
        kd_flag=False,
        requires_adaptive_max_pool3d=False,
    ):
        """Decode the input features.

        Args:
            x (tensor): layer-0 features.
            x_1 (tensor): layer-1 features.
            x_2 (tensor): layer-2 features.
            x_3 (tensor): layer-3 features.
            x_4 (tensor): layer-4 featuers.
            batch (int): The batch size.
            kd_flag (bool, optional): Required to be true for DiscoNet. Defaults to False.
            requires_adaptive_max_pool3d (bool, optional): If set to true, use adaptive max pooling 3d. Defaults to False.

        Returns:
            if kd_flag is true, return a list of output from layer-8 to layer-5
            else return a list of a single element: the output after passing through the decoder
        """
        # -------------------------------- Decoder Path --------------------------------
        x_5 = F.relu(
            self.bn5_1(
                self.conv5_1(
                    torch.cat((F.interpolate(x_4, scale_factor=(2, 2)), x_3), dim=1)
                )
            )
        )
        x_5 = F.relu(self.bn5_2(self.conv5_2(x_5)))

        x_2 = x_2.view(batch, -1, x_2.size(1), x_2.size(2), x_2.size(3))
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = (
            F.adaptive_max_pool3d(x_2, (1, None, None))
            if requires_adaptive_max_pool3d
            else x_2
        )
        x_2 = x_2.permute(0, 2, 1, 3, 4).contiguous()
        x_2 = x_2.view(-1, x_2.size(2), x_2.size(3), x_2.size(4)).contiguous()

        x_6 = F.relu(
            self.bn6_1(
                self.conv6_1(
                    torch.cat((F.interpolate(x_5, scale_factor=(2, 2)), x_2), dim=1)
                )
            )
        )
        x_6 = F.relu(self.bn6_2(self.conv6_2(x_6)))

        x_1 = x_1.view(batch, -1, x_1.size(1), x_1.size(2), x_1.size(3))
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = (
            F.adaptive_max_pool3d(x_1, (1, None, None))
            if requires_adaptive_max_pool3d
            else x_1
        )
        x_1 = x_1.permute(0, 2, 1, 3, 4).contiguous()
        x_1 = x_1.view(-1, x_1.size(2), x_1.size(3), x_1.size(4)).contiguous()

        x_7 = F.relu(
            self.bn7_1(
                self.conv7_1(
                    torch.cat((F.interpolate(x_6, scale_factor=(2, 2)), x_1), dim=1)
                )
            )
        )
        x_7 = F.relu(self.bn7_2(self.conv7_2(x_7)))

        x = x.view(batch, -1, x.size(1), x.size(2), x.size(3))
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = (
            F.adaptive_max_pool3d(x, (1, None, None))
            if requires_adaptive_max_pool3d
            else x
        )
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, x.size(2), x.size(3), x.size(4)).contiguous()

        x_8 = F.relu(
            self.bn8_1(
                self.conv8_1(
                    torch.cat((F.interpolate(x_7, scale_factor=(2, 2)), x), dim=1)
                )
            )
        )
        res_x = F.relu(self.bn8_2(self.conv8_2(x_8)))

        if self.train_completion:
            occ_x = F.relu(self.bn9_1(self.conv9_1(res_x)))
            occ_x = F.relu(self.bn9_2(self.conv9_2(occ_x)))
            # get binary classification logits
            free_x = F.relu(self.bn9_3(self.conv9_3(res_x)))
            free_x = F.relu(self.bn9_4(self.conv9_4(free_x)))

            res_x = torch.stack((free_x, occ_x), dim=1)

        if kd_flag:
            return [res_x, x_7, x_6, x_5]
        else:
            return [res_x]


class STPN_KD(Backbone):
    """Used by non-intermediate models. Pass the output from encoder directly to decoder."""

    def __init__(self, height_feat_size=13, compress_level=0, train_completion=False):
        super().__init__(height_feat_size, compress_level, train_completion)

    def forward(self, x):
        batch, seq, z, h, w = x.size()
        encoded_layers = super().encode(x)
        decoded_layers = super().decode(
            *encoded_layers, batch, kd_flag=True, requires_adaptive_max_pool3d=True
        )
        return (*decoded_layers, encoded_layers[3], encoded_layers[4])



class CNNNet(nn.Module):
    """
    Modified from:
    https://arxiv.org/pdf/2012.12395.pdf

    Args:
        config (object): The Config object.
        layer (int, optional): Collaborate on which layer. Defaults to 3.
        in_channels (int, optional): The input channels. Defaults to 13.
        kd_flag (bool, optional): Whether to use knowledge distillation (for DiscoNet to ues). Defaults to True.
        num_agent (int, optional): The number of agents (including RSU). Defaults to 5.
    """

    def __init__(
        self,
        config,
        layer=3,
        in_channels=13,
        kd_flag=True,
        num_agent=5,
        compress_level=0,
        train_completion=True,
    ):
        super().__init__()
        self.config = config
        self.kd_flag = kd_flag
        self.in_channels = in_channels
        self.num_agent = num_agent
        self.train_completion = train_completion
        self.stpn = STPN_KD(config.map_dims[2], compress_level, train_completion)

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
        return torch.squeeze(warp_feat_trans, dim=0) # [512, 16, 16]


    def build_neighbors_feature_list(self, b, agent_idx, all_warp, num_agent, local_com_mat, device,
                                     size) -> None:
        for j in range(num_agent):
            if j != agent_idx:
                warp_feat = self.feature_transformation(b, j, local_com_mat, all_warp, device, size)
                self.neighbor_feat_list.append(warp_feat)
        
    def fusion(self, mode="sum"):
        fused = torch.sum(torch.stack(self.neighbor_feat_list), dim=0)
        if mode == "union":
            fused[fused>0.] = 1.0 
        return fused 

    @staticmethod
    def agents_to_batch(feats):
        agent_num = feats.shape[1]
        feat_list = []
        for i in range(agent_num):
            feat_list.append(feats[:, i, :, :, :])
        feat_mat = torch.cat(tuple(feat_list), 0)
        return feat_mat

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

    def forward(self, bevs, trans_matrices=None, num_agent_tensor=None, batch_size=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)
        x = x_8

        # cls_preds, loc_preds, result = super().get_cls_loc_result(x)
        
        if self.train_completion:
            # fuse reconstructed BEVs
            ind_result = torch.argmax(torch.softmax(x, dim=1), dim=1)
            result = self.ego_late_fusion(ind_result, bevs.squeeze(1), trans_matrices, num_agent_tensor, batch_size)
        else:
            # do detection
            cls_preds, loc_preds, result = super().get_cls_loc_result(x)


        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, x_3
        elif self.train_completion:
            return result, x_8
        else:
            return result


class Conv3D(nn.Module):
    """3D cnn used in the encoder."""

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x