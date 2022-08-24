from coperception.models.det.base import NonIntermediateModelBase
import torch
import torch.nn.functional as F
import torch.nn as nn

class FaFNet(NonIntermediateModelBase):
    """The model of early fusion. Used as lower-bound and upper-bound depending on the input features (fused or not).

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
        super().__init__(config, layer, in_channels, kd_flag, num_agent, compress_level, train_completion)
        self.train_completion = train_completion

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
