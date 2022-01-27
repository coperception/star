import torch
from utils.models.base.FusionBase import FusionBase


class MeanFusion(FusionBase):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(MeanFusion, self).__init__(config, layer, in_channels, kd_flag)

    def fusion(self):
        return torch.mean(torch.stack(self.neighbor_feat_list), dim=0)
