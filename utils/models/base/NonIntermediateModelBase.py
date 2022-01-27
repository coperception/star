from utils.models.backbone.Backbone import *
from utils.models.base.DetModelBase import DetModelBase


class NonIntermediateModelBase(DetModelBase):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(NonIntermediateModelBase, self).__init__(config, layer, in_channels, kd_flag)
        self.stpn = STPN_KD(height_feat_size=config.map_dims[2])