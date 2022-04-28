from utils.models.backbone.Backbone import *
from utils.models.base.DetModelBase import DetModelBase


class IntermediateModelBase(DetModelBase):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, layer1_channel=64, layer2_channel=128, layer3_channel=256):
        super().__init__(config, layer, in_channels, kd_flag)
        self.u_encoder = LidarEncoder(height_feat_size=in_channels, layer1_channel=layer1_channel, layer2_channel=layer2_channel, layer3_channel=layer3_channel)
        self.decoder = LidarDecoder(height_feat_size=in_channels, layer1_channel=layer1_channel, layer2_channel=layer2_channel, layer3_channel=layer3_channel)
