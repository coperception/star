from utils.models.base.NonIntermediateModelBase import NonIntermediateModelBase


class FaFNet(NonIntermediateModelBase):
    def __init__(self, config, layer=3, in_channels=13, kd_flag=True):
        super(FaFNet, self).__init__(config, layer, in_channels, kd_flag)

    def forward(self, bevs, maps=None, vis=None, batch_size=None):
        bevs = bevs.permute(0, 1, 4, 2, 3)  # (Batch, seq, z, h, w)

        x_8, x_7, x_6, x_5, x_3, x_2 = self.stpn(bevs)
        x = x_8

        cls_preds, loc_preds, result = super().get_cls_los_result(x)

        if self.kd_flag == 1:
            return result, x_8, x_7, x_6, x_5, x_3
        else:
            return result
