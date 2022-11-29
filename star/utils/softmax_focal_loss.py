import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(SoftmaxFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, target):
        # print(inputs.size())
        # print(target.size())
        prob = F.softmax(inputs, dim=1)
        reverse_target = 1 - target
        one_hot_target = torch.stack([reverse_target, target], dim=1)
        # print(one_hot_target.size())
        pt = (prob*one_hot_target).sum(dim=1) 
        ce_loss = - pt.log() # [B, C, H, W] C is the pc height channel
        alpha_t = (self.alpha * one_hot_target).sum(dim=1)

        loss = alpha_t * torch.pow((1-pt), self.gamma) * ce_loss
        # print("loss", loss.mean())

        if self.reduction=="mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()

        return loss



