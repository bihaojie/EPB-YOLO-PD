import torch
import torch.nn as nn

class SlideLoss(nn.Module):
    def __init__(self, loss_fcn):
        super(SlideLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true, auto_iou=0.5):
        loss = self.loss_fcn(pred, true)
        auto_iou = max(auto_iou, 0.2)
        low_iou = true <= auto_iou - 0.1
        medium_iou = (true > (auto_iou - 0.1)) & (true < auto_iou)
        high_iou = true >= auto_iou

        low_weight = 1.0
        medium_weight = torch.exp(1.0 - auto_iou)
        high_weight = torch.exp(-(true - 1.0))

        modulating_weight = low_weight * low_iou + medium_weight * medium_iou + high_weight * high_iou

        loss *= modulating_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

