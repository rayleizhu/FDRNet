import torch
from torch import nn
import torch.nn.functional as F


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''
    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1. - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss

    
class DiceLoss(nn.Module):
    def __init__(self, apply_sigmoid=True, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.apply_sigmoid = apply_sigmoid
    
    def forward(self, pred, target):
        if self.apply_sigmoid:
            pred = F.sigmoid(pred)
            
        numerator = 2 * torch.sum(pred * target) + self.smooth
        denominator = torch.sum(pred + target) + self.smooth
        return 1 - numerator / denominator