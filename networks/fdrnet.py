import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvmodels
from efficientnet_pytorch import EfficientNet
import math
import numpy as np


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConstantNormalize(nn.Module):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ConstantNormalize, self).__init__()
        mean = torch.Tensor(mean).view([1, 3, 1, 1])
        std = torch.Tensor(std).view([1, 3, 1, 1])
        # https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x):
        return (x  - self.mean) / (self.std + 1e-5)


class Conv1x1(nn.Sequential):
     def __init__(self, in_planes, out_planes=16, has_se=False, se_reduction=None):
        if has_se:
            if se_reduction is None:
                # se_reduction= int(math.sqrt(in_planes))
                se_reduction = 2
            super(Conv1x1, self).__init__(SELayer(in_planes, se_reduction),
                                           nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                           nn.BatchNorm2d(out_planes),
                                           nn.ReLU()
                                           )
        else:
            super(Conv1x1, self).__init__(nn.Conv2d(in_planes, out_planes, 1, bias=False),
                                          nn.BatchNorm2d(out_planes),
                                          nn.ReLU()
                                         )

# https://pytorch.org/docs/stable/_modules/torchvision/models/resnet.html#resnext50_32x4d
class ResBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ResBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class FRUnit(nn.Module):
    """
    Factorisation and Reweighting unit
    """
    def __init__(self, channels=32, mu_init=0.5, reweight_mode='manual', normalize=True):
        super(FRUnit, self).__init__()
        assert reweight_mode in ['manual', 'constant', 'learnable', 'nn']
        self.mu = mu_init
        self.reweight_mode = reweight_mode
        self.normalize = normalize
        self.inv_conv = ResBlock(channels, channels)
        self.var_conv = ResBlock(channels, channels)
        if reweight_mode == 'learnable':
            self.mu = nn.Parameter(torch.tensor(mu_init))
            # self.x = nn.Parameter(torch.tensor(0.))
            # self.mu = torch.sigmoid(self.x)
        elif reweight_mode == 'nn':
            self.fc = nn.Sequential(nn.Linear(channels, 1),
                                    nn.Sigmoid()
                                   )
        else:
            self.mu = mu_init


    def forward(self, feat):
        inv_feat = self.inv_conv(feat)
        var_feat = self.var_conv(feat)

        if self.normalize:
            inv_feat = F.normalize(inv_feat)
            var_feat = F.normalize(var_feat)
            # var_feat = inv_feat - (inv_feat * var_feat).sum(keepdim=True, dim=1) * inv_feat
            # var_feat = F.normalize(var_feat)
        
        if self.reweight_mode == 'nn':
            gap = feat.mean([2, 3])
            self.mu = self.fc(gap).view(-1, 1, 1, 1)

        mix_feat = self.mu * var_feat + (1 - self.mu) * inv_feat
        return inv_feat, var_feat, mix_feat


    def set_mu(self, val):
        assert self.reweight_mode == 'manual'
        self.mu = val


ml_features = []

def feature_hook(module, fea_in, fea_out):
#     print("hooker working")
    # module_name.append(module.__class__)
    # features_in_hook.append(fea_in)
    global ml_features
    ml_features.append(fea_out)
    return None

class FDRNet(nn.Module):
    # decompose net
    def __init__(self,
                 backbone='efficientnet-b0',
                 proj_planes=16,
                 pred_planes=32,
                 use_pretrained=True,
                 fix_backbone=False,
                 has_se=False,
                 dropout_2d=0,
                 normalize=False,
                 mu_init=0.5,
                 reweight_mode='constant'):
                 
        super(FDRNet, self).__init__()

        self.mu_init = mu_init
        self.reweight_mode = reweight_mode

        # load backbone
        if use_pretrained:
            self.feat_net = EfficientNet.from_pretrained(backbone)
            # https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py
        else:
            self.feat_net = EfficientNet.from_name(backbone)
        
        # remove classification head to get correct param count
        self.feat_net._avg_pooling=None
        self.feat_net._dropout=None
        self.feat_net._fc=None
        
        # register hook to extract multi-level features
        in_planes = []
        feat_layer_ids = list(range(0, len(self.feat_net._blocks), 2))
        for idx in feat_layer_ids:
            self.feat_net._blocks[idx].register_forward_hook(hook=feature_hook)
            in_planes.append(self.feat_net._blocks[idx]._bn2.num_features)
        
        if fix_backbone:
            for param in self.feat_net.parameters():
                param.requires_grad = False
        
        self.norm = ConstantNormalize()

        # 1*1 projection conv
        proj_convs = [ Conv1x1(ip, proj_planes, has_se=has_se) for ip in in_planes ]
        self.proj_convs = nn.ModuleList(proj_convs)

        # two stream feature
        self.stem_conv = Conv1x1(proj_planes*len(in_planes), pred_planes, has_se=has_se)
        self.fr = FRUnit(pred_planes, mu_init=mu_init, reweight_mode=reweight_mode, normalize=normalize)
        self.fc = nn.Linear(pred_planes, 1)
  
        # prediction   
        pred_layers = []
        if dropout_2d > 1e-6:
            pred_layers.append(nn.Dropout2d(p=dropout_2d))
        pred_layers.append(nn.Conv2d(pred_planes, 1, 1))
        self.pred_conv = nn.Sequential(*pred_layers)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True
        
    def forward(self, x):
        global ml_features
        
        b, c, h, w = x.size()
        ml_features = []

        _ = self.feat_net.extract_features(self.norm(x))
        
        h_f, w_f = ml_features[0].size()[2:]
        proj_features = []
        for i in range(len(ml_features)):
            cur_proj_feature = self.proj_convs[i](ml_features[i])
            cur_proj_feature_up = F.interpolate(cur_proj_feature, size=(h_f, w_f), mode='bilinear')
            proj_features.append(cur_proj_feature_up)
        cat_feature = torch.cat(proj_features, dim=1)
    
        stem_feat = self.stem_conv(cat_feature)
        
        # factorised feature
        inv_feat, var_feat, mix_feat = self.fr(stem_feat)
        
        if self.training:
            logits = F.interpolate(self.pred_conv(mix_feat), size=(h, w), mode='bilinear')
            g = self.fc(var_feat.mean([2, 3]))
            return logits, inv_feat, g
        else:
            if self.reweight_mode != 'learnable': 
                mix_feat = self.mu_init * var_feat + (1 - self.mu_init) * inv_feat
                logits = F.interpolate(self.pred_conv(mix_feat), size=(h, w), mode='bilinear')
            else:
                logits = F.interpolate(self.pred_conv(mix_feat), size=(h, w), mode='bilinear')
            return logits

