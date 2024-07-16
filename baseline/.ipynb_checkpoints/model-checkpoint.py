import math
import torch
import torch.nn as nn
from torch.nn import init, Softmax
import torch.nn.functional as F
from utils import weights_init_classifier, weights_init_kaiming
from resnet import resnet50
from torch.cuda.amp import autocast


class GeMP(nn.Module):
    def __init__(self, p=3.0, eps=1e-12):
        super(GeMP, self).__init__()
        self.p = p
        self.eps = eps

    def forward(self, x):
        p, eps = self.p, self.eps
        if x.ndim != 2:
            batch_size, fdim = x.shape[:2]
            x = x.view(batch_size, fdim, -1)
        return (torch.mean(x ** p, dim=-1) + eps) ** (1 / p)


class visible_module(nn.Module):
    def __init__(self, pretrained=True):
        super(visible_module, self).__init__()
        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v
        self.visible.layer3 = None
        self.visible.layer4 = None

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        x = self.visible.layer2(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, pretrained=True):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.thermal = model_t
        self.thermal.layer3 = None
        self.thermal.layer4 = None

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        x = self.thermal.layer2(x)
        return x


class base_module(nn.Module):
    def __init__(self, pretrained=True):
        super(base_module, self).__init__()
        base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        self.base = base
        self.base.conv1 = None
        self.base.bn1 = None
        self.base.relu = None
        self.base.maxpool = None
        self.layer1 = None
        self.layer2 = None

    def forward(self, x):
        x = self.base.layer3(x)
        x = self.base.layer4(x)

        return x




class embed_net(nn.Module):
    def __init__(self, class_num, pool_dim=2048, pretrained=True):
        super(embed_net, self).__init__()

        self.visible = visible_module(pretrained=pretrained)
        self.thermal = thermal_module(pretrained=pretrained)
        self.base = base_module(pretrained=pretrained)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.relu = nn.ReLU()
        self.pool = GeMP()
        self.gap = nn.AdaptiveAvgPool2d(1)

    @autocast()
    def forward(self, x_v, x_t, label1=None, label2=None, modal=0):
        if modal == 0:
            x_v = self.visible(x_v)
            x_t = self.thermal(x_t)
            x = torch.cat((x_v, x_t), 0)
            del x_v, x_t

        elif modal == 1:
            x = self.visible(x_v)

        elif modal == 2:
            x = self.thermal(x_t)

        if self.training:
            x = self.base(x)
            b, c, h, w = x.shape
            x = self.relu(x)
            x = x.view(b, c, h * w)
            x = self.pool(x)

            x_after_BN = self.bottleneck(x)

            cls_id = self.classifier(x_after_BN)

            return {
                'cls_id': cls_id,
                'feat': x_after_BN,
            }

        else:
            x = self.base(x)
            b, c, h, w = x.shape
            x = self.relu(x)
            x = x.view(b, c, h * w)
            x = self.pool(x)
            x_after_BN = self.bottleneck(x)

        return F.normalize(x, p=2.0, dim=1), F.normalize(x_after_BN, p=2.0, dim=1)




