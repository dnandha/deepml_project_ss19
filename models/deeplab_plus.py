# author: Simon Kiefhaber
# implementation of DeepLabV3+(https://arxiv.org/pdf/1706.05587.pdf)
# recommended crop size: 513x513

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
try:
    from ASPP import ASPP
except:
    from .ASPP import ASPP


# Bottleneck modified from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = norm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class DeepLabV3Plus(nn.Module):
    def __init__(self, out_channels, pretrained=True, norm=nn.BatchNorm2d, freeze_bn=False):
        super().__init__()

        self.freeze_bn = freeze_bn

        res = resnet50(pretrained)
        res.fc = None

        self.inplanes = res.inplanes
        self.encoder1 = nn.Sequential(
            res.conv1,
            res.bn1,
            res.relu,

            res.maxpool,
            res.layer1,  # ceil(size / 4)
        )

        self.ASPP = ASPP(2048, norm=norm)
        self.layer4 = res.layer4
        self.layer3 = res.layer3
        self.encoder2 = nn.Sequential(
            res.layer2,  # ceil(size / 8)
            self.layer3,  # ceil(size / 16)
            self.layer4,
            self.ASPP
        )

        self.layer4[0].dilation = (2, 2)
        for m in self.layer4.modules():
            if isinstance(m, nn.modules.Conv2d):
                m.stride = 1

        for m in self.encoder1.modules():
            if isinstance(m, nn.modules.ReLU):
                m.inplace = True

        for m in self.encoder2.modules():
            if isinstance(m, nn.modules.ReLU):
                m.inplace = True

        self.lowDecoder = nn.Sequential(
            nn.Conv2d(256, 48, 1, 1, bias=False),
            nn.BatchNorm2d(48),
            # nn.ReLU(True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(304, 256, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)

        # Decoder
        lowfeat = self.lowDecoder(enc1)
        dec = F.interpolate(enc2, (lowfeat.size(2), lowfeat.size(3)), mode='bilinear', align_corners=True)
        dec = torch.cat([dec, lowfeat], 1)

        final = self.final(dec)
        final = F.interpolate(final, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        final = F.log_softmax(final, dim=1)

        return final

    def train(self, mode=True):
        super(DeepLabV3Plus, self).train(mode)

        if self.freeze_bn:
            for module in self.modules():
                if isinstance(module, nn.modules.BatchNorm1d):
                    module.eval()
                if isinstance(module, nn.modules.BatchNorm2d):
                    module.eval()
                if isinstance(module, nn.modules.BatchNorm3d):
                    module.eval()

    def changeStride(self, ostride=8):
        self.layer4[0].dilation = (4, 4)
        for m in self.layer4.modules():
            if isinstance(m, nn.modules.Conv2d):
                m.stride = 1

        self.layer3[0].dilation = (2, 2)
        for m in self.layer3.modules():
            if isinstance(m, nn.modules.Conv2d):
                m.stride = 1

        self.ASPP.changeStride()
