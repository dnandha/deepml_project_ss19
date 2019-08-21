import torch.nn as nn
import torch.nn.functional as F
import torch


# Atrous Spatial Pyramid Pooling
class ASPP(nn.Module):
    def __init__(self, in_channels, rates=(6, 12, 18), features=256, norm=nn.BatchNorm2d):
        super().__init__()

        self.rates = rates

        self.c1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 1, bias=False),
            norm(features),
            #nn.ReLU(True)
        )
        self.c3_1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=rates[0], dilation=rates[0], bias=False),
            norm(features),
            #nn.ReLU(True)
        )
        self.c3_2 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=rates[1], dilation=rates[1], bias=False),
            norm(features),
            #nn.ReLU(True)
        )
        self.c3_3 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=rates[2], dilation=rates[2], bias=False),
            norm(features),
            #nn.ReLU(True)
        )

        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, features, 1, bias=False),
            norm(features),
            #nn.ReLU(True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(5 * 256, 256, 1, bias=False),
            norm(256),
            #nn.ReLU(True)
        )

    def forward(self, x):
        c1 = self.c1(x)
        c3_1 = self.c3_1(x)
        c3_2 = self.c3_2(x)
        c3_3 = self.c3_3(x)

        pool = self.pool(x)
        pool = F.interpolate(pool, (x.size(2), x.size(3)), mode='bilinear', align_corners=True)

        merge = torch.cat([c1, c3_1, c3_2, c3_3, pool], 1)

        return self.final(merge)

    def changeStride(self, ostride=8):
        for m in self.c3_1.modules():
            if isinstance(m, nn.modules.Conv2d):
                m.dilation = (self.rates[0] * 2, self.rates[0] * 2)
                m.padding = (self.rates[0] * 2, self.rates[0] * 2)

        for m in self.c3_2.modules():
            if isinstance(m, nn.modules.Conv2d):
                m.dilation = (self.rates[1] * 2, self.rates[1] * 2)
                m.padding = (self.rates[1] * 2, self.rates[1] * 2)

        for m in self.c3_3.modules():
            if isinstance(m, nn.modules.Conv2d):
                m.dilation = (self.rates[2] * 2, self.rates[2] * 2)
                m.padding = (self.rates[2] * 2, self.rates[2] * 2)
