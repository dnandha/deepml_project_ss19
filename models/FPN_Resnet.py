import torch
import torchvision.models.resnet as resnet
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os


class FPN_Resnet(nn.Module):
    """Implements the FeaturePyramidNetwork in Pytorch, https://arxiv.org/pdf/1612.03144.pdf"""

    def __init__(self, net_size, num_classes, pretrained=True, check_path="", upsample_method="bilinear"):
        """
        net_size: Specifies which ResNet architecture to use. Valid values are (18,34,50,101,152)
        num_classes: Number of output layers
        pretrained: Default True. Initializes the Resnet with weights trained on imagenet.
        check_path: If pretrained set to true, tries to load a state_dict from path.
        upsample_method: Method to scale up on Bottom-Up path. Valid values are ('bilinear', 'nearest')
        """

        super(FPN_Resnet, self).__init__()

        net_params = {18: [64, 64, 128, 256, 512],
                      34: [64, 64, 128, 256, 512],
                      50: [64, 256, 512, 1024, 2048],
                      101: [64, 256, 512, 1024, 2048],
                      152: [64, 256, 512, 1024, 2048]}

        resnets = {18: resnet.resnet18, 34: resnet.resnet34, 50: resnet.resnet50, 101: resnet.resnet101,
                   152: resnet.resnet152}
        upsample_methods = ["nearest", "bilinear"]

        assert net_size in resnets, "Netsize not found! Available sizes are {}".format(sorted(list(net_params.keys())))
        assert upsample_method in upsample_methods, "Upsample Method not found! Available Methods are: {}".format(
            upsample_methods)

        self.upsample_method = upsample_method

        if (pretrained):
            if (check_path == ""):
                self.resnet = resnets[net_size](pretrained=pretrained)
            else:
                self.resnet = resnets[net_size](pretrained=False)
        else:
            self.resnet = resnets[net_size](pretrained=False)

        self.resnet.fc = None

        self.lat0 = nn.Conv2d(3, num_classes, kernel_size=1, stride=1, padding=0)
        self.lat1 = nn.Conv2d(net_params[net_size][0], num_classes, kernel_size=1, stride=1, padding=0)
        self.lat2 = nn.Conv2d(net_params[net_size][1], num_classes, kernel_size=1, stride=1, padding=0)
        self.lat3 = nn.Conv2d(net_params[net_size][2], num_classes, kernel_size=1, stride=1, padding=0)
        self.lat4 = nn.Conv2d(net_params[net_size][3], num_classes, kernel_size=1, stride=1, padding=0)

        self.to_top_bottom = nn.Conv2d(net_params[net_size][4], num_classes, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)
        self.smooth5 = nn.Conv2d(num_classes, num_classes, kernel_size=3, stride=1, padding=1)

        if (check_path != "" and os.path.exists(check_path)):
            self.load_state_dict(torch.load(check_path))
        elif check_path != "":
            print("Checkpoint not found!")
            exit()

    def forward(self, x):

        # BottomUp-Pathway
        l0 = self.lat0(x)  # 1. lateral connection (Pyramidlevel0) @size: H*W

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)

        l1 = self.lat1(x)  # 2. lateral connection (PyramidLevel1) @size: H/2*W/2

        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)

        l2 = self.lat2(x)  # 3. lateral connection (PyramidLevel2) @size: H/4*W/4

        x = self.resnet.layer2(x)

        l3 = self.lat3(x)  # 4. lateral connection (PyramidLevel3) @size: H/8*W/8

        x = self.resnet.layer3(x)

        l4 = self.lat4(x)  # 5. lateral connection (PyramidLevel4) @size: H/16*H/16

        x = self.resnet.layer4(x)

        # TopBottom-Pathway
        # TopBottom-Pathway
        p1 = self.to_top_bottom(x)  # PyramidLevel5 prediction @size:H/32*W/32

        p2 = self.smooth1(self._add_and_upscale(p1, l4))  # PyramidLevel4 prediction @size:H/16*W/16. Endpoint of lat4

        p3 = self.smooth2(self._add_and_upscale(p2, l3))  # PyramidLevel3 prediction @size:H/8*W/8. Endpoint of lat3

        p4 = self.smooth3(self._add_and_upscale(p3, l2))  # PyramidLevel2 prediction @size:H/4*W/4. Endpoint of lat2

        p5 = self.smooth4(self._add_and_upscale(p4, l1))  # PyramidLevel1 prediction @size:H/2*W/2. Endpoint of lat1

        p6 = self.smooth5(self._add_and_upscale(p5, l0))  # PyramidLevel0 prediction @size:H*W. Endpoint of lat0

        p1 = F.log_softmax(p1, dim=1)
        p2 = F.log_softmax(p2, dim=1)
        p3 = F.log_softmax(p3, dim=1)
        p4 = F.log_softmax(p4, dim=1)
        p5 = F.log_softmax(p5, dim=1)
        p6 = F.log_softmax(p6, dim=1)

        return p1, p2, p3, p4, p5, p6

    def _add_and_upscale(self, x, lat):
        _, _, H, W = lat.data.size()
        return F.upsample(x, size=(H, W), mode=self.upsample_method).add_(lat)


if __name__ == "__main__":
    bla = Variable(torch.randn(3, 3, 5, 5))
    m = torch.nn.Softmax(dim=1)
    soft_bla = m(bla)[0]
    print(torch.sum(soft_bla, dim=0))

    net = FPN_Resnet(net_size=18, num_classes=20, pretrained=False, check_path="", upsample_method="bilinear")

    input_net = Variable(torch.rand(5, 3, 64, 64))

    pred = net(input_net)
    p5 = pred[4][3]
    print(p5.size())
    print(torch.sum(p5, dim=0))
