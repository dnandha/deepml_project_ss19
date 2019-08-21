import torch
import torchvision.models.resnet as resnet
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os



class Down_U_Net_Block(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Down_U_Net_Block, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, outplanes, 3,  stride=1, padding=1)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3,  stride=1, padding=1)

        #self.bn1 = nn.BatchNorm2d(outplanes)
        self.bn = nn.BatchNorm2d(outplanes)

        #self.relu1 = nn.ReLU(True)
        self.relu = nn.ReLU(True)
    def forward(self, x):
        x = self.conv1(x)
        #x = self.relu1(x)
        #x = self.bn1(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Up_U_Net_Block(nn.Module):
    def __init__(self, inplanes, outplanes):

        super(Up_U_Net_Block, self).__init__()

        self.upconv = nn.ConvTranspose2d(inplanes, inplanes//2, 2, 2, bias=False)
        self.conv1 = nn.Conv2d(inplanes, outplanes, 3,  stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3,  stride=1, padding=1, bias=False)

        #self.bn1 = nn.BatchNorm2d(outplanes)
        self.bn = nn.BatchNorm2d(outplanes)

        #self.relu1 = nn.ReLU(True)
        self.relu = nn.ReLU(True)


    def forward(self, x, y):
        x = torch.cat((self.upconv(x),y), 1)

        x = self.conv1(x)
        #x = self.relu1(x)
        #x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)#F.relu(x)

        return x

#Uses a given upsample method instead of 
class ScaleAdd_U_Net_Block(nn.Module):
    def __init__(self, inplanes, outplanes, upsample_method):

        super(ScaleAdd_U_Net_Block, self).__init__()

        self.upsample_method = upsample_method

        self.latconv = nn.ConvTranspose2d(inplanes//2, inplanes, 1) #Lateral connection

        self.conv1 = nn.Conv2d(inplanes, outplanes, 3,  stride=1, padding=1)
        self.conv2 = nn.Conv2d(outplanes, outplanes, 3,  stride=1, padding=1)
        self.bn = nn.BatchNorm2d(outplanes)

    def forward(self, x, y):
        # x is small input, y is lateral input
        x = F.interpolate(x, scale_factor=2, mode=self.upsample_method)
        y = self.latconv(y)
        x = x+y
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = F.relu(x)

        return x


class Mod_U_Net(nn.Module):
    """Implements the U-Net in Pytorch, https://arxiv.org/pdf/1505.04597.pdf"""

    # Caution! Modified.

    def __init__(self, num_classes, upsample_method="bilinear", check_path=""):

        super(Mod_U_Net, self).__init__()

        upsample_methods = ["nearest", "bilinear"]

        assert upsample_method in upsample_methods, "Upsample Method not found! Available Methods are: {}".format(
            upsample_methods)

        self.upsample_method = upsample_method

        self.num_classes = num_classes
        # Added BatchNorm because its good :D
        # Also using upscaling instead of upconv. Vlt sagen sie das im FPN_Resnet Paper.

        if (check_path != "" and os.path.exists(check_path)):
            self.load_state_dict(torch.load(check_path))
        elif check_path != "":
            print("Checkpoint not found!")
            exit()

        self.down_block1 = Down_U_Net_Block(3, 64)
        self.down_block2 = Down_U_Net_Block(64, 128)
        self.down_block3 = Down_U_Net_Block(128, 256)
        self.down_block4 = Down_U_Net_Block(256, 512)
        self.down_block5 = Down_U_Net_Block(512, 1024)

        self.up_block4 = ScaleAdd_U_Net_Block(1024, 512, self.upsample_method)
        self.up_block3 = ScaleAdd_U_Net_Block(512, 256, self.upsample_method)
        self.up_block2 = ScaleAdd_U_Net_Block(256, 128, self.upsample_method)
        self.up_block1 = ScaleAdd_U_Net_Block(128, 64, self.upsample_method)

        self.final = nn.Conv2d(64, self.num_classes, 1)

    def forward(self, x):

        # Down Path
        s1 = self.down_block1(x)

        x = F.max_pool2d(s1, 2)
        s2 = self.down_block2(x)

        x = F.max_pool2d(s2, 2)
        s3 = self.down_block3(x)

        x = F.max_pool2d(s3, 2)
        s4 = self.down_block4(x)

        x = F.max_pool2d(s4, 2)
        s5 = self.down_block5(x)

        # Outputs of blocks are saved in s1,s2,s3,s4,s5

        x = self.up_block4(s5, s4)

        x = self.up_block3(x, s3)

        x = self.up_block2(x, s2)

        x = self.up_block1(x, s1)

        x = self.final(x)

        return x


class U_Net(nn.Module):
    """Implements the U-Net in Pytorch, https://arxiv.org/pdf/1505.04597.pdf"""
    # Caution! Modified.

    def __init__(self, num_classes, check_path = ""):

        super(U_Net, self).__init__()

        self.num_classes = num_classes
        #Added BatchNorm because its good :D
        #Also using upscaling instead of upconv. Vlt sagen sie das im FPN_Resnet Paper.


        if (check_path != "" and os.path.exists(check_path)):
            self.load_state_dict(torch.load(check_path), strict=False)
        elif check_path != "":
            print("Checkpoint not found!")
            exit()


        self.down_block1 = Down_U_Net_Block(3, 64)
        self.down_block2 = Down_U_Net_Block(64, 128)
        self.down_block3 = Down_U_Net_Block(128, 256)
        self.down_block4 = Down_U_Net_Block(256, 512)
        self.down_block5 = Down_U_Net_Block(512, 1024)

        self.up_block4 = Up_U_Net_Block(1024, 512)
        self.up_block3 = Up_U_Net_Block(512 , 256)
        self.up_block2 = Up_U_Net_Block(256 , 128)
        self.up_block1 = Up_U_Net_Block(128 , 64)

        self.final = nn.Conv2d(64, self.num_classes, 1)

        self.max1 = nn.MaxPool2d(2)
        self.max2 = nn.MaxPool2d(2)
        self.max3 = nn.MaxPool2d(2)
        self.max4 = nn.MaxPool2d(2)

    def forward(self, x):


        #Down Path
        s1 = self.down_block1(x)

        x = self.max1(s1)#F.max_pool2d(s1, 2)
        s2 = self.down_block2(x)

        x = self.max1(s2)#F.max_pool2d(s2, 2)
        s3 = self.down_block3(x)

        x = self.max1(s3)#F.max_pool2d(s3, 2)
        s4 = self.down_block4(x)

        x = self.max1(s4)#F.max_pool2d(s4, 2)
        s5 = self.down_block5(x)

        #Outputs of blocks are saved in s1,s2,s3,s4,s5

        x = self.up_block4(s5, s4)

        x = self.up_block3(x, s3)

        x = self.up_block2(x, s2)

        x = self.up_block1(x, s1)

        x = self.final(x)

        return x

class U_Net_Categorie(nn.Module):
    """Implements the U-Net in Pytorch, https://arxiv.org/pdf/1505.04597.pdf"""
    # Caution! Modified.

    def __init__(self, num_classes, num_categories ,check_path = ""):

        super(U_Net_Categorie, self).__init__()

        self.num_classes = num_classes
        self.num_categories = num_categories
        #Added BatchNorm because its good :D
        #Also using upscaling instead of upconv. Vlt sagen sie das im FPN_Resnet Paper.


        if (check_path != "" and os.path.exists(check_path)):
            self.load_state_dict(torch.load(check_path), strict=False)
        elif check_path != "":
            print("Checkpoint not found!")
            exit()


        self.down_block1 = Down_U_Net_Block(3, 64)
        self.down_block2 = Down_U_Net_Block(64, 128)
        self.down_block3 = Down_U_Net_Block(128, 256)
        self.down_block4 = Down_U_Net_Block(256, 512)
        self.down_block5 = Down_U_Net_Block(512, 1024)

        self.up_block4 = Up_U_Net_Block(1024, 512)
        self.up_block3 = Up_U_Net_Block(512 , 256)
        self.up_block2 = Up_U_Net_Block(256 , 128)
        self.up_block1 = Up_U_Net_Block(128 , 64)

        self.final = nn.Conv2d(64, self.num_classes, 1)
        self.final_cat = nn.Conv2d(64, self.num_categories, 1)

        self.max1 = nn.MaxPool2d(2)
        self.max2 = nn.MaxPool2d(2)
        self.max3 = nn.MaxPool2d(2)
        self.max4 = nn.MaxPool2d(2)

    def forward(self, x):


        #Down Path
        s1 = self.down_block1(x)

        x = self.max1(s1)#F.max_pool2d(s1, 2)
        s2 = self.down_block2(x)

        x = self.max1(s2)#F.max_pool2d(s2, 2)
        s3 = self.down_block3(x)

        x = self.max1(s3)#F.max_pool2d(s3, 2)
        s4 = self.down_block4(x)

        x = self.max1(s4)#F.max_pool2d(s4, 2)
        s5 = self.down_block5(x)

        #Outputs of blocks are saved in s1,s2,s3,s4,s5

        x = self.up_block4(s5, s4)

        x = self.up_block3(x, s3)

        x = self.up_block2(x, s2)

        x = self.up_block1(x, s1)

        x1 = self.final(x)
        x2 = self.final_cat(x)
        return x1, x2


'''
class U_Net_Categorie(nn.Module):
    """Implements the U-Net in Pytorch, https://arxiv.org/pdf/1505.04597.pdf"""
    # Caution! Modified.

    def __init__(self, num_classes, num_categories ,check_path = ""):

        super(U_Net_Categorie, self).__init__()

        self.num_classes = num_classes
        self.num_categories = num_categories

        #Added BatchNorm because its good :D
        #Also using upscaling instead of upconv. Vlt sagen sie das im FPN_Resnet Paper.


        if (check_path != "" and os.path.exists(check_path)):
            self.load_state_dict(torch.load(check_path), strict=False)
        elif check_path != "":
            print("Checkpoint not found!")
            exit()


        self.down_block1 = Down_U_Net_Block(3, 64)
        self.down_block2 = Down_U_Net_Block(64, 128)
        self.down_block3 = Down_U_Net_Block(128, 256)
        self.down_block4 = Down_U_Net_Block(256, 512)
        self.down_block5 = Down_U_Net_Block(512, 1024)

        self.up_block4 = Up_U_Net_Block(1024, 512)
        self.up_block3 = Up_U_Net_Block(512 , 256)
        self.up_block2 = Up_U_Net_Block(256 , 128)
        self.up_block1 = Up_U_Net_Block(128 , 64)

        self.final = nn.Conv2d(64, self.num_classes, 1)
        self.final_cat = nn.Conv2d(64, self.num_categories, 1)


    def forward(self, x):


        #Down Path
        s1 = self.down_block1(x)

        x = F.max_pool2d(s1, 2)
        s2 = self.down_block2(x)

        x = F.max_pool2d(s2, 2)
        s3 = self.down_block3(x)

        x = F.max_pool2d(s3, 2)
        s4 = self.down_block4(x)

        x = F.max_pool2d(s4, 2)
        s5 = self.down_block5(x)

        #Outputs of blocks are saved in s1,s2,s3,s4,s5

        x = self.up_block4(s5, s4)

        x = self.up_block3(x, s3)

        x = self.up_block2(x, s2)

        x = self.up_block1(x, s1)

        x1 = self.final(x)

        x2 = self.final_cat(x)

        return x1, x2
'''


if __name__ == "__main__":

    net = U_Net(38).cuda()




    from losses.Loss import CrossEntropyLoss

    from time import time
    from tqdm import tqdm
    net = U_Net(20)

    loss = CrossEntropyLoss()

    label = torch.rand((8, 3, 256, 128)).float()
    target = torch.rand(8, 20, 256, 128)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-6)


    start = time()
    for i in tqdm(range(1)):
        optimizer.zero_grad()
        pred = net(label)

        l = loss(pred, target)
        optimizer.step()
    print(time() - start)