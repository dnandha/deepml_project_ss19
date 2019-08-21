import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import random
from collections import namedtuple

#Copied from cityscapescripts/labels
#---------------------------------------------------------------------------------------------
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , False        , (119, 11, 32) ),
    #Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , True         , (  0,  0,142) ),
]

trainid_to_label = {label.trainId:label for label in labels}
trainid_to_color = {label.trainId:label.color for label in labels}

#---------------------------------------------------------------------------------------------
def toRGB(input_mask):

    mask = input_mask.clone()

    if len(mask.size()) == 3: #Falls noch eine Batchdimension da ist
        mask = mask.squeeze(0)

    rgbmask = torch.zeros([3, *mask.size()])

    for i in range(20):
        if i != 19:
            color = trainid_to_label[i].color
        else:
            color = trainid_to_label[255].color

        i = mask == i
        rgbmask[0][i] = color[0]
        rgbmask[1][i] = color[1]
        rgbmask[2][i] = color[2]

    return rgbmask


# Creates an RGB Pil Image of a given mask
def toPil(mask):
    return Image.fromarray(toRGB(mask).byte().permute(1, 2, 0).numpy())


# Creates a nice visualization of a given image, ground truth mask and prediction
# Input: img = torch.Tensor.size = [3,H,W], mask =[H,W], pred=[H,W]
def compareImageRGB(img, mask, pred):
    img = img * 255
    img = img.byte()

    pred[mask == 255] = 255

    diff = mask == pred
    diff *= 255

    diff = torch.stack([diff, diff, diff], 0)
    mask = toRGB(mask).byte()
    pred = toRGB(pred).byte()

    finalimg = torch.cat([img, mask, pred, diff], 2)

    return finalimg


# Returns an RGB PIL image of the method above
def compareImgPil(img, mask, pred):
    return Image.fromarray(compareImageRGB(img, mask, pred).permute(1, 2, 0).numpy())


class Cityscape(Dataset):
    num_train_classes = 19


    def __init__(self, path, split="fine", subset=None, img_transforms=None, mask_transforms=None,
                 crop_size=None, random_crops=True,scale_size=None, random_scale=False, random_flip=False, return_path=False):


        assert split in ["fine", "coarse", "fc", "val"]
        self.mean_std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.path = Path(path)
        self.split = split
        self.imgs = []
        self.masks = []
        self.scale_size = scale_size
        self.random_scale = random_scale
        self.random_flip = random_flip
        self.crop_size = crop_size
        self.random_crops = random_crops
        self.return_path = return_path

        if img_transforms is None:
            self.img_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:
            self.img_trans = img_transforms

        if mask_transforms is None:
            pass

        self.path_coarse = self.path / "leftImg8bit_trainextra" / "leftImg8Bit" / "train_extra"
        self.path_gtCoarse = self.path / "gtCoarse" / "train_extra"

        self.path_gtFine = self.path / "gtFine_trainvaltest" / "gtFine"
        self.path_gtTrain = self.path_gtFine / "train"
        self.path_gtTest = self.path_gtFine / "val"

        self.path_train_val = self.path / "leftImg8bit_trainvaltest" / "leftImg8bit"
        self.path_ImgFineTrain = self.path_train_val / "train"
        self.path_ImgFineTest = self.path_train_val / "val"

        if split == "fine":
            self.imgs = list(self.path_ImgFineTrain.glob("*/*.png"))
            self.masks = list(self.path_gtTrain.glob("*/*_gtFine_labelIds.png"))

        if split == "coarse":
            self.imgs = list(self.path_coarse.glob("*/*.png"))
            self.masks = list(self.path_gtCoarse.glob("*/*_gtCoarse_labelIds.png"))

        if split == "fc":
            self.imgs = list(self.path_ImgFineTrain.glob("*/*.png")) + list(self.path_coarse.glob("*/*.png"))
            self.masks = list(self.path_gtTrain.glob("*/*_gtFine_labelIds.png")) + list(
                self.path_gtCoarse.glob("*/*_gtCoarse_labelIds.png"))

        # Load val data
        if split == "val":
            self.imgs = list(self.path_ImgFineTest.glob("*/*.png"))
            self.masks = list(self.path_gtTest.glob("*/*_gtFine_labelIds.png"))

        if subset is not None:
            self.imgs = [self.imgs[i] for i in subset]
            self.masks = [self.masks[i] for i in subset]

    def __len__(self):
        return len(self.imgs)

    def load_original(self, index):
        img, _ = self.__getitem__(index)

        img  *= torch.tensor(self.mean_std[1]).reshape(-1,1,1)
        img += torch.tensor(self.mean_std[0]).reshape(-1,1,1)

        return img

    def __getitem__(self, item):

        img = Image.open(self.imgs[item])
        mask = Image.open(self.masks[item])

        if self.scale_size is not None:
            if self.random_scale is False:
                img = img.resize(self.scale_size, Image.BILINEAR)
                mask = mask.resize(self.scale_size, Image.NEAREST)
            else:
                tmp_wd, tmp_ht = img.size

                rd_scale = (random.random() * (self.scale_size[1] - self.scale_size[0])) + self.scale_size[0]

                tmp_wd = int(32 * round((tmp_wd * rd_scale)/32)) #Magic number :D Bild muss 5 mal durch 2 teilbar sein (wegen Maxpool)
                tmp_ht = int(32 * round((tmp_ht * rd_scale)/32))
                img = img.resize((tmp_wd, tmp_ht), Image.BILINEAR)
                mask = mask.resize((tmp_wd, tmp_ht), Image.NEAREST)

        img = self.img_trans(img)
        mask = torch.from_numpy(np.asarray(mask)).long()

        if self.crop_size is not None:
             if self.random_crops:
                dim_x = img.size()[1] #Width
                dim_y = img.size()[2] #Height

                if self.crop_size[0] < dim_x:
                    random_x = random.randint(0, dim_x - self.crop_size[0] - 1)
                else:
                    random_x = 0

                if self.crop_size[1] < dim_y:
                    random_y = random.randint(0, dim_y - self.crop_size[1] - 1)
                else:
                    random_y = 0

                img = img[:, random_x:random_x + self.crop_size[0], random_y:random_y + self.crop_size[1]]
                mask = mask[random_x:random_x + self.crop_size[0], random_y:random_y + self.crop_size[1]]
             else:
                img = img[:, self.crop_size[0][1]:self.crop_size[1][1], self.crop_size[0][0]:self.crop_size[1][0]]
                mask = mask[self.crop_size[0][1]:self.crop_size[1][1], self.crop_size[0][0]:self.crop_size[1][0]]


        if self.random_flip and random.random() >= 0.5:
            img = torch.flip(img, [2])
            mask = torch.flip(mask, [1])

        # Replace the Ids! Ignoreindex is 255
        for i in range(0, 34):
            if (labels[i].trainId != 255):
                mask[mask == i] = labels[i].trainId

            else:
                mask[mask == i] = 255

        if self.return_path:
            return [img, mask, self.imgs[item], self.masks[item]]
        else:
            return [img, mask]


if __name__ == "__main__":
    import time
    import time

    dataset = Cityscape("C:\\Users\\mehrt\\Datasets\\Cityscape", split="fine", scale_size=(1024, 512), subset=[i for i in range(3)], random_flip=True)
    data_loader = DataLoader(dataset, 1, num_workers=4)


    for [img, mask] in data_loader:

        img = img.squeeze(0)

        img *= torch.tensor(Cityscape.mean_std[1]).unsqueeze(1).unsqueeze(1)
        img += torch.tensor(Cityscape.mean_std[0]).unsqueeze(1).unsqueeze(1)
        img *= 255

        rgbmask = toRGB(mask)

        final = torch.cat([img, rgbmask], 1)
        pil_img = Image.fromarray(final.byte().permute(1, 2, 0).numpy())
        pil_img.show()
        time.sleep(5)