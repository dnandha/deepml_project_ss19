from torch.utils.data import Dataset
import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import json
import random
import numpy as np

dataset_stats = {}
dataset_stats["mean"]   = [0.4192, 0.4587, 0.4701]
dataset_stats["var"] = [0.0697, 0.0758, 0.0919]
dataset_stats["num_classes"] = 66


class Vistas(Dataset):
    def __init__(self, path, split="train", subset=None, img_transforms=None, mask_transforms=None,
                 crop_size=None, random_crops=True, scale_size=None, random_scale=False, random_flip=False,
                 return_path=False):
        """ params:
        @path            path to the dataset/config.json
        @split           Has to be one of 'train', 'val', 'test'. Loads the corresponding data
        @subset          Has to be a list of ints. Specifies which image should be in the dataset, by the given indices. If left at None loads all images
        @img_transforms  Transform which are applied to the image
        @mask_transforms Transforms which are applied to the mask
        @crop_size       Has to be a tuple of 4 ints (x1,y1,x2,y2). Crops the square with the given upper left and lower right coordinates.
        @random_crops    If set to true, crop_size has to be only a tuple of 2 ints. Generates a random crop from a given image with given height and width.
        @scale_size      Scale images to the given size (height, width) in pixels. If height or width <=1: Scale Proportional
        @random_scale    Given (lowBound, upperBound) randomly scale the images between (lowBound,lowBound)and (upperBound, upperBound). Floats expected.
        @random_flip     Given a number between 0 and 1, flips the image horizontally with given chance. Instead of 0, you can alternativly pass a False.
        @return_path     In addtion to image and mask, also return the file_path for them (as str).
        """
        assert split in ["train", "val", "test"]

        self.random_flip = random_flip
        self.random_scale = random_scale
        self.random_crops = random_crops
        self.path = Path(path)
        self.split = split
        self.imgs = []
        self.masks = []
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.return_path = return_path

        if img_transforms is None:
            # todo: remove redundance
            if random_crops:
                self.img_trans = transforms.Compose([
                    transforms.RandomCrop(self.crop_size, pad_if_needed=True),
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats["mean"], dataset_stats["var"])
                ])
            else:
                self.img_trans = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(dataset_stats["mean"], dataset_stats["var"])
                ])

        if mask_transforms is None:
            if random_crops:
                self.mask_trans = transforms.Compose([
                    transforms.RandomCrop(self.crop_size, pad_if_needed=True)
                ])

        config_file = open(self.path/"config.json")
        config = json.load(config_file)
        self.labels = config["labels"]
        self.version = config["version"]

        self.path_train_img = self.path / "training" / "images"
        self.path_train_gt = self.path / "training" / "labels"

        self.path_val_img = self.path / "validation" / "images"
        self.path_val_gt = self.path / "validation" / "labels"

        self.path_test_img = self.path / "testing" / "images"

        # Load images and masks!

        if split == "train":
            tmp_images = list(self.path_train_img.glob("*"))
            tmp_masks = list(self.path_train_gt.glob("*"))
        elif split == "val":
            tmp_images = list(self.path_val_img.glob("*"))
            tmp_masks = list(self.path_val_gt.glob("*"))
        elif split == "test":
            tmp_images = list(self.path_test_img.glob("*"))
            tmp_masks = []
        else:
            print("Wrong split")
            exit()

        if len(tmp_images) == 0:
            print("No images found. Exiting")
            exit()

        if len(tmp_images) == len(tmp_masks):
            for i, j in zip(tmp_images, tmp_masks):
                self.imgs.append(i)
                self.masks.append(j)
        else:
            for i in tmp_images:
                self.imgs.append(i)

        if subset is not None:
            self.imgs = [self.imgs[i] for i in subset]
            self.masks = [self.masks[i] for i in subset]


    def toRGB(self, mask):
        if len(mask.size()) == 3:  # Falls noch eine Batchdimension da ist
            mask = mask.squeeze(0)

        rgbmask = torch.zeros([3, *mask.size()])
        for label_id, label in enumerate(self.labels):
            # set all pixels with the current label to the color of the current label
            indizes = mask==label_id
            rgbmask[0, indizes] = label["color"][0]
            rgbmask[1, indizes] = label["color"][1]
            rgbmask[2, indizes] = label["color"][2]

        return rgbmask


    def toPil(self, mask, categorie=False):
        return Image.fromarray(self.toRGB(mask).byte().permute(1, 2, 0).numpy())


    def load_original(self, index):
        img, _ = self.__getitem__(index)

        img  *= torch.tensor(dataset_stats["var"]).reshape(-1,1,1)
        img += torch.tensor(dataset_stats["mean"]).reshape(-1,1,1)

        return img


    def __version__(self):
        return self.version


    def __len__(self):
        return len(self.imgs)


    def __getitem__(self, item):
        img = Image.open(self.imgs[item])
        if len(self.masks) != 0:
            mask = Image.open(self.masks[item])

        if self.scale_size is not None:
            if self.random_scale is False:
                img = img.resize(self.scale_size, Image.BILINEAR)
                if len(self.masks) != 0:
                    mask = mask.resize(self.scale_size, Image.NEAREST)
            else:
                tmp_wd, tmp_ht = img.size

                rd_scale = (random.random() * (self.scale_size[1] - self.scale_size[0])) + self.scale_size[0]
                #32 because maxpool has to divide 5 times by 2. So we round it to the nearest multiple of 32.
                tmp_wd = int(32 * round((tmp_wd * rd_scale) / 32))
                tmp_ht = int(32 * round((tmp_ht * rd_scale) / 32))
                img = img.resize((tmp_wd, tmp_ht), Image.BILINEAR)
                if len(self.masks) != 0:
                    mask = mask.resize((tmp_wd, tmp_ht), Image.NEAREST)

        img = self.img_trans(img)
        mask = self.mask_trans(mask)

        if len(self.masks) != 0:
            mask = torch.from_numpy(np.array(mask)).long()

        if self.crop_size is not None:
            if not self.random_crops:
                img = img[:, self.crop_size[0][1]:self.crop_size[1][1], self.crop_size[0][0]:self.crop_size[1][0]]
                if len(self.masks) != 0:
                    mask = mask[self.crop_size[0][1]:self.crop_size[1][1], self.crop_size[0][0]:self.crop_size[1][0]]

        if self.random_flip and random.random() >= 0.5:
            img = torch.flip(img, [2])
            if len(self.masks) != 0:
                mask = torch.flip(mask, [1])

        if self.return_path:
            if len(self.masks) != 0:
                return [img, mask, self.imgs[item], self.masks[item]]
            else:
                return [img, self.imgs[item]]
        else:
            if len(self.masks) != 0:
                return [img, mask]
            else:
                return [img]


if __name__ == "__main__":
    from tqdm import tqdm
    import time
    import matplotlib.pyplot as plt
    dataset = Vistas("C:/Users/mehrt/Datasets/mapillary-vistas-dataset_public_v1.1", split="train")




    for [img, mask ] in tqdm(dataset):
        dataset.toPil(mask).show()
        time.sleep(2)
    print(dataset[0][1].size())
    print(dataset[0][3])
    pilimg = dataset.toPil(dataset[0][1])
    pilimg.save("hello.png")
    structured_labels = {}

    for label in dataset.labels:
        sub_labels = label["name"].split("--")
        current_point = structured_labels

        for level, sub_label in enumerate(sub_labels):
            if sub_label in current_point:
                current_point = current_point[sub_label]
            else:
                current_point[sub_label] = {}
                current_point = current_point[sub_label]

    print(structured_labels)

    for level1 in structured_labels.keys():
        print("-"+level1+":")
        for level2 in structured_labels[level1].keys():
            if(len(structured_labels[level1][level2]) > 0):
                print("\t-" + level2 + ":")
            else:
                print("\t-" + level2)
            for level3 in structured_labels[level1][level2].keys():
                print("\t\t -" + level3)
