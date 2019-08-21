import torch
import torchvision
from datasets import Cityscapes as cityscape
import re
import  numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


dataset = cityscape.Cityscape("C:\\Users\\mehrt\\Datasets\\Cityscape", split="val", scale_size=(1026, 513),
                              subset=None, return_path=True)

for i,[img, mask, t_imgpath, t_maskpath] in enumerate(dataset):

    pil_imgpath = re.sub("Cityscape", "Cityscape1026", str(t_imgpath))
    pil_maskpath = re.sub("Cityscape", "Cityscape1026", str(t_maskpath))


    Path(Path(pil_imgpath).parent).mkdir(parents=True,exist_ok=True)
    Path(Path(pil_maskpath).parent).mkdir(parents=True, exist_ok=True)

    try:
        mask = mask.numpy()
        mask = np.uint8(mask)
        pilmask = Image.fromarray(mask)
        pilimg = torchvision.transforms.functional.to_pil_image(img, mode="RGB")
    except:
        print(pil_imgpath)

    pilmask.save(pil_maskpath)
    pilimg.save(pil_imgpath)
    print(str(i)+"/"+str(len(dataset)))
