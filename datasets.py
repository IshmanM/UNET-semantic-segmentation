
# @author: Ishman Mann
# @date: 03/12/2022
# 
# @description:
#   Utils for UNET model training
#
# @resources:
#   Institute of Computer Graphics and Vision. (2019). Semantic Drone Dataset. 
#       Graz University of Technology. Retrieved from http://dronedataset.icg.tugraz.at
#
# @notes:
#
#
# @ToDo:
#   Add image transformations
#   Need to convert RGB masks into multichannel input with num_classes channels. Pixel of a class's channel = 1 if pixel is of that class.
#       RGB -> 
##############################################

import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class semanticDroneDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, colormap: list, transform: function = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.colormap = colormap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", ".jpg"))
        
        image = np.array(Image.open(image_path).convert("RGB"))
        raw_mask = np.array(Image.open(mask_path).convert("RGB"))

        mask = []
        for color in self.colormap:
            map = np.all(np.equal(image, color), axis=0)
            mask.append(map.astype("float32"))
        mask = np.stack(mask, axis=0)

        return image, mask