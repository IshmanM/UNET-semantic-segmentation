
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
#
#   !! Make use of OpenCV
#   Save the mask data restructuring in a dir
##############################################

import os
from PIL import Image
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A


class semanticDroneDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, colormap: list, transform: A.Compose = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(image_dir)
        self.colormap = colormap

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))

        image = np.rollaxis(np.array(Image.open(image_path).convert("RGB")), 2, 0)
        raw_mask = np.rollaxis(np.array(Image.open(mask_path).convert("RGB")), 2, 0)

        mask = []
        bitmask = np.empty(raw_mask.shape)
        for label_rgb in self.colormap:
            for color in range(3):
                bitmask[color] = label_rgb[color]
            label_map = np.all(np.equal(raw_mask, bitmask), axis=0)
            mask.append(label_map.astype("float32"))
        mask = np.stack(mask, axis=0)

        print("Mask shape: ", mask.shape)
        return image, mask