
# @author: Ishman Mann
# @date: 03/12/2022
# 
# @description:
#   Dataset class for Semantic Drone Dataset, compatible with PyTorch DataLoader.
#
# @resources:
#   Institute of Computer Graphics and Vision. (2019). Semantic Drone Dataset. 
#       Graz University of Technology. Retrieved from http://dronedataset.icg.tugraz.at
#
# @notes:
#
#
# @ToDo:
#   Consider dynamic patchifying
#
##############################################

import os
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transform_multiple as TFM


class semanticDroneDataset(Dataset):
    """
    Dataset class to access images and masks from the Semantic Drone Dataset. 
    RGB masks are converted to multi-channel masks for multi-class semantic segmentation.
    Inputted transformations are applied to the images and converted masks.
    """
    def __init__(
        self, image_dir: str, mask_dir: str, image_save_type: str, mask_save_type: str, 
        colormap: list[int], transforms: list[TFM.transform_multiple] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.mask_save_type = mask_save_type
        self.image_save_type = image_save_type
        self.image_subpaths = os.listdir(image_dir)
        self.colormap = colormap
        self.transforms = transforms

    def __len__(self):
        return len(self.image_subpaths)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.image_subpaths[index])
        filename = self.image_subpaths[index].replace(('.' + self.image_save_type), '')
        mask_path = os.path.join(self.mask_dir, (filename + '.' + self.mask_save_type))
        
        image = np.rollaxis(np.array(Image.open(image_path).convert("RGB"), dtype=np.float16), 2, 0)
        rgb_mask = np.rollaxis(np.array(Image.open(mask_path).convert("RGB"), dtype=np.float16), 2, 0)

        image = torch.from_numpy(image)

        # Generate multi-channel mask from the rgb mask
        mask = []
        bitmask = np.empty(shape=rgb_mask.shape, dtype=np.float16)
        for label_rgb in self.colormap:
            for color in range(3):
                bitmask[color] = label_rgb[color]
            label_map = np.all(np.equal(rgb_mask, bitmask), axis=0)
            mask.append(label_map.astype("float16"))
        mask = torch.from_numpy(np.stack(mask, axis=0))

        # Augment image and mask
        for transform in self.transforms:
            if transform.modifies_vals_or_idxs == "vals":
                transformed_tensors = transform(image=image)
                image = transformed_tensors["image"]
            else:
                transformed_tensors = transform(image=image, mask=mask)
                image, mask = transformed_tensors["image"], transformed_tensors["mask"]

        return image, mask, filename