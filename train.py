# @author: Ishman Mann
# @date: 01/12/2022
# 
# @description:
#   
#
# @resources:
#   Olaf, R. et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
#       University of Freiburg. Retrieved from https://arxiv.org/pdf/1505.04597.pdf
#
# @notes:
#
#
# @ToDo:
#   !! Modify utils.py according to input labels shape
#
#   need to adjust run_patchify.py, main.py, utils.py, datasets.py to new semantic_drone_dataset stucture 
##############################################


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import albumentations as A

from model import UNET_model

import shutil
import sklearn
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets
from torchvision import transforms

from timeit import default_timer as timer 
from tqdm.auto import tqdm # for progress bar

from utils import semanticDroneDataset_dataloader
from dotenv import load_dotenv


load_dotenv('.env')

# Globals and hyperparameters

COLORMAP_PATH = os.environ["COLORMAP_PATH"]

PATCHIFIED_TRAIN_IMAGES_DIR = os.environ["PATCHIFIED_TRAIN_IMAGES_DIR"]
PATCHIFIED_TRAIN_MASKS_DIR = os.environ["PATCHIFIED_TRAIN_MASKS_DIR"]
PATCHIFIED_TEST_IMAGES_DIR = os.environ["PATCHIFIED_TEST_IMAGES_DIR"]
PATCHIFIED_TEST_MASKS_DIR = os.environ["PATCHIFIED_TEST_MASKS_DIR"]
PATCHIFIED_VALIDATION_IMAGES_DIR = os.environ["PATCHIFIED_VALIDATION_IMAGES_DIR"]
PATCHIFIED_VALIDATION_MASKS_DIR =os.environ["PATCHIFIED_VALIDATION_MASKS_DIR"]

PATCH_WIDTH = int(os.environ["PATCH_WIDTH"])
PATCH_HEIGHT = int(os.environ["PATCH_HEIGHT"])
IMAGE_SAVE_TYPE = os.environ["IMAGE_SAVE_TYPE"]
MASK_SAVE_TYPE = os.environ["MASK_SAVE_TYPE"]

NUM_WORKERS = 4
PIN_MEMORY = True
BATCH_SIZE = 1
MODEL_IN_CHANNELS = 3
MODEL_HIDDEN_CHANNELS = [64, 128, 256, 512]
MODEL_CONV_PADDING = 1
SGD_LEARNING_RATE = 0.01
SGD_MOMENTUM = 0.99
SGD_DAMPENING = 0.0

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) # Set random seed

    # Load and preprocess data

    COLORMAP_DF = pd.read_csv(COLORMAP_PATH)

    COLORMAP = COLORMAP_DF.loc[:,[" r"," g"," b"]].values.tolist()
    CLASSES = COLORMAP_DF.loc[:,"name"].values.tolist()
    NUM_CLASSES = COLORMAP_DF.shape[0]

    train_transform = A.Compose(
       [A.resize(width=PATCH_WIDTH, height=PATCH_HEIGHT),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        A.pytorch.ToTensorV2()]
    )
    validation_transform = A.Compose(
       [A.resize(width=PATCH_WIDTH, height=PATCH_HEIGHT),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        A.pytorch.ToTensorV2()]
    )
    test_transform = A.Compose(
       [A.resize(width=PATCH_WIDTH, height=PATCH_HEIGHT),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        A.pytorch.ToTensorV2()]
    )

    train_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_TRAIN_IMAGES_DIR, masks_dir=PATCHIFIED_TRAIN_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=True, transform=train_transform,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    validation_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_VALIDATION_IMAGES_DIR, masks_dir=PATCHIFIED_VALIDATION_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=False, transform=validation_transform,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_TEST_IMAGES_DIR, masks_dir=PATCHIFIED_TEST_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=False, transform=test_transform,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Instantiate UNET_model, optimizer, and loss function

    model = UNET_model(in_channels=MODEL_IN_CHANNELS, out_channels=NUM_CLASSES, 
                       hidden_channels=MODEL_HIDDEN_CHANNELS, conv_padding=MODEL_CONV_PADDING)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=SGD_LEARNING_RATE, 
                                momentum=SGD_MOMENTUM, dampening=SGD_DAMPENING)
    
    loss_function = nn.CrossEntropyLoss()

