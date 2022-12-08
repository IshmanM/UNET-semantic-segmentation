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
#   update requirements.txt & import cleanup
#   add more transformations
#
#   !! Convert labels to images and save in test()
#   !! Save model checkpoints
##############################################


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import transform_multiple as TM

from model import UNET_model

import shutil
import sklearn
from sklearn.model_selection import train_test_split

import torch
from torch import nn, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets
from torchvision import transforms

from timeit import default_timer as timer 
from tqdm.auto import tqdm

from utils import semanticDroneDataset_dataloader, train, test
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
NUM_EPOCHS = 1
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
    
    train_transforms = [TM.center_crop(output_size=(PATCH_WIDTH, PATCH_HEIGHT)),
                        TM.normalize(mean=[0.0], std=[255.0], inplace=False)]

    validation_transforms = [TM.center_crop(output_size=(PATCH_WIDTH, PATCH_HEIGHT)),
                             TM.normalize(mean=[0.0], std=[255.0], inplace=False)]

    test_transforms = [TM.center_crop(output_size=(PATCH_WIDTH, PATCH_HEIGHT)),
                       TM.normalize(mean=[0.0], std=[255.0], inplace=False)]

    train_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_TRAIN_IMAGES_DIR, masks_dir=PATCHIFIED_TRAIN_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=True, transforms=train_transforms,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_VALIDATION_IMAGES_DIR, masks_dir=PATCHIFIED_VALIDATION_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=False, transforms=validation_transforms,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    test_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_TEST_IMAGES_DIR, masks_dir=PATCHIFIED_TEST_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=False, transforms=test_transforms,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Instantiate UNET_model, optimizer, and loss function

    model = UNET_model(in_channels=MODEL_IN_CHANNELS, out_channels=NUM_CLASSES, 
                       hidden_channels=MODEL_HIDDEN_CHANNELS, conv_padding=MODEL_CONV_PADDING)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=SGD_LEARNING_RATE, 
                                momentum=SGD_MOMENTUM, dampening=SGD_DAMPENING)
    
    loss_function = nn.CrossEntropyLoss()

    # Training
    
    scaler = cuda.amp.GradScaler()

    epochs_loop = tqdm(range(NUM_EPOCHS))
    for epoch in epochs_loop:
        train_loss, train_accuracy, train_dice_coeff = train(model=model, dataloader=train_loader, 
                                                             loss_function=loss_function, 
                                                             optimizer=optimizer, 
                                                             scaler=scaler, 
                                                             device=DEVICE)
        
        val_loss, val_accuracy, val_dice_coeff = test(model=model, dataloader=val_loader, 
                                                      loss_function=loss_function, 
                                                      device=DEVICE)

        epochs_loop.set_description(f"Epoch: {epoch}/{NUM_EPOCHS}")
        epochs_loop.set_postfix(train_loss=train_loss, train_accuracy=train_accuracy, train_dice_coeff=train_dice_coeff,
                                val_loss=val_loss, val_accuracy=val_accuracy, val_dice_coeff=val_dice_coeff)
