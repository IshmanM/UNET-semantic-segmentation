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
#   !! Convert labels to images and save in test(), and cat using unpatchify
#   !! Save model checkpoints
#   !! Save metrics
#
#
#   Save memory usage, e.g. run patchify for smaller batches, etc...
##############################################


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import transform_multiple as TFM

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

from utils import *
from dotenv import load_dotenv


load_dotenv('.env')

# Globals and hyperparameters

COLORMAP_PATH = os.environ["COLORMAP_PATH"]

PATCHIFIED_TRAIN_IMAGES_DIR = os.environ["PATCHIFIED_TRAIN_IMAGES_DIR"]
PATCHIFIED_TRAIN_MASKS_DIR = os.environ["PATCHIFIED_TRAIN_MASKS_DIR"]
PATCHIFIED_VALIDATION_IMAGES_DIR = os.environ["PATCHIFIED_VALIDATION_IMAGES_DIR"]
PATCHIFIED_VALIDATION_MASKS_DIR =os.environ["PATCHIFIED_VALIDATION_MASKS_DIR"]

PATCH_WIDTH = int(os.environ["PATCH_WIDTH"])
PATCH_HEIGHT = int(os.environ["PATCH_HEIGHT"])
IMAGE_SAVE_TYPE = os.environ["IMAGE_SAVE_TYPE"]
MASK_SAVE_TYPE = os.environ["MASK_SAVE_TYPE"]

MODEL_LOAD_PATH = None
MODEL_SAVE_PATH = "models\\model_v1\\model\\model.pth"
HYPERPARAMETER_SAVE_PATH = "models\\model_v1\\model.json"
TRAIN_LOGS_DIR = "models\\model_v1\\train_logs"

NUM_WORKERS = 4
PIN_MEMORY = True
BATCH_SIZE = 4
NUM_EPOCHS = 1
MODEL_IN_CHANNELS = 3
MODEL_HIDDEN_CHANNELS = [16, 32, 64, 128]
                      # [64, 128, 256, 512]
MODEL_CONV_PADDING = 1
SGD_LEARNING_RATE = 0.01
SGD_MOMENTUM = 0.90
SGD_DAMPENING = 0.0


if __name__ == "__main__":

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", DEVICE)

    torch.manual_seed(42) # Set random seed

    # Load and preprocess data

    COLORMAP_DF = pd.read_csv(COLORMAP_PATH)

    COLORMAP = COLORMAP_DF.loc[:,[" r"," g"," b"]].values.tolist()
    CLASSES = COLORMAP_DF.loc[:,"name"].values.tolist()
    NUM_CLASSES = COLORMAP_DF.shape[0]
    
    train_transforms = [TFM.center_crop(output_size=(PATCH_WIDTH, PATCH_HEIGHT)),
                        TFM.normalize(mean=[0.0], std=[255.0], inplace=False)]

    val_transforms = [TFM.center_crop(output_size=(PATCH_WIDTH, PATCH_HEIGHT)),
                      TFM.normalize(mean=[0.0], std=[255.0], inplace=False)]

    train_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_TRAIN_IMAGES_DIR, masks_dir=PATCHIFIED_TRAIN_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=True, transforms=train_transforms,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
    val_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_VALIDATION_IMAGES_DIR, masks_dir=PATCHIFIED_VALIDATION_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=False, transforms=val_transforms,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )
   
    # Save hyperparameters for reference

    hyperperameters = {
        PATCH_WIDTH: PATCH_WIDTH,
        PATCH_HEIGHT: PATCH_HEIGHT,
        NUM_WORKERS: NUM_WORKERS,
        PIN_MEMORY: PIN_MEMORY,
        BATCH_SIZE: BATCH_SIZE,
        NUM_EPOCHS: NUM_EPOCHS,
        MODEL_IN_CHANNELS: MODEL_IN_CHANNELS, 
        MODEL_HIDDEN_CHANNELS: MODEL_HIDDEN_CHANNELS,
        MODEL_CONV_PADDING: MODEL_CONV_PADDING,
        SGD_LEARNING_RATE: SGD_LEARNING_RATE,
        SGD_MOMENTUM: SGD_MOMENTUM,
        SGD_DAMPENING: SGD_DAMPENING,
    }

    save_dict_as_json(save_path=HYPERPARAMETER_SAVE_PATH, data=hyperperameters)

    # Instantiate UNET_model, optimizer, and loss function. Optionally load a saved model

    model = UNET_model(
        in_channels=MODEL_IN_CHANNELS, 
        out_channels=NUM_CLASSES, 
        hidden_channels=MODEL_HIDDEN_CHANNELS, 
        conv_padding=MODEL_CONV_PADDING
    ).to(DEVICE)

    optimizer = torch.optim.SGD(params=model.parameters(), lr=SGD_LEARNING_RATE, 
                                momentum=SGD_MOMENTUM, dampening=SGD_DAMPENING)
    
    loss_function = nn.CrossEntropyLoss()

    if MODEL_LOAD_PATH != None:
        last_epoch, last_train_loss = load_model(model_path=MODEL_LOAD_PATH, model=model, optimizer=optimizer)
    else:
        last_epoch = 0

    # Train model
    
    scaler = cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch}/{NUM_EPOCHS}")

        train_loss, train_accuracy, train_dice_coeff = train(
            model=model, dataloader=train_loader, 
            loss_function=loss_function, 
            optimizer=optimizer, 
            scaler=scaler, 
            device=DEVICE
        )

        save_model(save_path=MODEL_SAVE_PATH, loss=train_loss, 
                   epoch=(epoch + last_epoch), model=model, optimizer=optimizer)

        save_metrics(save_dir=TRAIN_LOGS_DIR, name='train', epoch=epoch, 
                     loss=train_loss, accuracy=train_accuracy, dice_coeff=train_dice_coeff)
        
        val_loss, val_accuracy, val_dice_coeff = test(
            model=model, dataloader=val_loader, 
            loss_function=loss_function, 
            device=DEVICE
        )
        
        save_metrics(save_dir=TRAIN_LOGS_DIR, name='validation', epoch=epoch, 
                loss=val_loss, accuracy=val_accuracy, dice_coeff=val_dice_coeff)
