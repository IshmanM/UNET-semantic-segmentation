# @author: Ishman Mann
# @date: 01/12/2022
# 
# @description:
#   
#
# @resources:
#   Olaf. R, et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
#       University of Freiburg, Germany. https://arxiv.org/pdf/1505.04597.pdf
#
# @notes:
#
#
# @ToDo:
#
#
##############################################


import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

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



if __name__ == "__main__":

    # Set device as GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set random seeds
    torch.manual_seed(42)




    # Load and preprocess data





    #Instantiate UNET model

    IN_CHANNELS = 1
    OUT_CHANNELS = 2
    HIDDEN_CHANNELS = [64, 128, 256, 512]
    PADDING = 0

    model = UNET_model(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS, conv_padding=PADDING)

    # Set optimizer and loss function 

    LEARNING_RATE = 0.01
    MOMENTUM = 0.99
    DAMPENING = 0.0

    optimizer = torch.optim.SGD(params=model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, dampening=DAMPENING)
    
    loss_fn = nn.CrossEntropyLoss()

