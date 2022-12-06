# @author: Ishman Mann
# @date: 03/12/2022
# 
# @description:
#   Utils for UNET model training
#
# @resources:
#   Olaf, R. et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
#       University of Freiburg. Retrieved from https://arxiv.org/pdf/1505.04597.pdf
#
#   Ekin, T. (2019). Metrics to Evaluate your Semantic Segmentation Model.
#       Towards Data Science. Retrieved from https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
#
# @notes:
#
#
# @ToDo:
#   When preprocessing data, consider:
#       a) Overlap tile padding strategy to prevent pixels from being cut out in outut, vs. 
#       b) Basic padding  
#   Use batch size of 1, as per the paper, and momentum of 0.99 in optimizer
#
#   !! centerCrop may possibly pose an issue due to misallignment of y predicted when  computing accuracy, 
#       so consider basic padding / img transformation based padding instead of cropping
#       !! However, what if data labels are of smaller size than padded predictions?
#
#   !! Need to handle images in small patches, such as w/patchify. 
#          Best to create a func to do it once and save it in a dir
#
##############################################

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trans
from datasets import semanticDroneDataset
import albumentations as A
from patchify import patchify
import os
from PIL import Image
import numpy as np
import re

# Data loading and preprocessing utils

def patchify_images(images_dir: str, patches_dir: str, save_type: str, patch_size: tuple, step: int, rgb: bool = True):

    image_subpaths = os.listdir(images_dir)
    
    if rgb:
        patch_size += (3,) # 3rd dimension for RGB
    
    for image_subpath in image_subpaths:
        image_path = os.path.join(images_dir, image_subpath)
        image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)

        patches = patchify(image=image, patch_size=patch_size, step=step)

        for x in range(patches.shape[0]):
            for y in range(patches.shape[1]):
                
                if rgb:
                    patch = patches[x, y, 0, ...]
                else:
                    patch = patches[x, y, ...]
                
                print("patches:", patches.shape)
                print("patch:", patch.shape)
                

                patch = Image.fromarray(patch)

                patch_subpath = re.sub('\.[0-9A-Za-z]*','',image_subpath) + '_' + str(x) + str(y) + '.' + save_type
                patch_path = os.path.join(patches_dir, patch_subpath)
                patch.save(patch_path)
            
            



def semanticDroneDataset_dataloaders(
    image_dir: str,
    mask_dir: str,
    colormap: list,
    train_split: float = 0.8,
    validation_split: float = 0.1,
    test_split: float = 0.1,
    batch_size: int = 1,
    num_workers = 4,
    pin_memory = True,
    transform: A.Compose = None):

    if train_split + validation_split + test_split != 1:
        raise ValueError("train_split, validation_split, and test_split must sum to 1")

    ds = semanticDroneDataset(image_dir=image_dir, mask_dir=mask_dir, colormap=colormap, transform=transform)
    train_ds, validation_ds, test_ds = random_split(ds, [train_split, validation_split, test_split])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)

    validation_loader = DataLoader(validation_ds, batch_size=batch_size, shuffle=False, 
                                   num_workers=num_workers, pin_memory=pin_memory)

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, validation_loader, test_loader


# Training utils

def dice_coefficient(y: torch.tensor, y_predicted: torch.Tensor):
    """
    Compute dice coefficient, a useful metric for semantic segmentation 
    """
    SMOOTH = 1e-8 # To avoid division by 0
    return (2*(y*y_predicted).sum()) / ((y + y_predicted).sum() + SMOOTH)


def metrics(y: torch.Tensor, y_predicted: torch.Tensor, num_labels: int):
    """
    Compute basic accuracy and multiclass dice coefficient
    """
    # Cropping may be necessary if no padding is used in model training
    if y.shape != y_predicted.shape:                                       # !! is this comparison valid? Need to see and acount for what the dataset labels shape looks like!
        crop = trans.CenterCrop(size=y_predicted.shape[-2:])
        y = crop(y)

    accuracy = torch.mean((y == y_predicted).float()).item()             # !! is this comparison valid? Need to see and acount for what the dataset labels shape looks like!   

    # Average the dice coefficients computed for each class
    dice_coeff = 0
    for label in range(num_labels):
        label_idxs_in_y = (y == label).float()
        label_idxs_in_y_predicted = (y_predicted == label).float()
        dice_coeff += dice_coefficient(label_idxs_in_y, label_idxs_in_y_predicted)
    dice_coeff /= num_labels

    return accuracy, dice_coeff


def train(model: nn.Module, dataloader: DataLoader, loss_function: nn.Module, 
               optimizer: torch.optim.Optimizer, device: torch.device = "cuda"):
    """
    Enable train mode and train model for 1 epoch
    """
    model.train()

    loss, accuracy, dice_coeff = 0, 0, 0
    data_size = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        # Forward step
        y_logits = model(x)                                                                 # !! is this shaping valid?
        y_predicted = torch.softmax(y_logits, dim=1).argmax(dim=1)                          # !! is this shaping valid?

        # Compute accuracy and loss
        batch_size = y.shape[0] # Batch length may differ for the final batch
        data_size += batch_size

        batch_loss = loss_function(y_logits, y)
        loss += batch_loss*batch_size
        batch_accuracy, batch_dice_coeff = metrics(y.argmax(dim=1), y_predicted, num_labels=y.shape[1])*batch_size
        accuracy += batch_accuracy
        dice_coeff += batch_dice_coeff

        # Backward step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    
    loss /= data_size
    accuracy /= data_size
    dice_coeff /= data_size

    return loss, accuracy, dice_coeff


def test(model: nn.Module, dataloader: DataLoader, loss_function: nn.Module,
              device: torch.device = "cuda"):
    """
    Disable train mode and test model
    """
    model.eval()

    loss, accuracy, dice_coeff = 0, 0, 0
    data_size = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        # Forward step
        y_logits = model(x)                                                                # !! is this shaping valid?
        y_predicted = torch.softmax(y_logits, dim=1).argmax(dim=1)                         # !! is this shaping valid?

        # Compute accuracy and loss
        batch_size = y.shape[0] # Batch length may differ for the final batch
        data_size += batch_size

        loss += loss_function(y_logits, y)*batch_size
        batch_accuracy, batch_dice_coeff = metrics(y.argmax(dim=1), y_predicted, num_labels=y.shape[1])*batch_size
        accuracy += batch_accuracy
        dice_coeff += batch_dice_coeff

    loss /= data_size
    accuracy /= data_size
    dice_coeff /= data_size

    return loss, accuracy, dice_coeff