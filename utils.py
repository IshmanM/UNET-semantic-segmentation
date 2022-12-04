# @author: Ishman Mann
# @date: 03/12/2022
# 
# @description:
#   Utils for UNET model training
#
# @resources:
#   Olaf, R. et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
#       University of Freiburg. https://arxiv.org/pdf/1505.04597.pdf
#
#   Ekin, T. (2019). Metrics to Evaluate your Semantic Segmentation Model.
#       Towards Data Science. https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
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
#   !!! Need to see and acount for what the dataset labels shape looks like!
##############################################

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trans


# Data loading and preprocessing utils







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


def train_loop(model: nn.Module, dataloader: DataLoader, loss_function: nn.Module, 
               optimizer: torch.optim.Optimizer, device: torch.device = "cuda"):
    """
    Enable train mode and train model for 1 epoch
    """
    x = x.to(device)
    y = y.to(device)

    model.train()

    loss, accuracy, dice_coeff = 0, 0, 0
    data_size = 0

    for x, y in dataloader:
        
        # Forward step
        y_logits = model(x)                                                                 # !! is this shaping valid?
        y_predicted = torch.softmax(y_logits, dim=1).argmax(dim=1)                          # !! is this shaping valid?

        # Compute accuracy and loss
        batch_size = x.shape[0] # Batch length may differ for the final batch
        data_size += batch_size

        batch_loss = loss_function(y_logits, y)
        loss += batch_loss*batch_size
        accuracy, dice_coeff += metrics(y, y_predicted, num_labels=x.shape[1])*batch_size

        # Backward step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
    
    loss /= data_size
    accuracy /= data_size
    dice_coeff /= data_size

    return loss, accuracy, dice_coeff


def test_loop(model: nn.Module, dataloader: DataLoader, loss_function: nn.Module,
              device: torch.device = "cuda"):
    """
    Disable train mode and test model
    """
    x = x.to(device)
    y = y.to(device)

    model.eval()

    loss, accuracy, dice_coeff = 0, 0, 0
    data_size = 0

    for x, y in dataloader:

        # Forward step
        y_logits = model(x)                                                                # !! is this shaping valid?
        y_predicted = torch.softmax(y_logits, dim=1).argmax(dim=1)                         # !! is this shaping valid?

        # Compute accuracy and loss
        batch_size = x.shape[0] # Batch length may differ for the final batch
        data_size += batch_size

        loss += loss_function(y_logits, y)*batch_size
        accuracy, dice_coeff += metrics(y, y_predicted, num_labels=x.shape[1])*batch_size

    loss /= data_size
    accuracy /= data_size
    dice_coeff /= data_size

    return loss, accuracy, dice_coeff