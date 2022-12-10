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
#   Consider dynamic patchifying
#
##############################################

import os
import re
import numpy as np
import json
from PIL import Image
from patchify import patchify, unpatchify
from tqdm.auto import tqdm
import torch
from torch import nn, cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import transform_multiple as TFM
from datasets import semanticDroneDataset

# Image data handling utils

def patchify_images(
    images_dir: str, patches_dir: str, save_type: str, 
    patch_size: tuple, step: int, rgb: bool = True
):
    """
    Save patches of large images in desired directory. 
    Compatible with both RGB and single-channel (typically greyscale) images.
    """
    image_subpaths = os.listdir(images_dir)
    
    if rgb:
        patch_size += (3,) # 3rd dimension for RGB
        convert_value = "RGB"
    else:
        convert_value = "L" # For single-channel images
    
    for image_subpath in tqdm(image_subpaths):
        image_path = os.path.join(images_dir, image_subpath)
        image = np.array(Image.open(image_path).convert(convert_value), dtype=np.uint8)

        patches = patchify(image=image, patch_size=patch_size, step=step)

        for y in range(patches.shape[0]):
            for x in range(patches.shape[1]):
                
                if rgb:
                    patch = patches[y, x, 0, ...]
                else:
                    patch = patches[y, x, ...]

                patch = Image.fromarray(patch)

                patch_subpath = re.sub(
                    '\.[0-9A-Za-z]*', 
                    '', image_subpath
                ) + '_' + str(y) + '_' + str(x) + '.' + save_type
                
                patch_path = os.path.join(patches_dir, patch_subpath)
                patch.save(patch_path)


def unpatchify_images(
    patches_dir: str, images_dir: str, image_size: tuple, 
    save_type: str, rgb: bool = True
):
    """
    """
    
    all_patch_subpaths = os.listdir(patches_dir)

    image_names = set([re.sub(
        '_[0-9]+_[0-9]+\.[0-9A-Za-z]*',
        '', patch_subpath
    ) for patch_subpath in all_patch_subpaths])

    for image_name in tqdm(image_names):
        
        image_patch_subpaths = list(filter(
            re.compile(f'^{image_name}').match, 
            all_patch_subpaths
        ))

        patch_y_idxs = set(
            [re.sub(
                (f'{image_name}_' + '|_[0-9]+\.[0-9A-Za-z]*'), 
                '', patch_subpath
            ) for patch_subpath in image_patch_subpaths]
        )

        patch_x_idxs = set(
            [re.sub(
                (f'{image_name}_[0-9]+_' + '|\.[0-9A-Za-z]*'), 
                '', patch_subpath
            ) for patch_subpath in image_patch_subpaths]
        )

        patches = []
        for y in patch_y_idxs:

            patches_x_layer = []
            for x in patch_x_idxs:
                patch_subpath = list(filter(
                    re.compile(f'{image_name}_{y}_{x}').match,
                    image_patch_subpaths
                ))[0]
                patch_path = os.path.join(patches_dir, patch_subpath)
                if rgb:
                    patch = np.array(Image.open(patch_path).convert("RGB"), dtype=np.uint8)
                    patch = np.expand_dims(patch, axis=0)
                else:
                   patch = np.array(Image.open(patch_path).convert("L"), dtype=np.uint8) 
                patches_x_layer.append(patch)
            patches_x_layer = np.stack(patches_x_layer, axis=0)

            patches.append(patches_x_layer)
        patches = np.stack(patches, axis=0)

        image_shape = (patches.shape[0]*patches.shape[2], patches.shape[1]*patches.shape[3])
        if rgb: 
            image_shape += (3,)

        image = unpatchify(patches=patches, imsize=image_shape)


       


        break




def semanticDroneDataset_dataloader(
    images_dir: str, masks_dir: str, image_save_type: str, mask_save_type: str,
    colormap: list, shuffle: bool = False, transforms: list[TFM.transform_multiple] = None,
    batch_size: int = 1, num_workers = 4, pin_memory = True
):
    """
    Generate train, validation, and test DataLoaders of desired split sizes. 
    """
    ds = semanticDroneDataset(
        image_dir=images_dir, 
        mask_dir=masks_dir, 
        image_save_type=image_save_type, 
        mask_save_type=mask_save_type, 
        colormap=colormap, 
        transforms=transforms
    )

    ds_loader = DataLoader(
        dataset=ds, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
  
    return ds_loader


def save_mask_as_rgb_image(
    mask: torch.Tensor, colormap: list[int], 
    save_dir: str, filename: str, save_type: str
):
    """
    Convert multi-channel masks to RGB masks and save result as an image. 
    """
    save_path = os.path.join(save_dir, (filename + '.' + save_type))

    num_labels = mask.shape[0]
    mask = mask.argmax(dim=0)
    rgb_mask = torch.zeros(size=((3,) + mask.shape))

    for label in range(num_labels):
        label_map = (mask == label).float()

        for color in range(3):
            rgb_mask[color] += label_map*colormap[label][color]

    rgb_mask /= 255.0 # save_image multiplies pixels by 255
    save_image(tensor=rgb_mask, fp=save_path)
        

# Training utils

def dice_coefficient(y: torch.tensor, y_predicted: torch.Tensor):
    """
    Compute dice coefficient, a useful metric for semantic segmentation 
    """
    SMOOTH = 1e-8 # To avoid division by 0
    return (2*(y*y_predicted).sum().item() + SMOOTH) / ((y + y_predicted).sum().item() + SMOOTH)


def metrics(y: torch.Tensor, y_predicted: torch.Tensor, num_labels: int):
    """
    Compute basic accuracy and multiclass dice coefficient
    """
    # Cropping may be necessary if no padding is used in model training
    if y.shape != y_predicted.shape:
        crop = TFM.center_crop(size=y_predicted.shape[-2:])
        y = crop(y)

    accuracy = torch.mean((y == y_predicted).float()).item()

    # Average the dice coefficients computed for each class
    dice_coeff = 0
    for label in range(num_labels):
        label_idxs_in_y = (y == label).float()
        label_idxs_in_y_predicted = (y_predicted == label).float()
        dice_coeff += dice_coefficient(label_idxs_in_y, label_idxs_in_y_predicted)
    dice_coeff /= num_labels

    return (accuracy, dice_coeff)


def train(
    model: nn.Module, dataloader: DataLoader, loss_function: nn.Module, 
    optimizer: torch.optim.Optimizer, scaler: cuda.amp.GradScaler, device: torch.device = "cuda"
):
    """
    Enable train mode and train model for 1 epoch.
    Uses autocasting to improve perfomance & maintain accuracy during mixed precision training.
    Uses gradient scaling to prevent underflow of float16 values.
    """
    model.train()

    loss, accuracy, dice_coeff = 0, 0, 0
    data_size = 0

    dataloader_loop = tqdm(dataloader)
    for x, y, filename in dataloader_loop:
        x, y = x.to(device), y.to(device)
        
        with cuda.amp.autocast():
            # Forward step
            y_logits = model(x)
            y_predicted = torch.softmax(y_logits, dim=1).argmax(dim=1)
        
            # Compute accuracy and loss
            batch_size = y.shape[0] # Batch length may differ for the final batch
            data_size += batch_size

            batch_loss = loss_function(y_logits, y)
            loss += batch_loss.item()*batch_size
            batch_accuracy, batch_dice_coeff = metrics(y.argmax(dim=1), y_predicted, num_labels=y.shape[1])
            batch_accuracy, batch_dice_coeff = batch_accuracy*batch_size, batch_dice_coeff*batch_size
            accuracy += batch_accuracy
            dice_coeff += batch_dice_coeff

        # Backward step
        optimizer.zero_grad()    
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        dataloader_loop.set_postfix(
            train_loss=(loss/data_size), 
            train_accuracy=(accuracy/data_size),
            train_dice_coeff=(dice_coeff/data_size)
        )

    loss /= data_size
    accuracy /= data_size
    dice_coeff /= data_size

    return loss, accuracy, dice_coeff


def test(
    model: nn.Module, dataloader: DataLoader, loss_function: nn.Module, device: torch.device = "cuda", 
    prediction_save_dir: str = None, save_type: str = None, colormap: list[int] = None
):
    """
    Disable train mode and test model. Optionally save predictions to a directory.
    """
    model.eval()

    loss, accuracy, dice_coeff = 0, 0, 0
    data_size = 0

    with torch.no_grad():
        dataloader_loop = tqdm(dataloader)
        for x, y, filename in dataloader_loop:
            x, y = x.to(device), y.to(device)

            # Forward step
            y_logits = model(x)
            y_predicted = torch.softmax(y_logits, dim=1).argmax(dim=1)

            # Compute accuracy and loss
            batch_size = y.shape[0] # Batch length may differ for the final batch
            data_size += batch_size

            loss += loss_function(y_logits, y).item()*batch_size
            batch_accuracy, batch_dice_coeff = metrics(y.argmax(dim=1), y_predicted, num_labels=y.shape[1])
            batch_accuracy, batch_dice_coeff = batch_accuracy*batch_size, batch_dice_coeff*batch_size
            accuracy += batch_accuracy
            dice_coeff += batch_dice_coeff

            # Update prograss bar
            dataloader_loop.set_postfix(
                test_loss=(loss/data_size), 
                test_accuracy=(accuracy/data_size),
                test_dice_coeff=(dice_coeff/data_size)
            )

            # Save predictions
            if prediction_save_dir != None:
                for batch in range(batch_size):
                    save_mask_as_rgb_image(
                        mask=y_predicted[batch], 
                        colormap=colormap, 
                        save_dir=prediction_save_dir,
                        filename=filename[batch],
                        save_type=save_type
                    )

    loss /= data_size
    accuracy /= data_size
    dice_coeff /= data_size

    return loss, accuracy, dice_coeff


def save_model(save_path: str, epoch: int, loss: float, model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Save model and optimizer state dicts, as well as last epoch and loss value.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }

    torch.save(obj=checkpoint, f=save_path)


def load_model(load_path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None):
    """
    Load model and optimizer state dicts from inputted path. 
    Return last epoch and loss value.
    """
    checkpoint = torch.load(f=load_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = int(checkpoint['epoch'])
    loss = float(checkpoint['loss'])

    return epoch, loss


def save_dict_as_json(save_path: str, data: dict):
    """
    Save a python dict as a .json file.
    """
    data = json.dumps(data, indent=4)
    with open(save_path, 'w') as json_file:
        json_file.write(data)


def save_metrics(save_dir: str, name: str , epoch: int, **metrics: float):
    """
    Save metrics for TensorBoard.
    """
    with SummaryWriter(log_dir=save_dir) as writer:
        
        for metric, value in metrics.items():
            tag = f"{metric}/{name}"
            writer.add_scalar(tag=tag, scalar_value=value, global_step=epoch)


