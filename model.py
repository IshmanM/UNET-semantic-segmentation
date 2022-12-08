# @author: Ishman Mann
# @date: 01/12/2022
# 
# @description:
#   Class definitions for UNET_model
#
# @resources:
#   Olaf, R. et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. 
#       University of Freiburg. https://arxiv.org/pdf/1505.04597.pdf
#
# @notes:
#
#
# @ToDo:
#   When preprocessing data, consider:
#       a) Overlap tile padding strategy to prevent pixels from being cut out in outut, vs. 
#       b) Basic padding  
#
#   !! centerCrop may possibly pose an issue due to misallignment of y predicted when computing accuracy, 
#       so consider basic padding / img transformation based padding instead of cropping
##############################################

import torch
from torch import nn
import torchvision.transforms as trans
import transform_multiple as TM


class conv_unit(nn.Module):
    """
    Conv layer units used frequently in UNET
    """
    def __init__(self, in_channels: int, out_channels: int, padding: int = 1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),  
            nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        return x


class encode_unit(nn.Module):
    """
    Encode units downsample and add complexity to input data by adding feature maps. 
    """
    def __init__(self, in_channels: int, out_channels: int, conv_padding: int = 1):
        super().__init__()

        self.conv = conv_unit(in_channels=in_channels, out_channels=out_channels, padding=conv_padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        skip_layer = x        
        x = self.pool(x)
        
        # Skip layers are returned for later use by decode units.
        return x, skip_layer


class decode_unit(nn.Module):
    """
    Decode units upsample input data and use skip connections to incorporate spatial information
    """
    def __init__(self, in_channels: int, out_channels: int, conv_padding: int = 1):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2, padding=0) # padding unneeded
        
        self.conv = conv_unit(in_channels=in_channels, out_channels=out_channels, padding=conv_padding)

    def forward(self, x: torch.Tensor, skip_layer: torch.Tensor):

        x = self.up_conv(x)

        # Skip layers may require cropping before concatenation to passing data if no padding is used on conv units
        if skip_layer.shape != x.shape:
            # crop = TM.center_crop(size=x.shape[-2:])
            # skip_layer = crop(skip_layer)
            x = trans.functional.resize(x, size=skip_layer.shape[-2:])
        x = torch.cat(tensors=(skip_layer, x), dim=1)

        x = self.conv(x)

        return x

    
class UNET_model(nn.Module):
    """
    UNET_model consists of the following stages: Encoding -> Bridge -> Decoding -> End Convlution
    Skip layers link Encoding and Decoding stages
    """
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: list[int] = [64, 128, 256, 512], conv_padding: int = 1): # out_channels will be number of classes
        super().__init__()

        self.encode_units = nn.ModuleList()
        self.decode_units = nn.ModuleList()
        
        for channels in hidden_channels:
            self.encode_units.append(module=encode_unit(in_channels=in_channels, out_channels=channels, conv_padding=conv_padding))
            
            # inserted at index 0 as decode unit shapes will occur in reverse order to encode unit shapes when indexing
            self.decode_units.insert(index=0, module=decode_unit(in_channels=channels*2, out_channels=channels, conv_padding=conv_padding))
            in_channels = channels

        self.bridge = conv_unit(in_channels=hidden_channels[-1], out_channels=2*hidden_channels[-1], padding=conv_padding)
        self.end_conv = nn.Conv2d(in_channels=hidden_channels[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0) # kernel_size=1 => padding=0 

    def forward(self, x: torch.Tensor):
        
        skip_layers = []
        
        for encode_unit in self.encode_units:
            x, skip_layer = encode_unit(x) 

            # inserted at index 0 as skip layers will be accessed in the reverse order of their generation
            skip_layers.insert(0, skip_layer)

        x = self.bridge(x)

        for decode_unit_idx in range(len(self.decode_units)):
            x = self.decode_units[decode_unit_idx](x, skip_layer=skip_layers[decode_unit_idx])

        x = self.end_conv(x)

        return x


if __name__ == "__main__":
    # Forward pass of UNET_model on random tensor, for testing

    INPUT_SHAPE = (5, 3, 571, 571)
    rand = torch.randn(INPUT_SHAPE)
    model = UNET_model(in_channels=3, out_channels=2, hidden_channels=[64, 128, 256, 512], conv_padding=1)
    out = model(rand)

    print("Input Shape: ", INPUT_SHAPE)
    print("Output Shape: ", out.shape)
    print("Model Architecture:\n", model)