
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class conv_unit(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, padding:int=0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )
    
    def forward(self, x:torch.Tensor):
        x = self.conv(x)
        return x



class UNET_model(nn.module):

    def __init__(depth:int, in_channels:int, out_channels:int): # out_channels will be number of classes
        super().__init__()

        self.


    def forward(self, x:torch.Tensor):



        return x

