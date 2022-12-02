
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class down_unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int = 0):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels)
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        skip_layer = x
        x = self.pool(x)
        
        return x, skip_layer


class up_unit(nn.Module):
    def __init__(self, skip_layer: torch.Tensor, ):
        super().__init__()

        self.skip_layer = skip_layer

    def forward(self, x: torch.Tensor):

        # !! first resize skip_layer to x





        return x

    

class UNET_model(nn.module):

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: list[int]): # out_channels will be number of classes
        super().__init__()

        # self.down_steps = nn.ModuleList()
        # self.up_steps = nn.ModuleList()

        # for channels in hidden_channels:
            


    def forward(self, x: torch.Tensor):



        return x

