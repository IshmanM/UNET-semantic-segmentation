
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trans


class conv_unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=padding),
            nn.ReLU()   
        )
        self.norm = nn.BatchNorm2d(num_features=out_channels)
    
    def forward(self, ):
        x = self.conv(x)
        x = self.conv(x)
        x = self.norm(x)
        return x


class encode_unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_padding: int = 0):
        super().__init__()

        self.conv = conv_unit(in_channels=in_channels, out_channels=out_channels, padding=conv_padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        skip_layer = x
        x = self.pool(x)
        
        return x, skip_layer


class decode_unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, conv_padding: int = 0):
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels=in_channels,out_channels=in_channels/2)
        
        self.conv = conv_unit(in_channels=in_channels/2, out_channels=out_channels, padding=conv_padding)

    def forward(self, x: torch.Tensor, skip_layer: torch.Tensor):

        x = self.up_conv(x)

        if skip_layer.shape != x.shape:
            crop = trans.CenterCrop(size=x.shape[-2:]) # !! might use overlap tile strategy to prevent pixels from being cut out in outut
            skip_layer = crop(skip_layer)
        x = torch.cat(tensors=[skip_layer, x], dim=1)

        x = self.conv(x)

        return x

    

class UNET_model(nn.module):

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: list[int]): # out_channels will be number of classes
        super().__init__()

        # self.down_steps = nn.ModuleList()
        # self.up_steps = nn.ModuleList()

        # for channels in hidden_channels:
            


    def forward(self, x: torch.Tensor):



        return x

