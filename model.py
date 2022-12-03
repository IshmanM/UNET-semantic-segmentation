
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trans


class conv_unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, padding: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=padding),
            nn.ReLU(),  
            nn.BatchNorm2d(num_features=out_channels)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv(x)
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

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: list[int], conv_padding: int = 0): # out_channels will be number of classes
        super().__init__()

        self.encode_units = nn.ModuleList()
        self.decode_units = nn.ModuleList()
        
        for channels in hidden_channels:
            self.encode_units.append(module=encode_unit(in_channels=in_channels, out_channels=channels, conv_padding=conv_padding))
            self.decode_units.insert(index=0, module=decode_unit(in_channels=channels*2, out_channels=channels, conv_padding=conv_padding))
            in_channels = channels

        self.bridge = conv_unit(in_channels=hidden_channels[-1], out_channels=2*hidden_channels[-1], conv_padding=conv_padding)
        self.end_conv = nn.Conv2d(in_channels=hidden_channels[0], out_channels=out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        
        skip_layers = []
        
        for encode_unit in self.encode_units:
            x, skip_layer = self.encode_units[encode_unit](x) 
            skip_layers.append(skip_layer)
        
        x = self.bridge(x)

        for decode_unit in range(len(self.decode_units)):
            x = self.decode_units[decode_unit](x, skip_layer=skip_layer[decode_unit])

        x = self.end_conv(x)

        return x

