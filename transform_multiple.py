
# @author: Ishman Mann
# @date: 03/12/2022
# 
# @description:
#   Transformation classes that apply the same transformation to multiple torch.Tensor objects
#
# @resources:
#
#
# @notes:
#
#
# @ToDo:
#   Create classes that allow instantiation of functional transformation objects, which will be passed into 
#   semanticDroneDataset as a list
#   
#   !! Create additional transformation classes
#
##############################################

from abc import ABC, abstractmethod
import torch
import torchvision.transforms.functional as TF
import random


class transform_multiple(ABC):
    """
    Abstract class for applying the same transformation to multiple torch.Tensor objects.
    If a transformation modifies elements value, it should not modify element indexes, and vice-versa.
    """
    def __init__(self, modifies_vals_or_idxs: str):
        if modifies_vals_or_idxs != "idxs" and modifies_vals_or_idxs != "vals":
            raise ValueError("modifies_vals_or_idxs must be 'vals' or 'idxs'")
        self.modifies_vals_or_idxs = modifies_vals_or_idxs
    
    @abstractmethod
    def __call__(self, **tensors: torch.Tensor):
        pass


class normalize(transform_multiple):
    """
    Transformation for normalizing images and other torch.Tensor objects.
    """
    def __init__(self, mean: list[float] = [0.0], std: list[float] = [255.0], inplace: bool = False):
        super().__init__(modifies_vals_or_idxs="vals")
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, **tensors: torch.Tensor):
        transformed_tensors = {}
        for key, tensor in tensors.items():
            tensor = TF.normalize(tensor, mean=self.mean, std=self.std, inplace=self.inplace)
            transformed_tensors[key] = tensor
        return transformed_tensors


class center_crop(transform_multiple):
    """
    Transformation for cropping height and width (dimensions [-2:]) of inputted torch.Tensor objects. 
    """  
    def __init__(self, output_size: list[int]):
        super().__init__(modifies_vals_or_idxs="idxs")
        self.output_size = output_size

    def __call__(self, **tensors: torch.Tensor):
        transformed_tensors = {}
        for key, tensor in tensors.items():
            tensor = TF.center_crop(tensor, output_size=self.output_size)
            transformed_tensors[key] = tensor
        return transformed_tensors

    