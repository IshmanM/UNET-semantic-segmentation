# @author: Ishman Mann
# @date: 09/12/2022
# 
# @description:
#   
#
# @resources:
# 
#
# @notes:
#
#
# @ToDo:
#   
#
##############################################

import os
import pandas as pd
import torch
from utils import *
import transform_multiple as TFM
from model import UNET_model


# Globals and hyperparameters

COLORMAP_PATH = os.environ["COLORMAP_PATH"]

PATCHIFIED_TEST_IMAGES_DIR = os.environ["PATCHIFIED_TEST_IMAGES_DIR"]
PATCHIFIED_TEST_MASKS_DIR = os.environ["PATCHIFIED_TEST_MASKS_DIR"]

PATCH_WIDTH = int(os.environ["PATCH_WIDTH"])
PATCH_HEIGHT = int(os.environ["PATCH_HEIGHT"])
IMAGE_SAVE_TYPE = os.environ["IMAGE_SAVE_TYPE"]
MASK_SAVE_TYPE = os.environ["MASK_SAVE_TYPE"]

MODEL_LOAD_PATH = "models\\model_v1\\model\\model.pth"

PREDICTION_MASKS_SAVE_DIR = "predictions\\model_v1\\masks"
PREDICTION_METRICS_SAVE_PATH = "predictions\\model_v1\\metrics.txt"

NUM_WORKERS = 4
PIN_MEMORY = True
BATCH_SIZE = 4
NUM_EPOCHS = 1
MODEL_IN_CHANNELS = 3
MODEL_HIDDEN_CHANNELS = [16, 32, 64, 128]
MODEL_CONV_PADDING = 1


if __name__ == "__main__":

    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: ", DEVICE)
    
    torch.manual_seed(42) # Set random seed

    # Load and preprocess data

    COLORMAP_DF = pd.read_csv(COLORMAP_PATH)
    COLORMAP = COLORMAP_DF.loc[:,[" r"," g"," b"]].values.tolist()
    NUM_CLASSES = COLORMAP_DF.shape[0]
    
    test_transforms = [TFM.center_crop(output_size=(PATCH_WIDTH, PATCH_HEIGHT)),
                       TFM.normalize(mean=[0.0], std=[255.0], inplace=False)]

    test_loader = semanticDroneDataset_dataloader(
        images_dir=PATCHIFIED_TEST_IMAGES_DIR, masks_dir=PATCHIFIED_TEST_MASKS_DIR,
        image_save_type=IMAGE_SAVE_TYPE, mask_save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP, shuffle=False, transforms=test_transforms,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Instantiate UNET_model and loss function. Load the saved model.

    model = UNET_model(
        in_channels=MODEL_IN_CHANNELS, 
        out_channels=NUM_CLASSES, 
        hidden_channels=MODEL_HIDDEN_CHANNELS, 
        conv_padding=MODEL_CONV_PADDING
    ).to(DEVICE)

    load_model(load_path=MODEL_LOAD_PATH, model=model)
    
    loss_function = nn.CrossEntropyLoss()

    # Test model

    test_loss, test_accuracy, test_dice_coeff = test(
        model=model, 
        dataloader=test_loader, 
        loss_function=loss_function, 
        device=DEVICE,
        prediction_save_dir=PREDICTION_MASKS_SAVE_DIR,
        save_type=MASK_SAVE_TYPE,
        colormap=COLORMAP
    )
    
    with open(PREDICTION_METRICS_SAVE_PATH, 'w') as metrics_file:
        
        metrics_file.writelines('\n'.join([
            f"test_loss: {test_loss}",
            f"test_accuracy: {test_accuracy}",
            f"test_dice_coeff: {test_dice_coeff}"
        ]))


    
    
