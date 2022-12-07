# @author: Ishman Mann
# @date: 05/12/2022
# 
# @description:
#   Save patchified dataset images before model training 
#
# @resources:
#
#
# @notes:
#
#
# @ToDo:
#   Consider dynamic patchifying
#
##############################################

from utils import patchify_images
from dotenv import load_dotenv
import os


load_dotenv('.env')

# Hyperpetameters and globals

PATCH_WIDTH = int(os.environ["PATCH_WIDTH"])
PATCH_HEIGHT = int(os.environ["PATCH_HEIGHT"])
PATCH_SIZE = (PATCH_WIDTH, PATCH_HEIGHT)

STEP = 1000

IMAGE_SAVE_TYPE = os.environ["IMAGE_SAVE_TYPE"]
MASK_SAVE_TYPE = os.environ["MASK_SAVE_TYPE"]

PATCHIFY_QUEUE = [{"name":"TRAIN_IMAGES", "save_type":IMAGE_SAVE_TYPE}, 
                  {"name":"TRAIN_MASKS", "save_type":MASK_SAVE_TYPE}, 
                  {"name":"TEST_IMAGES", "save_type":IMAGE_SAVE_TYPE}, 
                  {"name":"TEST_MASKS", "save_type":MASK_SAVE_TYPE}, 
                  {"name":"VALIDATION_IMAGES", "save_type":IMAGE_SAVE_TYPE}, 
                  {"name":"VALIDATION_MASKS", "save_type":MASK_SAVE_TYPE}]
                  

if __name__ == "__main__":
                      
    for image_set in PATCHIFY_QUEUE:
        
        images_dir = os.environ[image_set["name"] + "_DIR"]
        patches_dir = os.environ["PATCHIFIED_" + image_set["name"] + "_DIR"]

        patchify_images(images_dir=images_dir, 
                        patches_dir=patches_dir, 
                        save_type=image_set["save_type"], 
                        patch_size=PATCH_SIZE, 
                        step=STEP)
 
                    
