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

if __name__ == "__main__":

    load_dotenv('.env')

    PATCH_SIZE = (1000, 1000)
    STEP = 1000

    IMAGE_SAVE_TYPE = os.environ["IMAGE_SAVE_TYPE"]
    MASK_SAVE_TYPE = os.environ["MASK_SAVE_TYPE"]

    PATCHIFY_QUEUE = ["TRAIN_IMAGES", "TRAIN_MASKS", 
                      "TEST_IMAGES", "TEST_MASKS", 
                      "VALIDATION_IMAGES", "VALIDATION_MASKS"]

    for image_set in PATCHIFY_QUEUE:
        
        images_dir = os.environ[image_set + "_DIR"]
        patches_dir = os.environ["PATCHIFIED_" + image_set + "_DIR"]

        patchify_images(images_dir=images_dir, 
                        patches_dir=patches_dir, 
                        save_type=IMAGE_SAVE_TYPE, 
                        patch_size=PATCH_SIZE, 
                        step=STEP)
 
                    
