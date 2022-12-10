# @author: Ishman Mann
# @date: 05/12/2022
# 
# @description:
#   Unpatchify predicted RGB masks after model testing 
#
# @resources:
#
#
# @notes:
#
#
# @ToDo:
#   Consider dynamic unpatchifying
#
##############################################

from utils import unpatchify_images
from dotenv import load_dotenv
import os


load_dotenv('.env')

# Globals

MASK_SAVE_TYPE = os.environ["MASK_SAVE_TYPE"]
PATCHES_DIR = "predictions\\model_v1\\masks"
IMAGES_DIR = "predictions\\model_v1\\unpatchified_masks"


if __name__ == "__main__":
    
    unpatchify_images(
        patches_dir=PATCHES_DIR,
        images_dir=IMAGES_DIR,
        save_type=MASK_SAVE_TYPE
    )
    