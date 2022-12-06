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

if __name__ == "__main__":

    PATCH_SIZE = (1000, 1000)
    STEP = 1000

    BASE_IMAGES_DIR = "data\\semantic_drone_dataset\\training_set\\images"
    BASE_PATCHES_DIR = "data\\semantic_drone_dataset\\patchified\\images"
    BASE_SAVE_TYPE = "jpg"

    MASK_IMAGES_DIR = "data\\semantic_drone_dataset\\training_set\\gt\\semantic\\label_images"
    MASK_PATCHES_DIR = "data\\semantic_drone_dataset\\patchified\\label_images"
    MASK_SAVE_TYPE = "png"

    patchify_images(images_dir=BASE_IMAGES_DIR, 
                    patches_dir=BASE_PATCHES_DIR, 
                    save_type=BASE_SAVE_TYPE, 
                    patch_size=PATCH_SIZE, 
                    step=STEP)
    
    patchify_images(images_dir=MASK_IMAGES_DIR, 
                    patches_dir=MASK_PATCHES_DIR, 
                    save_type=MASK_SAVE_TYPE, 
                    patch_size=PATCH_SIZE, 
                    step=STEP)
    
