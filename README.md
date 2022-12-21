# U-Net Semantic Segmentation

## Description

This program is designed to perform semantic segmentation on images with multiple classes. 

A U-Net model was designed with reference to the original 2015 paper by Olaf, R. et. al. To downscale and add complexity to input image tensors, an encoding stage with CNN layers is used. This helps identify features and what classes they belong to, but loses information on pixel location. The downscaled image tensors are then upscaled in a decoding stage, in which skip connections are made to corresponding CNN layers of the encoding stage. This provides spatial information with respect to the pixel locations of classes in the image.

<image src="README_assets/UNET_structure.png" width="500">

Figure 1. U-Net Structure. (Olaf, R. et. al., 2015)

The Semantic Drone Dataset by the Graz University of Technology's Institute of Computer Graphics and Vision was used to train and test the U-Net model. To compensate for a limited GPU (Nvidia GeForce GTX 1050 ti), images of size 4000 x 6000 pixels were patched to significantly smaller sizes before training or testing. After saving test results, patches of the same image were combined to re-create a full 4000 x 6000 pixel image.

## Application

Semantic segmentation of images has a variety of existing/potential applications accross industries:
- Biomedical samples may be better analysed for the presence of various tissues and diseases.
- Autonomous vehicles may rely on live image segmentaion to identify pixels belonging to pedestrians, vehicles, and other objects. In combination with LiDAR, 3D space may be accurately classified to allow for safe transportation.

## Results

<image src="README_assets/model_v1_accuracy.png" width="500">

Figure 2. model_v1 Accuracy

<image src="README_assets/model_v1_dice_coeff.png" width="500">

Figure 3. model_v1 Dice Coefficient

<image src="README_assets/model_v1_loss.png" width="500">

Figure 4. model_v1 Loss

<image src="README_assets/476.jpg" width="500">

Figure 5. Original Image #476 (Institute of Computer Graphics and Vision, 2019)

<image src="README_assets/model_v1_476.png" width="500">

Figure 6. Semantic Segmentation of Image #476 by model_v1

## Key Learnings

Training a semantic segmantation model for large images is computationally expensive. Training using a reasonable GPU and allocation of RAM requires the patchification of large images to a smaller size. In the case of model_v1, this was 500 x 500 pixels.

As a consequence of patchification, however, pixels along the edges of patches lost the context of surrounding pixels. As a result, the accuracy of predictions for these pixels was lower than the rest of their respective patches and gridlines were seen where the patches were combined to form full 4000 x 6000 images, such as in Figure 6. 

To counteract this, a possible tecnique is to use patches of slighty larger size. The aditional pixels along patch edges may act as padding, as opposed to the blank pixels currently used for padding in the U-Net model's CNN layers. Modifications to patch size should be automatically determined based on the amount of padding required (the number of convolutional layers). 

Another padding technique may involve applying resize transformations to existing patches.

## Resources

- Institute of Computer Graphics and Vision. (2019). Semantic Drone Dataset. Graz University of Technology. Retrieved from http://dronedataset.icg.tugraz.at
- Olaf, R. et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. University of Freiburg. Retrieved from https://arxiv.org/pdf/1505.04597.pdf
- Ekin, T. (2019). Metrics to Evaluate your Semantic Segmentation Model. Towards Data Science. Retrieved from https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
