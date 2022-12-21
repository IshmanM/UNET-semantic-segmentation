# Semantic Segmentation

## Description

This program is designed to perform semantic segmentation on images with multiple classes. 

A U-Net model was designed with reference to the original 2015 paper by Olaf, R. et. al. To downscale and add complexity to input image tensors, an encoding stage with CNN layers is used. This helps identify features and what classes they belong to, but loses information on pixel location. The downscaled image tensors are then upscaled in a decoding stage, in which skip connections are made to corresponding CNN layers of the encoding stage. This provides spatial information with respect to the pixel locations of classes in the image.

![Alt text](README_assets\UNET_structure.png)
Figure 1. U-Net Structure. (Olaf, R. et. al., 2015)

The Semantic Drone Dataset by the Graz University of Technology's Institute of Computer Graphics and Vision was used to train and test the U-Net model. To compensate for a limited GPU (Nvidia GeForce GTX 1050 ti), images of size 4000 x 6000 pixels were patched to significantly smaller sizes before training or testing. After saving test results, patches of the same image were combined to re-create a full 4000 x 6000 pixel image.

## Application

Semantic segmentation of images has a variety of existing/potential applications accross industries:
- Biomedical samples may be better analysed for the presence of various tissues and diseases.
- Autonomous vehicles may rely on live image segmentaion to identify pixels belonging to pedestrians, vehicles, and other objects. In combination with LiDAR, 3D space may be accurately classified to allow for safe transportation.

## Results

![Alt text](README_assets\model_v1_accuracy.png)
Figure 2. model_v1 Accuracy

![Alt text](README_assets\model_v1_dice_coeff.png)
Figure 3. model_v1 Dice Coefficient

![Alt text](README_assets\model_v1_loss.png)
Figure 4. model_v1 Loss

![Alt text](README_assets\476.jpg)
Figure 5. Original Image #476 (Institute of Computer Graphics and Vision, 2019)

![Alt text](README_assets\model_v1_476.png)
Figure 6. Semantic Segmentation of Image #476 by model_v1

## Key Learnings

computationally expensive

but patchifying loses image location data


## Resources

- Institute of Computer Graphics and Vision. (2019). Semantic Drone Dataset. Graz University of Technology. Retrieved from http://dronedataset.icg.tugraz.at
- Olaf, R. et. al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. University of Freiburg. Retrieved from https://arxiv.org/pdf/1505.04597.pdf
- Ekin, T. (2019). Metrics to Evaluate your Semantic Segmentation Model. Towards Data Science. Retrieved from https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2