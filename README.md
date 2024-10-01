# Eyes On The Ground Challenge

# Context
This project contains the code used to submit for the Eyes on the Ground competition on Zindi.

In this competition, the objective was to predict the extent of damage from crop images. The damage can be caused by drought, flooding, and the stage of growth can vary between the images.


# About this repo
This repository contains the model/dataset classes, training, inference, and submission generation codes. The experiments and training itself were conducted in a 2*T4 Kaggle notebook, with experiment tracking and model versioning in W&B.


<!-- 

# Approach
### Hardware and tools
I used free 2*T4 kaggle notebooks with ~28Gb of vRAM and W&B for experiment tracking, github for code versioning, and created a private Kaggle dataset for the data.

I used Albumentations for image augmentations, Pytorch and timm pretrained models.


### Augmentations
#### Training 
```
TRAIN_TFS = A.Compose([
    A.Transpose(),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(transpose_mask=True)
])
```
#### Inference 
```
VAL_TFS = A.Compose([
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(transpose_mask=True)
 ])
```
### Model
Using pretrained models from timm:

Our model class is a base model followed by a many to one linear layer then a sigmoid 

Using the same architecture I trained a classification model to filter images wih approximately no damage, since sigmoid makes it difficult for the model to output the value 0 (due to vanishing gradients).

Regression model :
Classification model  : -->
