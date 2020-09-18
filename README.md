# ImageSemanticSegmentation

Notebook to import any dataset with a real image and a mask with all the pixel identified. Moreover, you can select different parametres easily with the form in the notebook.

# Table of contents

* [Introduction](#introduction)
* [Methodology](#methodology)
* [Dataset](#dataset)


# Introduction
>[Table of contents](#table-of-contents)

In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze.[1][2] Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images. More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

### An Image segmentation example


[Introducir una imagen de ejemplo]

# Methodology
>[Table of contents](#table-of-contents)

The dataset is chosen, the dataset object is created (augmentation can be used here), Some model is chosen and also the loss function, metrics, callbacks and optimizer.
With this you can train your dataset with the chosen model or download a pre-trained if you prefer and predict directly. 
To finish you can use the Tensorboard to check the evolution of the model.

# Dataset
>[Table of contents](#table-of-contents)

For the time being you can use two dataset
* oxford_iiit_pet
* cityscapes

### Oxford_iiit_pet

 You can get more information here https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet
 
 The important for us is that you have a real image and a mask with 3 label. 
  Class 1 : Pixel belonging to the pet.
  Class 2 : Pixel bordering the pet.
  Class 3 : None of the above/ Surrounding pixel.

  For the sake of convenience, in preprocesing  let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}.

### Cityscapes

'''working in readme'''

