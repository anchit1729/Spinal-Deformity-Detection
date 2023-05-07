# FYP21019: Spinal Deformity Progression Prediction - Codebase

## Author: Anchit Mishra

Note: All code in this directory has been written by me (Anchit Mishra). My partner for the FYP, Xinda Chen, worked independently on his own implementation of pedicle segmentation and reconstruction.

This codebase contains two separate folders:
1. Segmentation-Anchit - All code pertaining to the deep learning model developed in this project. It provides visualisation, training and inference code for testing purposes, along with a pretrained model in `models/unet_model_v11.pt` and corresponding predictions on 1080 images in the `predictions` folder. 

2. Reconstruction-Anchit - All code pertaining to the reconstruction script developed in this project. The file `reconstruct.py` contains the actual programming logic used to perform spinal reconstruction. Three instances are also provided as an example to visualise the results obtained from the algorithm; these can be found in the `images` and `annotations` directories.

NOTE: Due to privacy concerns, the dataset is not publicly accessible (ownership lies with the University of Hong Kong and Queen Mary Hospital, Hong Kong).

- `human_lumbar_vertebra` - The directory containing assets to render lumbar vertebra models
- `human_thoracic_vertebra` - The directory containing assets to render thoracic vertebra models
- `cleaned_data` - The directory containing a training dataset of 1080 X-ray images and corresponding ground truth labels

Further, the `Segmentation-Anchit` directory has been split into two parts - `Segmentation-1-Anchit` and `Segmentation-2-Anchit`. After downloading, the contents of these directories should be copied into a single `Segmentation-Anchit` folder before use. The `human_lumbar_vertebra` and `human_thoracic_vertebra` folders should be pasted inside the `Reconstruction-Anchit` folder, and the `cleaned_data` folder should be pasted inside the `Segmentation-Anchit` folder. Once this setup is complete, the code may executed normally.

## Segmentation

The code used for implementing semantic segmentation of pedicles is in the Jupyter notebook format. Most importantly, the `train.ipynb` notebook provides the code used for training the segmentation model, and the `inference.ipynb` notebook provides the code used for testing the model and generating predictions. Notably, libraries used for this stage of the project include PyTorch, Albumentations, Segmentation Models PyTorch, MatPlotLib and SKimage.

## Reconstruction

The code used for implementing spine reconstruction is in the form of a regular Python script. The `reconstruct.py` file provides the code used for performing reconstruction, and takes the following command line arguments:

```python reconstruct.py [width] [height] [case_number]```

Here, the `width` and `height` are the width and height of the input images (in pixels) respectively, and the `case_number` is the name of the case on which reconstruction is to be performed. For example, the cases provided in the codebase are case 00039, 01146 and 01230. Keep in mind that all images must have the same width and height, and in some edge cases, the mask generated by the segmentation model may need to be manually edited by a human agent before it can accurately be used by the script for reconstruction.
