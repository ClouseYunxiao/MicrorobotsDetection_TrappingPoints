# MicrorobotsDetection_TrappingPoints

This repository is the official implementation of "Machine Learning-Based Real-Time Localisation and Automatic Trapping of Multiple Microrobots in Optical Tweezer"(MARSS 2022).

The key contributions of this paper are threefold:
*  A deep neural network model is constructed based on a modified YOLOv4-tiny architecture for multiple microrobots localisation, where a ConvNext-based block is used as the backbone for feature extraction and Gaussian modeling is used to optimise the output of the detection head. Self-supervised learning with Mosaic data augmentation is used for model training.
*  A machine learning-based ellipse detection technique is proposed to identify the optimal trapping points for each localised microrobot, through which the positions of the laser spots can be determine to manipulate the microrobots stably. 

The implementation is as follows.

## 1. Requirements
* To create a new conda environment:
```
conda create -f environment.yaml
```
A new conda environment named as "microrobots" will be established.

## 2. Data Pre-processing and Data Augmentation
We adopt edge detection as data preprocessing and Mosaic data augmentation in our work. The data set is self-supervised constructed, and Mosaic data augmentation is used for reduce the volume of the dataset as the data collection of microrobots is tedious and verify the adaptation of the proposed netwrok. 
* Data pre-processing:
```
python data_preprocessing.py
```

* Mosaic data augmentation:
```
python mosaic.py
```

## 3. Training
* To train the model(s) in the paper, run this command:
```
python train.py
```

## 4. Evaluation
* To evaluate the results, run this command:
```
python evaluate.py
```

## 5. Results
* Our model achieves the following performance:

|       Methods       |    Precesion(%)  |     Recall(%)    |        FPS       |
| :-----------------: | :--------------: | :--------------: | :--------------: |
|      train set 1    |       98.6       |      99.81       |       36.02      |
|      train set 2    |       97.3       |      99.931      |       36.95      |

*
