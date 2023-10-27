# Multiclass Image Classification using CNN with Tensorflow

## Business Objective

Image classification helps to classify a given set of images into their respective category classes. This project aims to build an image classification model that can recognize objects in images, with applications including self-driving cars for autonomous driving.

---

## Data Description

We will be using a dataset of images categorized into three types: driving license, social security, and others. The training and testing data each contain these three subfolders. There are around 50 images in each subfolder of the testing data and approximately 200 images in each subfolder of the training data.

---

## Aim

The aim of this project is to build a sequential model using tensorflow that can perform multiclass classification on a given set of data images.

---

## Tech Stack

- Language: `Python`
- Libraries: `numpy`, `matplotlib`, `tensorflow`, `cv2`

---

## Approach

1. Importing the required libraries.
2. Load and read the data images.
3. Data Visualization:
   - Count plot.
4. Data Pre-processing:
   - Create train and validation data.
   - Normalize the data.
   - Reshape the data images.
5. Data Augmentation:
   - Using ImageDataGenerator.
6. Model Training:
   - Create a sequential model.
   - Add convolution, maxpool, dropout layers.
   - Add the softmax activation function (As this is a multiclass classification problem).
   - Pass the optimizer parameter.
   - Compile the model.
   - Fit and train the model.
   - Check for the predictions.
   - Save the model in h5 format.
7. Inferencing the model:
   - Prediction on the test data.

---

## Modular Code Overview

1. **Input**: Contains all the data for analysis, including training_data and testing_data with subfolders.
2. **Source Folder**: Contains modularized code for various project steps. This folder consists of:
   - `Engine.py`
   - `ML_Pipeline`
     - The `ML_Pipeline` folder contains functions organized into different Python files, which are called inside the `engine.py` file.
3. **Output**: Contains the fitted model trained for this data, which can be easily loaded and used for future applications without retraining.
4. **Lib**: Contains two notebooks:
   - `training_classification.ipynb`
   - `Inference.ipynb`

---
   
## Key Concepts Explored

1. How the human brain recognizes objects.
2. Basics of convolutional neural networks.
3. Input images to CNN.
4. Activation functions: Step, Sigmoid, ReLU, leaky ReLU.
5. Pooling layers: max and average pooling.
6. Flattening in CNN.
7. Multiclass and multilabel classification.
8. Sigmoid and softmax functions.
9. Loss function: binary cross-entropy.
10. Optimizers: gradient descent, stochastic gradient descent.
11. Data augmentation.
12. Model building and training.
13. Saving the model.
14. Making predictions on the test data.

---

