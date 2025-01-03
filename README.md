# Face Mask Detection using Convolutional Neural Networks (CNN)

## Overview
This project implements a deep learning-based system to detect whether a person is wearing a face mask. It uses a Convolutional Neural Network (CNN) trained on a labeled dataset of images with and without masks.

## Steps Involved

### 1. **Setting Up Kaggle API**
- Install the Kaggle package to access datasets.
- Configure the Kaggle API by uploading the `kaggle.json` file, which contains the API token.

### 2. **Downloading and Extracting the Dataset**
- Use the Kaggle API to download the face mask dataset.
- Extract the contents of the dataset, which include images of people with and without masks.

### 3. **Data Preparation**
- Read and organize image files into two categories:
  - `with_mask`
  - `without_mask`
- Assign labels:
  - `1` for images with masks.
  - `0` for images without masks.
- Combine all images into a single dataset and their respective labels into a label array.
- Resize images to `(128, 128)` and convert them to numpy arrays for processing.

### 4. **Data Splitting and Scaling**
- Split the dataset into training and testing sets (80-20 split).
- Normalize pixel values by dividing by 255 to scale the image data to the range `[0, 1]`.

### 5. **Building the CNN Model**
- The CNN consists of:
  - Convolutional layers with ReLU activation.
  - MaxPooling layers to reduce spatial dimensions.
  - Fully connected layers (Dense layers) with Dropout for regularization.
  - Final output layer with a sigmoid activation function to predict two classes: with mask (1) and without mask (0).

### 6. **Model Training**
- Train the CNN on the training dataset using `sparse_categorical_crossentropy` as the loss function and the Adam optimizer.
- Validate the model using a 10% split of the training dataset.

### 7. **Model Evaluation**
- Evaluate the trained model on the test dataset and print the accuracy.
- Plot the training and validation loss, as well as accuracy over epochs.

### 8. **Predictive System**
- A simple predictive system allows the user to input the path to an image and determine whether the person in the image is wearing a mask.
- The system preprocesses the input image, passes it through the trained model, and prints the prediction.

## Requirements
- Python 3.7 or above
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- OpenCV
- PIL
- Kaggle API credentials (`kaggle.json`)

## Dataset
The dataset is downloaded from Kaggle and contains labeled images of people with and without masks.

## Results
The trained model achieves good accuracy on the test dataset and can be used to predict mask usage in unseen images.
