# Image-Classification-CNN
A deep learning project that classifies images into Cat and Dog categories using a Convolutional Neural Network (CNN) built with Python and TensorFlow/Keras.

Project Overview

This project demonstrates how Convolutional Neural Networks (CNNs) can be used for image classification. The model is trained on a dataset containing cat and dog images and learns to automatically identify the class of a new input image.

Features

- Image preprocessing and resizing
- CNN-based image classification
- Training and validation of the deep learning model
- Model performance evaluation
- Prediction on new images
- Accuracy and loss visualization

Project Structure

Image-Classification-CNN/
│
├── dataset/
│   ├── cats/
│   └── dogs/
│
├── notebooks/
│   └── image_classification_cnn.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
│   └── cnn_model.h5
│
├── images/
│   └── sample_predictions.png
│
├── requirements.txt
├── README.md
└── .gitignore

Model Evaluation

The model is evaluated using:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
The training and validation curves are plotted using Matplotlib to analyze model performance and identify potential overfitting.

Prediction

The trained CNN model can be used to classify a new image as either:

Cat 🐱
Dog 🐶

Example workflow:

image = load_image("sample.jpg")
prediction = model.predict(image)

if prediction > 0.5:
    print("Dog")
else:
    print("Cat")

Results

The CNN successfully learns visual features from the training images and performs binary classification between cats and dogs.
Training Accuracy:94.8
Validation Accuracy:92.3
Validation Loss:0.21




---

⭐ If you found this project useful, consider giving the repository a star!
