"""
predict.py
Loads the trained CNN model and predicts whether a given image
is a Cat or a Dog.

Usage:
    python predict.py --image path/to/image.jpg
"""

import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

MODEL_PATH = "../models/cat_dog_cnn.h5"
IMG_SIZE = (150, 150)


def predict_image(img_path, model_path=MODEL_PATH):
    model = load_model(model_path)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = "Dog 🐶" if prediction > 0.5 else "Cat 🐱"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    print(f"Prediction: {label} (confidence: {confidence * 100:.2f}%)")
    return label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cat vs Dog from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to the trained model")
    args = parser.parse_args()

    predict_image(args.image, args.model)
