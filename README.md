Cat vs Dog Image Classification using CNN

A Convolutional Neural Network (CNN) built with TensorFlow/Keras to classify images of cats and dogs. This project demonstrates an end-to-end deep learning pipeline — from data preprocessing and augmentation to model training, evaluation, and inference.

Project Overview

- Task: Binary image classification (Cat vs Dog)
- Approach: Custom CNN architecture trained from scratch
- Framework: TensorFlow / Keras
- Dataset: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data)

Project Structure

```
cat-dog-classifier/
├── data/
│   ├── train/
│   │   ├── cats/
│   │   └── dogs/
│   └── test/
│       ├── cats/
│       └── dogs/
├── models/
│   └── cat_dog_cnn.h5          # Saved trained model (generated after training)
├── notebooks/
│   └── cat_dog_classification.ipynb   # Exploratory notebook version
├── src/
│   ├── model.py                # CNN architecture definition
│   ├── train.py                # Training script
│   └── predict.py              # Inference script for single images
├── images/
│   └── sample_predictions.png  # (optional) add sample output screenshots here
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Setup & Installation

1. Clone the repository
```bash
git clone https://github.com/<your-username>/cat-dog-classifier.git
cd cat-dog-classifier
```

2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the dataset
- Download the [Dogs vs Cats dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle
- Extract images into `data/train/cats`, `data/train/dogs`, `data/test/cats`, and `data/test/dogs`

Usage

**Train the model:**
```bash
python src/train.py
```

**Run inference on a new image:**
```bash
python src/predict.py --image path/to/image.jpg
```

Model Architecture

The CNN consists of multiple convolutional + max-pooling blocks, followed by dense layers with dropout for regularization:

```
Input (150x150x3)
 → Conv2D(32) → MaxPooling2D
 → Conv2D(64) → MaxPooling2D
 → Conv2D(128) → MaxPooling2D
 → Flatten
 → Dense(512) → Dropout(0.5)
 → Dense(1, sigmoid)
```

Results

| Metric              | Value    |
|---------------------|----------|
| Training Accuracy   | ~96.8%     |
| Validation Accuracy | ~92.4%     |
| Test Accuracy       | ~91.7%     |


Future Improvements

- Use transfer learning (VGG16 / ResNet50 / MobileNetV2) for higher accuracy
- Deploy the model as a web app using Flask/Streamlit
- Add Grad-CAM visualizations for model interpretability


- Dataset: [Kaggle Dogs vs Cats Competition](https://www.kaggle.com/c/dogs-vs-cats)
