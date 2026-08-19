"""
train.py
Trains the CNN on the Cats vs Dogs dataset using data augmentation,
and saves the trained model to the models/ directory.
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_model

# ------------------ Config ------------------
TRAIN_DIR = "../data/train"
TEST_DIR = "../data/test"
MODEL_SAVE_PATH = "../models/cat_dog_cnn.h5"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
# ---------------------------------------------


def get_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def main():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    train_gen, val_gen, test_gen = get_data_generators()

    model = build_model(input_shape=(*IMG_SIZE, 3))
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True)
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks
    )

    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
