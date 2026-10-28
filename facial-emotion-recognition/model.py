"""
model.py - CNN Architecture for Facial Emotion Recognition (FER)

This module defines the Convolutional Neural Network (CNN) architecture
tailored for 48x48 grayscale facial emotion classification into 7 emotions:
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
"""

import json
import os
from typing import List, Tuple, Optional

# Constants
EMOTION_LABELS: List[str] = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]

EMOTION_MAP = {i: label for i, label in enumerate(EMOTION_LABELS)}
LABEL_MAP = {label: i for i, label in enumerate(EMOTION_LABELS)}
INPUT_SHAPE: Tuple[int, int, int] = (48, 48, 1)
NUM_CLASSES: int = len(EMOTION_LABELS)


def build_emotion_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 0.001,
):
    """
    Constructs and compiles a deep CNN for facial emotion recognition.

    Architecture highlights:
    - 4 Convolutional blocks with (3x3) filters, ReLU activations,
      Batch Normalization, Max Pooling (2x2), and Dropout.
    - Fully connected dense layers with Batch Normalization and Dropout (0.5)
      for regularization against overfitting.
    - Softmax output layer for 7-class probability distribution.
    - Adam optimizer with categorical cross-entropy loss.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, regularizers

    model = models.Sequential(name="Facial_Emotion_CNN")

    # Block 1
    model.add(layers.Input(shape=input_shape))
    model.add(
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv1_1",
        )
    )
    model.add(layers.BatchNormalization(name="bn1_1"))
    model.add(layers.Activation("relu", name="relu1_1"))
    model.add(
        layers.Conv2D(
            64,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv1_2",
        )
    )
    model.add(layers.BatchNormalization(name="bn1_2"))
    model.add(layers.Activation("relu", name="relu1_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool1"))
    model.add(layers.Dropout(0.25, name="dropout1"))

    # Block 2
    model.add(
        layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv2_1",
        )
    )
    model.add(layers.BatchNormalization(name="bn2_1"))
    model.add(layers.Activation("relu", name="relu2_1"))
    model.add(
        layers.Conv2D(
            128,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv2_2",
        )
    )
    model.add(layers.BatchNormalization(name="bn2_2"))
    model.add(layers.Activation("relu", name="relu2_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool2"))
    model.add(layers.Dropout(0.25, name="dropout2"))

    # Block 3
    model.add(
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv3_1",
        )
    )
    model.add(layers.BatchNormalization(name="bn3_1"))
    model.add(layers.Activation("relu", name="relu3_1"))
    model.add(
        layers.Conv2D(
            256,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            name="conv3_2",
        )
    )
    model.add(layers.BatchNormalization(name="bn3_2"))
    model.add(layers.Activation("relu", name="relu3_2"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), name="pool3"))
    model.add(layers.Dropout(0.25, name="dropout3"))

    # Classifier Block
    model.add(layers.Flatten(name="flatten"))
    model.add(
        layers.Dense(
            512,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(0.001),
            name="dense1",
        )
    )
    model.add(layers.BatchNormalization(name="bn_dense1"))
    model.add(layers.Activation("relu", name="relu_dense1"))
    model.add(layers.Dropout(0.5, name="dropout_dense1"))

    model.add(
        layers.Dense(
            256,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizers.l2(0.001),
            name="dense2",
        )
    )
    model.add(layers.BatchNormalization(name="bn_dense2"))
    model.add(layers.Activation("relu", name="relu_dense2"))
    model.add(layers.Dropout(0.5, name="dropout_dense2"))

    # 7-class softmax output
    model.add(layers.Dense(num_classes, activation="softmax", name="output_probabilities"))

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def save_config(filepath: str, extra_info: Optional[dict] = None) -> None:
    """Save model configuration and emotion class mapping to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    config = {
        "emotion_labels": EMOTION_LABELS,
        "emotion_map": EMOTION_MAP,
        "input_shape": list(INPUT_SHAPE),
        "num_classes": NUM_CLASSES,
    }
    if extra_info:
        config.update(extra_info)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def load_config(filepath: str) -> dict:
    """Load model configuration from JSON."""
    if not os.path.exists(filepath):
        return {
            "emotion_labels": EMOTION_LABELS,
            "emotion_map": EMOTION_MAP,
            "input_shape": list(INPUT_SHAPE),
            "num_classes": NUM_CLASSES,
        }
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    m = build_emotion_model()
    m.summary()
