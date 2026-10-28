"""
train.py - Dataset Preparation and Model Training Pipeline for FER-2013

Supports:
- FER-2013 CSV format (emotion, pixels, Usage)
- Directory-based image dataset (train/, val/ or test/ subfolders with emotion names)
- Data preprocessing: grayscale 48x48, normalization to [0, 1]
- Training with ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau
- Real metric recording (history JSON, loss and accuracy plots)
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import (
    EMOTION_LABELS,
    INPUT_SHAPE,
    LABEL_MAP,
    NUM_CLASSES,
    build_emotion_model,
    save_config,
)


def print_dataset_instructions():
    """Prints clear instructions on how to obtain the FER-2013 dataset."""
    print("=" * 70)
    print("FER-2013 DATASET INSTRUCTIONS")
    print("=" * 70)
    print("The FER-2013 dataset was not found in the 'data/' folder.")
    print("To obtain the dataset, you have two convenient options:")
    print()
    print("Option 1: Download from Kaggle (CSV format)")
    print("  1. Visit: https://www.kaggle.com/datasets/deadskull7/fer2013")
    print("     or run: kaggle datasets download -d deadskull7/fer2013")
    print("  2. Extract 'fer2013.csv' and place it into: data/fer2013.csv")
    print()
    print("Option 2: Download from Kaggle (Image folder format)")
    print("  1. Visit: https://www.kaggle.com/datasets/msambare/fer2013")
    print("     or run: kaggle datasets download -d msambare/fer2013")
    print("  2. Extract so that the folder structure is:")
    print("       data/train/<angry, disgust, fear, happy, neutral, sad, surprise>")
    print("       data/test/<angry, disgust, fear, happy, neutral, sad, surprise>")
    print("=" * 70)


def load_fer2013_csv(
    csv_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses FER-2013 from a CSV file."""
    print(f"Loading FER-2013 dataset from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    has_usage = "Usage" in df.columns

    if has_usage:
        train_df = df[df["Usage"] == "Training"]
        val_df = df[df["Usage"] == "PublicTest"]
        test_df = df[df["Usage"] == "PrivateTest"]
    else:
        # Default split 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["emotion"])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["emotion"])

    def parse_split(split_df):
        pixels = split_df["pixels"].tolist()
        width, height = 48, 48
        images = []
        for p in pixels:
            img = np.fromstring(p, dtype=np.float32, sep=" ").reshape((height, width, 1))
            images.append(img / 255.0)
        X = np.array(images, dtype=np.float32)

        import tensorflow as tf
        y_int = split_df["emotion"].values
        y = tf.keras.utils.to_categorical(y_int, num_classes=NUM_CLASSES)
        return X, y

    X_train, y_train = parse_split(train_df)
    X_val, y_val = parse_split(val_df)
    X_test, y_test = parse_split(test_df)

    print(f"Loaded CSV successfully: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_fer2013_directory(
    base_dir: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads and preprocesses FER-2013 from image folders."""
    import cv2
    import tensorflow as tf

    print(f"Loading images from directory: {base_dir}")
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    val_dir = os.path.join(base_dir, "val")

    def read_folder(folder_path):
        images, labels = [], []
        if not os.path.exists(folder_path):
            return np.empty((0, 48, 48, 1), dtype=np.float32), np.empty((0, NUM_CLASSES), dtype=np.float32)

        for category_name in os.listdir(folder_path):
            cat_path = os.path.join(folder_path, category_name)
            if not os.path.isdir(cat_path):
                continue
            cat_norm = category_name.capitalize()
            if cat_norm not in LABEL_MAP:
                # Try lowercase match
                match = [k for k in LABEL_MAP if k.lower() == category_name.lower()]
                if match:
                    cat_norm = match[0]
                else:
                    continue

            class_idx = LABEL_MAP[cat_norm]
            for fname in os.listdir(cat_path):
                fpath = os.path.join(cat_path, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (48, 48))
                img = img.astype(np.float32) / 255.0
                images.append(img.reshape(48, 48, 1))
                labels.append(class_idx)

        X = np.array(images, dtype=np.float32)
        y = tf.keras.utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)
        return X, y

    if os.path.exists(train_dir):
        X_train, y_train = read_folder(train_dir)
        if os.path.exists(val_dir):
            X_val, y_val = read_folder(val_dir)
            X_test, y_test = read_folder(test_dir) if os.path.exists(test_dir) else (X_val, y_val)
        elif os.path.exists(test_dir):
            X_all_test, y_all_test = read_folder(test_dir)
            if len(X_all_test) > 1:
                X_val, X_test, y_val, y_test = train_test_split(
                    X_all_test, y_all_test, test_size=0.5, random_state=42
                )
            else:
                X_val, y_val = X_all_test, y_all_test
                X_test, y_test = X_all_test, y_all_test
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    else:
        # Check if emotions directly in base_dir
        X_all, y_all = read_folder(base_dir)
        if len(X_all) == 0:
            raise FileNotFoundError(f"No emotion image classes found inside {base_dir}")
        X_train, X_temp, y_train, y_temp = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test


def find_dataset(data_dir: str = "data") -> Optional[str]:
    """Scans the data directory for CSV or folder datasets."""
    if not os.path.exists(data_dir):
        return None

    csv_candidates = [
        os.path.join(data_dir, "fer2013.csv"),
        os.path.join(data_dir, "fer2013", "fer2013.csv"),
        os.path.join(data_dir, "FER2013.csv"),
    ]
    for c in csv_candidates:
        if os.path.isfile(c):
            return c

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv") and "fer" in f.lower():
                return os.path.join(root, f)

    train_folder = os.path.join(data_dir, "train")
    if os.path.isdir(train_folder):
        return data_dir

    return None


def generate_synthetic_samples(
    num_samples_per_class: int = 25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic facial feature patterns for smoke testing and offline verification."""
    import tensorflow as tf

    print(f"Generating synthetic verification data ({num_samples_per_class} per class)...")
    np.random.seed(42)
    images = []
    labels = []

    for class_idx in range(NUM_CLASSES):
        for _ in range(num_samples_per_class):
            img = np.zeros((48, 48), dtype=np.float32)
            # Oval face outline
            y_indices, x_indices = np.ogrid[:48, :48]
            mask = ((x_indices - 24) ** 2) / (18 ** 2) + ((y_indices - 24) ** 2) / (22 ** 2) <= 1.0
            img[mask] = 0.5 + np.random.uniform(-0.1, 0.1, size=img[mask].shape)
            # Eyes
            img[16:20, 14:18] = 0.1
            img[16:20, 30:34] = 0.1
            # Class-specific mouth signature
            if class_idx == 3:  # Happy: upward curve
                img[32:35, 18:30] = 0.9
            elif class_idx == 4:  # Sad: downward curve
                img[35:38, 18:30] = 0.2
            elif class_idx == 5:  # Surprise: open mouth
                img[30:36, 21:27] = 0.05
            else:
                img[34:36, 18:30] = 0.4
            noise = np.random.normal(0, 0.03, (48, 48))
            img = np.clip(img + noise, 0.0, 1.0)
            images.append(img.reshape(48, 48, 1))
            labels.append(class_idx)

    X = np.array(images, dtype=np.float32)
    y = tf.keras.utils.to_categorical(np.array(labels), num_classes=NUM_CLASSES)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    return X_train, y_train, X_val, y_val, X_test, y_test


def plot_and_save_history(history_dict: dict, output_dir: str = "outputs"):
    """Plots and saves actual training and validation accuracy and loss curves."""
    os.makedirs(output_dir, exist_ok=True)
    history_json_path = os.path.join(output_dir, "training_history.json")
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history_dict, f, indent=4)

    epochs = range(1, len(history_dict.get("accuracy", [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(epochs, history_dict.get("accuracy", []), "b-o", label="Training Accuracy", linewidth=2)
    if "val_accuracy" in history_dict:
        axes[0].plot(epochs, history_dict["val_accuracy"], "g--s", label="Validation Accuracy", linewidth=2)
    axes[0].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Loss Plot
    axes[1].plot(epochs, history_dict.get("loss", []), "r-o", label="Training Loss", linewidth=2)
    if "val_loss" in history_dict:
        axes[1].plot(epochs, history_dict["val_loss"], "m--s", label="Validation Loss", linewidth=2)
    axes[1].set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved real training curves to: {plot_path}")


def train(
    data_path: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 64,
    model_save_path: str = "models/emotion_model.keras",
    config_save_path: str = "models/config.json",
    output_dir: str = "outputs",
    use_synthetic: bool = False,
):
    """Executes model training with early stopping, model checkpointing, and real metric saving."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    os.makedirs(os.path.dirname(os.path.abspath(model_save_path)), exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    if use_synthetic:
        X_train, y_train, X_val, y_val, X_test, y_test = generate_synthetic_samples(num_samples_per_class=40)
    else:
        dataset_target = data_path or find_dataset("data")
        if dataset_target is None:
            print_dataset_instructions()
            print("\nDataset not currently available. Run with --synthetic to verify pipeline without dataset.")
            return False

        if os.path.isfile(dataset_target) and dataset_target.endswith(".csv"):
            X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013_csv(dataset_target)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = load_fer2013_directory(dataset_target)

    # Save test set for evaluate.py
    np.savez_compressed(
        os.path.join(output_dir, "test_data.npz"),
        X_test=X_test,
        y_test=y_test,
    )
    print(f"Saved held-out test data ({X_test.shape[0]} samples) to {output_dir}/test_data.npz")

    # Build model
    model = build_emotion_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("\nStarting CNN training...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # If ModelCheckpoint did not save because validation didn't trigger better score, save final model
    if not os.path.exists(model_save_path):
        model.save(model_save_path)

    # Save configuration
    save_config(
        config_save_path,
        extra_info={
            "epochs_trained": len(history.history["loss"]),
            "final_train_accuracy": float(history.history["accuracy"][-1]),
            "final_val_accuracy": float(history.history.get("val_accuracy", [0])[-1]),
            "batch_size": batch_size,
        },
    )
    print(f"Model saved to: {model_save_path}")
    print(f"Configuration saved to: {config_save_path}")

    # Plot actual history
    plot_and_save_history(history.history, output_dir=output_dir)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial Emotion Recognition CNN")
    parser.add_argument("--data", type=str, default=None, help="Path to fer2013.csv or data folder")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic samples for smoke verification")
    args = parser.parse_args()

    train(
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic=args.synthetic,
    )
