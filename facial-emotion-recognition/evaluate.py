"""
evaluate.py - Quantitative Evaluation and Metrics Generation for FER CNN

Calculates:
- Real test accuracy, precision, recall, and F1-score (macro and weighted)
- Per-class metrics classification report
- Confusion matrix heatmap and CSV
- Training vs. Validation curves visualization
- Stores all evaluation artifacts strictly based on real model evaluation in `outputs/`
"""

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model import EMOTION_LABELS, NUM_CLASSES, load_config
from train import find_dataset, load_fer2013_csv, load_fer2013_directory


def load_test_dataset(test_data_path: str, data_dir: str = "data"):
    """
    Loads test data either from pre-saved npz file or directly from data folder/csv.
    """
    if os.path.exists(test_data_path):
        print(f"Loading test set from pre-saved file: {test_data_path}")
        data = np.load(test_data_path)
        return data["X_test"], data["y_test"]

    dataset_target = find_dataset(data_dir)
    if dataset_target is not None:
        print(f"Loading test split directly from dataset: {dataset_target}")
        if os.path.isfile(dataset_target) and dataset_target.endswith(".csv"):
            _, _, _, _, X_test, y_test = load_fer2013_csv(dataset_target)
            return X_test, y_test
        else:
            _, _, _, _, X_test, y_test = load_fer2013_directory(dataset_target)
            return X_test, y_test

    return None, None


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    output_path: str = "outputs/confusion_matrix.png",
    normalize: bool = True,
):
    """Generates and saves a high-resolution confusion matrix heatmap."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
        fmt = ".2%"
        data_to_plot = cm_norm
    else:
        fmt = "d"
        data_to_plot = cm

    sns.heatmap(
        data_to_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgray",
        annot_kws={"size": 10},
    )
    ax.set_title("FER-2013 Confusion Matrix", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Emotion", fontsize=12, labelpad=10)
    ax.set_ylabel("True Emotion", fontsize=12, labelpad=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix heatmap to: {output_path}")


def evaluate(
    model_path: str = "models/emotion_model.keras",
    config_path: str = "models/config.json",
    test_data_path: str = "outputs/test_data.npz",
    output_dir: str = "outputs",
):
    """
    Evaluates the model on test data and records real metrics.
    No fabricated or assumed numbers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Verify model existence
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found.", file=sys.stderr)
        print("Please train the model first by running: python train.py", file=sys.stderr)
        return False

    # 2. Load model
    print(f"Loading trained model: {model_path}")
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return False

    config = load_config(config_path)
    labels = config.get("emotion_labels", EMOTION_LABELS)

    # 3. Load test data
    X_test, y_test = load_test_dataset(test_data_path)
    if X_test is None or len(X_test) == 0:
        print(
            "Error: No test data found. Provide 'outputs/test_data.npz' or ensure the dataset exists in 'data/'.",
            file=sys.stderr,
        )
        return False

    print(f"Evaluating model on {len(X_test)} test samples...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.flatten().astype(int)

    # 4. Metrics computation
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    report_dict = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=labels, zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    # 5. Save artifacts to outputs/
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    np.savetxt(cm_csv_path, cm, delimiter=",", fmt="%d", header=",".join(labels), comments="")

    cm_img_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, labels, output_path=cm_img_path, normalize=True)

    results_payload = {
        "status": "completed",
        "num_test_samples": int(len(X_test)),
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_dict,
        "emotion_labels": labels,
    }

    eval_json_path = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=4)

    # 6. Print summary to terminal
    print("\n" + "=" * 65)
    print("REAL MODEL EVALUATION RESULTS (HELD-OUT TEST SET)")
    print("=" * 65)
    print(f"Total Test Samples:   {len(X_test)}")
    print(f"Overall Accuracy:     {acc * 100:.2f}%")
    print(f"Macro Precision:      {macro_p * 100:.2f}%")
    print(f"Macro Recall:         {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:       {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score:    {weighted_f1 * 100:.2f}%")
    print("-" * 65)
    print("Classification Report:")
    print(report_text)
    print("=" * 65)
    print(f"Saved evaluation metrics to:  {eval_json_path}")
    print(f"Saved confusion matrix CSV to: {cm_csv_path}")
    print(f"Saved confusion matrix plot:   {cm_img_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Facial Emotion CNN on Test Data")
    parser.add_argument("--model", type=str, default="models/emotion_model.keras", help="Path to model")
    parser.add_argument("--config", type=str, default="models/config.json", help="Path to config")
    parser.add_argument("--test-data", type=str, default="outputs/test_data.npz", help="Path to test_data.npz")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        config_path=args.config,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
    )
