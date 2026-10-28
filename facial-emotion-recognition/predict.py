"""
predict.py - Face Detection and Emotion Prediction Pipeline

Performs:
1. Model loading with caching / validation.
2. OpenCV Haar Cascade face detection.
3. Face cropping, grayscale conversion, 48x48 resizing, and [0, 1] normalization.
4. Softmax probability inference for the 7 emotion classes.
5. Robust error handling for missing models, invalid images, and no detected faces.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from model import EMOTION_LABELS, INPUT_SHAPE, NUM_CLASSES, load_config


def get_haar_cascade_path() -> str:
    """Finds the OpenCV Haar cascade XML file for frontal face detection."""
    cascade_filename = "haarcascade_frontalface_default.xml"
    if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
        path = os.path.join(cv2.data.haarcascades, cascade_filename)
        if os.path.exists(path):
            return path

    # Fallbacks in standard python site-packages or local paths
    possible_paths = [
        cascade_filename,
        os.path.join(os.path.dirname(__file__), cascade_filename),
        os.path.join(sys.prefix, "Lib", "site-packages", "cv2", "data", cascade_filename),
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return cascade_filename


def load_trained_model(
    model_path: str = "models/emotion_model.keras",
    config_path: str = "models/config.json",
) -> Tuple[Optional[Any], Optional[Dict[str, Any]], Optional[str]]:
    """
    Loads the trained Keras emotion model and configuration.
    Returns: (model, config, error_message)
    """
    if not os.path.exists(model_path):
        err = (
            f"Model file '{model_path}' not found.\n"
            "Please train the model first by running:\n"
            "  python train.py\n"
            "or verify that a valid .keras model is placed in the 'models/' directory."
        )
        return None, None, err

    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
        config = load_config(config_path) if os.path.exists(config_path) else {
            "emotion_labels": EMOTION_LABELS,
            "input_shape": list(INPUT_SHAPE),
            "num_classes": NUM_CLASSES,
        }
        return model, config, None
    except Exception as e:
        return None, None, f"Failed to load model from '{model_path}': {str(e)}"


def preprocess_face_roi(face_roi: np.ndarray) -> np.ndarray:
    """
    Converts cropped face to 48x48 grayscale and normalizes pixel values to [0, 1].
    Matches training preprocessing identically.
    """
    if face_roi.ndim == 3:
        if face_roi.shape[2] == 4:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGBA2GRAY)
        elif face_roi.shape[2] == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    elif face_roi.ndim == 2:
        pass  # Already grayscale

    # Resize to standard 48x48
    face_resized = cv2.resize(face_roi, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0]
    face_normalized = face_resized.astype(np.float32) / 255.0

    # Reshape to (1, 48, 48, 1) tensor
    input_tensor = np.expand_dims(face_normalized, axis=(0, -1))
    return input_tensor, face_resized


def detect_and_predict(
    image_input: Union[str, np.ndarray],
    model: Any,
    config: Optional[Dict[str, Any]] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_face_size: Tuple[int, int] = (30, 30),
) -> Dict[str, Any]:
    """
    Accepts an image path or numpy image array (BGR or RGB), detects faces,
    and predicts emotion probabilities for each face.
    """
    labels = config.get("emotion_labels", EMOTION_LABELS) if config else EMOTION_LABELS

    # 1. Load / validate image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            return {
                "success": False,
                "error": f"Image file '{image_input}' does not exist.",
                "num_faces": 0,
                "faces": [],
                "annotated_image": None,
            }
        img = cv2.imread(image_input)
        if img is None:
            return {
                "success": False,
                "error": f"Failed to decode image from '{image_input}'. Ensure it is a valid JPG/PNG file.",
                "num_faces": 0,
                "faces": [],
                "annotated_image": None,
            }
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        return {
            "success": False,
            "error": "Unsupported image type. Expected filepath string or numpy ndarray.",
            "num_faces": 0,
            "faces": [],
            "annotated_image": None,
        }

    if img.size == 0 or img.ndim < 2:
        return {
            "success": False,
            "error": "Image is empty or has invalid dimensions.",
            "num_faces": 0,
            "faces": [],
            "annotated_image": None,
        }

    # Ensure 3-channel BGR for drawing annotations
    if img.ndim == 2:
        gray_full = img
        annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        gray_full = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        annotated = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        annotated = img.copy()

    # 2. Face detection
    cascade_path = get_haar_cascade_path()
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        return {
            "success": False,
            "error": f"Failed to load OpenCV Haar Cascade from: {cascade_path}",
            "num_faces": 0,
            "faces": [],
            "annotated_image": annotated,
        }

    faces_detected = face_cascade.detectMultiScale(
        gray_full,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_face_size,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    detected_list = []

    # If no faces found by Haar Cascade, check if the input image is already a cropped face (e.g. 48x48)
    if len(faces_detected) == 0:
        h, w = gray_full.shape[:2]
        # If square-ish and small or close to headshot, allow full-image evaluation with note
        if 40 <= h <= 512 and 40 <= w <= 512 and (0.7 <= (w / h) <= 1.4):
            input_tensor, face_48 = preprocess_face_roi(gray_full)
            preds = model.predict(input_tensor, verbose=0)[0]
            prob_dict = {labels[i]: float(preds[i]) for i in range(len(labels))}
            dominant_idx = int(np.argmax(preds))
            dominant_emotion = labels[dominant_idx]
            confidence = float(preds[dominant_idx])

            detected_list.append(
                {
                    "face_index": 0,
                    "bbox": (0, 0, w, h),
                    "dominant_emotion": dominant_emotion,
                    "confidence": confidence,
                    "probabilities": prob_dict,
                    "face_crop": face_48,
                    "is_full_frame_fallback": True,
                }
            )

            # Annotate full frame
            cv2.rectangle(annotated, (0, 0), (w - 1, h - 1), (0, 165, 255), 2)
            label_text = f"{dominant_emotion} ({confidence * 100:.1f}%)"
            cv2.putText(
                annotated,
                label_text,
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

            return {
                "success": True,
                "num_faces": 1,
                "faces": detected_list,
                "annotated_image": annotated,
                "warning": "No face was detected by Haar cascade; processed the entire image as a cropped face.",
            }

        return {
            "success": True,
            "num_faces": 0,
            "faces": [],
            "annotated_image": annotated,
            "message": "No human faces were detected in the image. Please ensure good lighting and front-facing visibility.",
        }

    # Process all detected faces
    for i, (x, y, w, h) in enumerate(faces_detected):
        # Add slight padding if possible
        pad_y = int(0.08 * h)
        pad_x = int(0.08 * w)
        y1 = max(0, y - pad_y)
        y2 = min(gray_full.shape[0], y + h + pad_y)
        x1 = max(0, x - pad_x)
        x2 = min(gray_full.shape[1], x + w + pad_x)

        face_roi = gray_full[y1:y2, x1:x2]
        input_tensor, face_48 = preprocess_face_roi(face_roi)

        preds = model.predict(input_tensor, verbose=0)[0]
        prob_dict = {labels[k]: float(preds[k]) for k in range(len(labels))}
        dominant_idx = int(np.argmax(preds))
        dominant_emotion = labels[dominant_idx]
        confidence = float(preds[dominant_idx])

        detected_list.append(
            {
                "face_index": i,
                "bbox": (int(x), int(y), int(w), int(h)),
                "dominant_emotion": dominant_emotion,
                "confidence": confidence,
                "probabilities": prob_dict,
                "face_crop": face_48,
                "is_full_frame_fallback": False,
            }
        )

        # Draw bounding box and label
        # Emotion color palette (BGR)
        emotion_colors = {
            "Happy": (34, 187, 51),       # Vibrant Green
            "Surprise": (238, 204, 51),   # Cyan/Yellow
            "Neutral": (200, 200, 200),   # Light Gray
            "Sad": (204, 119, 34),        # Slate Blue
            "Angry": (68, 68, 238),       # Crimson Red
            "Fear": (187, 85, 153),       # Purple
            "Disgust": (68, 170, 153),    # Olive Teal
        }
        box_color = emotion_colors.get(dominant_emotion, (0, 255, 0))

        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, 2)
        label_text = f"{dominant_emotion}: {confidence * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_y = max(y - 10, text_h + 10)
        cv2.rectangle(
            annotated,
            (x, label_y - text_h - 4),
            (x + text_w + 6, label_y + baseline),
            box_color,
            -1,
        )
        cv2.putText(
            annotated,
            label_text,
            (x + 3, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0) if dominant_emotion in ["Happy", "Surprise", "Neutral"] else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return {
        "success": True,
        "num_faces": len(detected_list),
        "faces": detected_list,
        "annotated_image": annotated,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Facial Emotion from Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="models/emotion_model.keras", help="Path to .keras model")
    parser.add_argument("--config", type=str, default="models/config.json", help="Path to config.json")
    parser.add_argument("--output", type=str, default="outputs/annotated_prediction.jpg", help="Save annotated image")
    args = parser.parse_args()

    model, config, err = load_trained_model(args.model, args.config)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    result = detect_and_predict(args.image, model, config)
    if not result["success"]:
        print(f"Prediction failed: {result['error']}", file=sys.stderr)
        sys.exit(1)

    print(f"Faces detected: {result['num_faces']}")
    for face in result["faces"]:
        print(f"- Face #{face['face_index'] + 1} at bbox {face['bbox']}:")
        print(f"  Dominant Emotion: {face['dominant_emotion']} ({face['confidence'] * 100:.2f}%)")
        print("  All Probabilities:")
        for emo, prob in face["probabilities"].items():
            print(f"    {emo:10s}: {prob * 100:5.2f}%")

    if result.get("annotated_image") is not None and args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        cv2.imwrite(args.output, result["annotated_image"])
        print(f"Saved annotated image to: {args.output}")


if __name__ == "__main__":
    main()
