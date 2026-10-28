"""
test_pipeline.py - Comprehensive Unit & Integration Tests for FER System

Tests:
1. Model loading and configuration integrity.
2. Tensor shape and preprocessing correctness.
3. Prediction pipeline on face image.
4. Robust handling of missing images / invalid paths.
5. Robust handling of images with no faces.
6. Robust handling of missing model file.
7. Verification that probabilities are valid (sum to ~1.0).
"""

import os
import sys
import unittest
import numpy as np
import cv2

from model import EMOTION_LABELS, INPUT_SHAPE, NUM_CLASSES, build_emotion_model
from predict import (
    detect_and_predict,
    get_haar_cascade_path,
    load_trained_model,
    preprocess_face_roi,
)


class TestFacialEmotionPipeline(unittest.TestCase):

    def test_01_constants_and_model_architecture(self):
        """Verify model output shape, classes, and parameter compilation."""
        self.assertEqual(len(EMOTION_LABELS), 7)
        self.assertEqual(NUM_CLASSES, 7)
        self.assertEqual(INPUT_SHAPE, (48, 48, 1))

        model = build_emotion_model(input_shape=(48, 48, 1), num_classes=7)
        self.assertEqual(model.output_shape, (None, 7))
        self.assertEqual(model.input_shape, (None, 48, 48, 1))

    def test_02_haar_cascade_availability(self):
        """Ensure OpenCV Haar Cascade exists and can be instantiated."""
        cascade_path = get_haar_cascade_path()
        self.assertTrue(os.path.exists(cascade_path), f"Cascade not found at: {cascade_path}")
        cascade = cv2.CascadeClassifier(cascade_path)
        self.assertFalse(cascade.empty(), "CascadeClassifier loaded empty.")

    def test_03_preprocessing_dimensions(self):
        """Verify that preprocessing produces exactly (1, 48, 48, 1) normalized [0, 1]."""
        dummy_face = np.random.randint(0, 256, (120, 100, 3), dtype=np.uint8)
        tensor, resized = preprocess_face_roi(dummy_face)
        self.assertEqual(tensor.shape, (1, 48, 48, 1))
        self.assertEqual(resized.shape, (48, 48))
        self.assertGreaterEqual(tensor.min(), 0.0)
        self.assertLessEqual(tensor.max(), 1.0)

    def test_04_prediction_on_valid_face(self):
        """Test end-to-end detection and prediction on generated face image."""
        model, config, err = load_trained_model("models/emotion_model.keras", "models/config.json")
        self.assertIsNone(err, f"Model load error: {err}")
        self.assertIsNotNone(model)

        # Create dummy portrait with face
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (30, 30, 30)
        cv2.circle(img, (100, 100), 70, (230, 200, 170), -1)
        cv2.circle(img, (75, 80), 8, (30, 30, 30), -1)
        cv2.circle(img, (125, 80), 8, (30, 30, 30), -1)
        cv2.ellipse(img, (100, 120), (30, 20), 0, 0, 180, (30, 30, 30), 3)

        result = detect_and_predict(img, model, config)
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["num_faces"], 1)

        face = result["faces"][0]
        self.assertIn(face["dominant_emotion"], EMOTION_LABELS)
        self.assertGreaterEqual(face["confidence"], 0.0)
        self.assertLessEqual(face["confidence"], 1.0)

        # Check probabilities
        probs = face["probabilities"]
        self.assertEqual(len(probs), 7)
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

    def test_05_invalid_image_path(self):
        """Verify robust error handling when image file does not exist."""
        model, config, _ = load_trained_model("models/emotion_model.keras", "models/config.json")
        result = detect_and_predict("non_existent_image_12345.jpg", model, config)
        self.assertFalse(result["success"])
        self.assertIn("does not exist", result["error"])

    def test_06_no_face_detected(self):
        """Verify handling when an image contains no faces (e.g., pure white background)."""
        model, config, _ = load_trained_model("models/emotion_model.keras", "models/config.json")
        # 800x800 blank canvas (not a square crop)
        blank_canvas = np.ones((800, 1200, 3), dtype=np.uint8) * 255
        result = detect_and_predict(blank_canvas, model, config)
        self.assertTrue(result["success"])
        self.assertEqual(result["num_faces"], 0)
        self.assertIn("No human faces were detected", result.get("message", ""))

    def test_07_missing_model_handling(self):
        """Verify helpful error message when attempting to load a non-existent model."""
        model, config, err = load_trained_model("models/non_existent_model.keras")
        self.assertIsNone(model)
        self.assertIsNotNone(err)
        self.assertIn("not found", err.lower())

    def test_08_database_operations(self):
        """Verify SQLite database table creation, insertion, queries, and deletion."""
        import database as db
        test_db = "outputs/test_records.db"
        if os.path.exists(test_db):
            os.remove(test_db)

        # 1. Initialize DB
        db.init_db(test_db)
        self.assertTrue(os.path.exists(test_db))

        # 2. Insert record
        record_id = db.insert_prediction(
            dominant_emotion="Happy",
            confidence=0.95,
            probabilities={"Happy": 0.95, "Neutral": 0.05},
            image_name="test_face.png",
            db_path=test_db,
        )
        self.assertGreater(record_id, 0)

        # 3. Query records
        df = db.fetch_all_predictions(db_path=test_db)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["dominant_emotion"], "Happy")
        self.assertEqual(df.iloc[0]["image_name"], "test_face.png")

        # 4. Check statistics
        stats = db.fetch_db_stats(db_path=test_db)
        self.assertEqual(stats["total_records"], 1)
        self.assertEqual(stats["top_emotion"], "Happy")

        # 5. Clear records
        deleted = db.clear_all_records(db_path=test_db)
        self.assertEqual(deleted, 1)
        self.assertEqual(db.fetch_db_stats(db_path=test_db)["total_records"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

