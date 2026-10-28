"""
database.py - SQLite Database Management for Facial Emotion Recognition Logs

Handles schema creation, persistent logging of facial detections and predictions,
analytical querying, filtering, and export capabilities.
"""

from datetime import datetime
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "emotion_records.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a database connection and ensures directory exists."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the SQLite database tables and indexes."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_name TEXT DEFAULT 'unnamed_image',
                face_index INTEGER DEFAULT 0,
                dominant_emotion TEXT NOT NULL,
                confidence REAL NOT NULL,
                angry_prob REAL DEFAULT 0.0,
                disgust_prob REAL DEFAULT 0.0,
                fear_prob REAL DEFAULT 0.0,
                happy_prob REAL DEFAULT 0.0,
                sad_prob REAL DEFAULT 0.0,
                surprise_prob REAL DEFAULT 0.0,
                neutral_prob REAL DEFAULT 0.0,
                bbox TEXT DEFAULT '',
                source TEXT DEFAULT 'Upload',
                notes TEXT DEFAULT ''
            );
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_emotion ON prediction_logs(dominant_emotion);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs(timestamp);")
        conn.commit()


def insert_prediction(
    dominant_emotion: str,
    confidence: float,
    probabilities: Dict[str, float],
    image_name: str = "uploaded_image",
    face_index: int = 0,
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0),
    source: str = "Upload",
    notes: str = "",
    timestamp: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Inserts a single face prediction entry into the database."""
    init_db(db_path)
    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}" if bbox else ""

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO prediction_logs (
                timestamp, image_name, face_index, dominant_emotion, confidence,
                angry_prob, disgust_prob, fear_prob, happy_prob, sad_prob,
                surprise_prob, neutral_prob, bbox, source, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ts,
                image_name,
                face_index,
                dominant_emotion,
                round(float(confidence), 4),
                round(float(probabilities.get("Angry", 0.0)), 4),
                round(float(probabilities.get("Disgust", 0.0)), 4),
                round(float(probabilities.get("Fear", 0.0)), 4),
                round(float(probabilities.get("Happy", 0.0)), 4),
                round(float(probabilities.get("Sad", 0.0)), 4),
                round(float(probabilities.get("Surprise", 0.0)), 4),
                round(float(probabilities.get("Neutral", 0.0)), 4),
                bbox_str,
                source,
                notes,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def log_multi_faces(
    faces: List[Dict[str, Any]],
    image_name: str = "uploaded_image",
    source: str = "Upload",
    db_path: str = DEFAULT_DB_PATH,
) -> List[int]:
    """Logs all faces detected in a single analysis run."""
    init_db(db_path)
    inserted_ids = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for face in faces:
        record_id = insert_prediction(
            dominant_emotion=face["dominant_emotion"],
            confidence=face["confidence"],
            probabilities=face["probabilities"],
            image_name=image_name,
            face_index=face.get("face_index", 0),
            bbox=face.get("bbox", (0, 0, 0, 0)),
            source=source,
            timestamp=ts,
            db_path=db_path,
        )
        inserted_ids.append(record_id)
    return inserted_ids


def fetch_all_predictions(
    limit: int = 200,
    emotion_filter: Optional[str] = None,
    source_filter: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Retrieves prediction records as a pandas DataFrame."""
    init_db(db_path)
    query = "SELECT * FROM prediction_logs WHERE 1=1"
    params = []

    if emotion_filter and emotion_filter != "All":
        query += " AND dominant_emotion = ?"
        params.append(emotion_filter)

    if source_filter and source_filter != "All":
        query += " AND source = ?"
        params.append(source_filter)

    query += " ORDER BY timestamp DESC, id DESC LIMIT ?"
    params.append(limit)

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def fetch_db_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Calculates summary statistics across stored prediction records."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM prediction_logs;")
        total_records = cursor.fetchone()[0]

        if total_records == 0:
            return {
                "total_records": 0,
                "avg_confidence": 0.0,
                "top_emotion": "None",
                "emotion_counts": {},
                "source_counts": {},
                "latest_timestamp": "None",
            }

        cursor.execute("SELECT AVG(confidence) FROM prediction_logs;")
        avg_conf = cursor.fetchone()[0] or 0.0

        cursor.execute(
            """
            SELECT dominant_emotion, COUNT(*) as cnt 
            FROM prediction_logs 
            GROUP BY dominant_emotion 
            ORDER BY cnt DESC;
            """
        )
        emotion_rows = cursor.fetchall()
        emotion_counts = {row["dominant_emotion"]: row["cnt"] for row in emotion_rows}
        top_emotion = emotion_rows[0]["dominant_emotion"] if emotion_rows else "None"

        cursor.execute(
            """
            SELECT source, COUNT(*) as cnt 
            FROM prediction_logs 
            GROUP BY source 
            ORDER BY cnt DESC;
            """
        )
        source_counts = {row["source"]: row["cnt"] for row in cursor.fetchall()}

        cursor.execute("SELECT MAX(timestamp) FROM prediction_logs;")
        latest_ts = cursor.fetchone()[0] or "None"

        return {
            "total_records": total_records,
            "avg_confidence": round(float(avg_conf), 4),
            "top_emotion": top_emotion,
            "emotion_counts": emotion_counts,
            "source_counts": source_counts,
            "latest_timestamp": latest_ts,
        }


def delete_record_by_id(record_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    """Deletes a specific record by its primary key ID."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_logs WHERE id = ?;", (record_id,))
        conn.commit()
        return cursor.rowcount > 0


def clear_all_records(db_path: str = DEFAULT_DB_PATH) -> int:
    """Deletes all records from the prediction table."""
    init_db(db_path)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM prediction_logs;")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name='prediction_logs';")
        conn.commit()
        return cursor.rowcount


def seed_demo_records(db_path: str = DEFAULT_DB_PATH) -> int:
    """Seeds sample prediction records for immediate demonstration and visualization."""
    init_db(db_path)
    demo_samples = [
        ("Happy", 0.942, {"Happy": 0.942, "Neutral": 0.031, "Surprise": 0.015, "Sad": 0.005, "Fear": 0.003, "Angry": 0.002, "Disgust": 0.002}, "portrait_smile.jpg", "Upload"),
        ("Neutral", 0.815, {"Neutral": 0.815, "Sad": 0.082, "Happy": 0.045, "Surprise": 0.025, "Angry": 0.015, "Fear": 0.010, "Disgust": 0.008}, "passport_photo.png", "Upload"),
        ("Surprise", 0.893, {"Surprise": 0.893, "Fear": 0.062, "Happy": 0.025, "Neutral": 0.012, "Sad": 0.004, "Angry": 0.002, "Disgust": 0.002}, "group_reaction.jpg", "Upload"),
        ("Sad", 0.768, {"Sad": 0.768, "Neutral": 0.124, "Fear": 0.052, "Angry": 0.035, "Disgust": 0.012, "Happy": 0.005, "Surprise": 0.004}, "candid_expression.jpg", "Upload"),
        ("Angry", 0.854, {"Angry": 0.854, "Disgust": 0.071, "Sad": 0.042, "Neutral": 0.021, "Fear": 0.008, "Happy": 0.002, "Surprise": 0.002}, "debate_portrait.png", "Upload"),
        ("Fear", 0.732, {"Fear": 0.732, "Surprise": 0.145, "Sad": 0.065, "Neutral": 0.032, "Angry": 0.018, "Disgust": 0.005, "Happy": 0.003}, "thriller_screen.jpg", "Upload"),
        ("Happy", 0.911, {"Happy": 0.911, "Surprise": 0.048, "Neutral": 0.025, "Sad": 0.008, "Disgust": 0.003, "Angry": 0.003, "Fear": 0.002}, "sample_happy.png", "Sample"),
    ]

    count = 0
    for emo, conf, probs, img, src in demo_samples:
        insert_prediction(
            dominant_emotion=emo,
            confidence=conf,
            probabilities=probs,
            image_name=img,
            source=src,
            notes="Initial demonstration log",
            db_path=db_path,
        )
        count += 1
    return count


if __name__ == "__main__":
    init_db()
    print("Database initialized successfully at:", DEFAULT_DB_PATH)
    stats = fetch_db_stats()
    print("Initial stats:", stats)
