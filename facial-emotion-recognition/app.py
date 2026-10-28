"""
app.py - Polished Streamlit Application for Facial Emotion Recognition (CNN)

Features:
- Home / Overview: Abstract, objectives, applications, 7 emotion cards, workflow diagram.
- Emotion Detection: Image upload (JPG/JPEG/PNG), sample test selector,
  multi-face detection, bounding boxes, dominant emotion confidence,
  probability bar chart for all 7 classes, and automatic persistent SQLite logging.
- Model Evaluation: Real training curves, confusion matrix, and classification report
  with graceful handling when evaluation has not yet been executed.
- Database & History: Persistent SQLite database records, historical analytics,
  filtering, inspection, CSV export, and record management.
- Cached resource loading to avoid reloading on rerun.
"""

from datetime import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import database as db

# Page configuration
st.set_page_config(
    page_title="EmoScan CNN | Facial Emotion Recognition",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Dark Glassmorphism with Modern Typography)
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Gradient header text */
        .gradient-title {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.4rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .gradient-subtitle {
            color: #94a3b8;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }

        /* Glassmorphic card styling */
        .glass-card {
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            backdrop-filter: blur(12px);
            margin-bottom: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.35);
        }

        /* Workflow steps */
        .flow-container {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 16px 0;
        }
        .flow-step {
            flex: 1;
            min-width: 170px;
            background: rgba(15, 23, 42, 0.7);
            border-left: 4px solid #6366f1;
            border-radius: 8px;
            padding: 12px 14px;
        }
        .flow-step-num {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #818cf8;
            font-weight: 700;
        }
        .flow-step-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-top: 2px;
        }
        .flow-step-desc {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }

        /* Status Pills */
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .status-ok {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .status-warning {
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }
        .status-db {
            background: rgba(99, 102, 241, 0.15);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.3);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Constants & Color Mapping
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTION_COLORS = {
    "Angry": "#ef4444",      # Crimson
    "Disgust": "#10b981",    # Emerald
    "Fear": "#a855f7",       # Purple
    "Happy": "#f59e0b",      # Amber
    "Sad": "#3b82f6",        # Blue
    "Surprise": "#06b6d4",   # Cyan
    "Neutral": "#94a3b8",    # Slate Silver
}
EMOTION_EMOJIS = {
    "Angry": "😡",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐",
}

MODEL_PATH = "models/emotion_model.keras"
CONFIG_PATH = "models/config.json"
OUTPUT_DIR = "outputs"

# Ensure database table exists on application load
db.init_db()


@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Loads and caches the emotion recognition model once using Streamlit cache_resource."""
    from predict import load_trained_model

    return load_trained_model(MODEL_PATH, CONFIG_PATH)


@st.cache_resource(show_spinner=False)
def get_cached_cascade():
    """Loads and caches the Haar Cascade Classifier."""
    from predict import get_haar_cascade_path

    cascade_path = get_haar_cascade_path()
    return cv2.CascadeClassifier(cascade_path)


def generate_sample_face(emotion: str) -> np.ndarray:
    """Generates an illustrative stylized facial image for zero-friction testing."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[:] = (30, 32, 40)  # Dark background

    # Face circle
    cv2.circle(img, (100, 100), 75, (235, 200, 175), -1)
    cv2.circle(img, (100, 100), 75, (180, 150, 130), 2)

    # Eyes
    cv2.circle(img, (75, 80), 8, (40, 40, 40), -1)
    cv2.circle(img, (125, 80), 8, (40, 40, 40), -1)

    # Mouth based on emotion
    if emotion == "Happy":
        cv2.ellipse(img, (100, 115), (35, 25), 0, 0, 180, (40, 40, 40), 3)
    elif emotion == "Sad":
        cv2.ellipse(img, (100, 135), (30, 20), 0, 180, 360, (40, 40, 40), 3)
    elif emotion == "Surprise":
        cv2.ellipse(img, (100, 125), (14, 20), 0, 0, 360, (40, 40, 40), -1)
    elif emotion == "Angry":
        cv2.line(img, (60, 68), (85, 78), (40, 40, 40), 3)
        cv2.line(img, (140, 68), (115, 78), (40, 40, 40), 3)
        cv2.line(img, (75, 130), (125, 130), (40, 40, 40), 3)
    else:  # Neutral
        cv2.line(img, (80, 125), (120, 125), (40, 40, 40), 3)

    return img


# Sidebar Navigation & System Status
with st.sidebar:
    st.markdown("### 🧠 EmoScan CNN")
    st.markdown("<p style='font-size:0.85rem; color:#94a3b8;'>Facial Emotion Recognition System</p>", unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🔍 Emotion Detection",
            "📊 Model Evaluation",
            "🗄️ Database & History",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### System Status")
    model_obj, config_obj, model_err = get_cached_model()

    if model_obj is not None:
        st.markdown(
            '<span class="status-pill status-ok">● Model Loaded (.keras)</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-pill status-warning">▲ Model Not Loaded</span>',
            unsafe_allow_html=True,
        )

    # Database quick count
    db_stats = db.fetch_db_stats()
    st.markdown(
        f'<span class="status-pill status-db">🗄️ Database: {db_stats["total_records"]} Records</span>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption("Powered by TensorFlow 2.x, OpenCV & SQLite")


# ==============================================================================
# TAB 1: OVERVIEW & HOME
# ==============================================================================
if nav_selection == "🏠 Overview":
    st.markdown('<div class="gradient-title">Facial Emotion Recognition Using CNN</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">End-to-End Affect Recognition System with Deep Learning, Computer Vision & Persistent SQLite Storage</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color:#f8fafc; margin-top:0;">📌 Project Abstract</h3>
                <p style="color:#cbd5e1; line-height:1.6; font-size:0.95rem;">
                    Facial Emotion Recognition (FER) is a pivotal challenge in affective computing and computer vision.
                    This project implements an end-to-end convolutional neural network pipeline trained on 
                    the <strong>FER-2013 benchmark dataset</strong> to classify human facial expressions into seven primary 
                    affective states: <em>Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral</em>.
                </p>
                <p style="color:#cbd5e1; line-height:1.6; font-size:0.95rem;">
                    The system combines <strong>OpenCV Haar Cascade</strong> for real-time frontal face localization 
                    with a custom <strong>deep CNN architecture</strong> featuring convolution blocks, batch normalization,
                    regularization via dropout, and softmax classification, integrated with an embedded 
                    <strong>SQLite Database</strong> for historical prediction tracking and auditability.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔄 System Architecture Workflow")
        st.markdown(
            """
            <div class="flow-container">
                <div class="flow-step">
                    <div class="flow-step-num">Step 1</div>
                    <div class="flow-step-title">Image Acquisition</div>
                    <div class="flow-step-desc">Accepts user-uploaded JPG/PNG image or sample face.</div>
                </div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 2</div>
                    <div class="flow-step-title">Face Detection</div>
                    <div class="flow-step-desc">Haar Feature Cascade isolates facial bounding box.</div>
                </div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 3</div>
                    <div class="flow-step-title">Preprocessing</div>
                    <div class="flow-step-desc">Grayscale conversion, 48×48 resize, [0, 1] normalization.</div>
                </div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 4</div>
                    <div class="flow-step-title">CNN Inference</div>
                    <div class="flow-step-desc">4-block feature extractor computes 7-class logits.</div>
                </div>
                <div class="flow-step">
                    <div class="flow-step-num">Step 5</div>
                    <div class="flow-step-title">Database Logging</div>
                    <div class="flow-step-desc">Records saved to SQLite with timestamps and confidence.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color:#f8fafc; margin-top:0;">🎯 Objectives & Key Deliverables</h3>
                <ul style="color:#cbd5e1; font-size:0.95rem; line-height:1.65;">
                    <li><strong>Accurate Emotion Localization:</strong> Robust face detection handling variable poses and scale.</li>
                    <li><strong>Rigorous Preprocessing Parity:</strong> Identical normalization and spatial dimension transformations across training and inference.</li>
                    <li><strong>Persistent Audit Logging:</strong> Structured SQLite database recording predictions, confidence, timestamps, and bounding boxes.</li>
                    <li><strong>Production-Grade UX:</strong> Multi-face support, probability histograms, CSV data export, and diagnostics.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style="color:#f8fafc; margin-top:0;">🎭 Seven Emotion Classes</h3>
                <p style="color:#94a3b8; font-size:0.85rem;">Standard Paul Ekman universal emotion taxonomy used in FER-2013:</p>
            """,
            unsafe_allow_html=True,
        )

        for label in EMOTION_LABELS:
            color = EMOTION_COLORS[label]
            emoji = EMOTION_EMOJIS[label]
            st.markdown(
                f"""
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:10px; padding:8px 12px; background:rgba(15,23,42,0.5); border-left:3px solid {color}; border-radius:6px;">
                    <div><span style="font-size:1.3rem; margin-right:8px;">{emoji}</span> <strong>{label}</strong></div>
                    <span style="font-family:'JetBrains Mono',monospace; font-size:0.75rem; color:{color};">{color}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="glass-card">
                <h4 style="color:#f8fafc; margin-top:0;">🌐 Practical Applications</h4>
                <ul style="color:#94a3b8; font-size:0.85rem; line-height:1.5; padding-left:18px;">
                    <li><strong>Healthcare & Teletherapy:</strong> Longitudinal mood monitoring with database logs.</li>
                    <li><strong>Driver Assistance (ADAS):</strong> Drowsiness and frustration alerts.</li>
                    <li><strong>E-Learning Platforms:</strong> Real-time student engagement tracking.</li>
                    <li><strong>User Experience (UX):</strong> Usability testing reaction analysis over time.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# TAB 2: EMOTION DETECTION
# ==============================================================================
elif nav_selection == "🔍 Emotion Detection":
    st.markdown('<div class="gradient-title">Live Facial Emotion Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Upload an image or test with sample portraits to detect faces, classify affective states, and log to the database.</div>',
        unsafe_allow_html=True,
    )

    from predict import detect_and_predict

    # Quick Model Check
    if model_obj is None:
        st.error(
            "⚠️ **Trained Model Not Loaded**: A trained model was not found at `models/emotion_model.keras`.\n\n"
            "To train a model run `python train.py` from the project directory."
        )

    # Input method selection
    input_choice = st.radio(
        "Choose Image Source:",
        ["📤 Upload Image (JPG, JPEG, PNG)", "🎨 Pre-loaded Sample Expressions"],
        horizontal=True,
    )

    image_to_process = None
    image_name = "unnamed_image"
    source_label = "Upload"

    if input_choice.startswith("📤"):
        uploaded_file = st.file_uploader(
            "Choose a facial portrait image...",
            type=["jpg", "jpeg", "png"],
            help="High-quality frontal portrait recommended for best results.",
        )
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file)
                image_to_process = np.array(pil_img.convert("RGB"))
                image_name = uploaded_file.name
                source_label = "Upload"
            except Exception as e:
                st.error(f"Failed to read image: {e}")
    else:
        sample_choice = st.selectbox(
            "Select Sample Face Expression:",
            ["Happy", "Sad", "Surprise", "Angry", "Neutral"],
        )
        sample_img = generate_sample_face(sample_choice)
        image_to_process = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        image_name = f"sample_{sample_choice.lower()}.png"
        source_label = "Sample"
        st.info(f"Loaded generated sample facial expression for: **{sample_choice}**")

    # Processing and Results Display
    if image_to_process is not None:
        st.markdown("---")
        det_col1, det_col2 = st.columns([1, 1], gap="medium")

        with det_col1:
            st.markdown("#### Original Input Image")
            st.image(image_to_process, use_container_width=True)

        with det_col2:
            st.markdown("#### Detection & Recognition HUD")

            if model_obj is None:
                st.warning("Prediction requires the trained model. Please train or load a model first.")
            else:
                with st.spinner("Analyzing facial features and computing emotion probabilities..."):
                    # Convert RGB to BGR for OpenCV pipeline
                    bgr_input = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
                    result = detect_and_predict(bgr_input, model_obj, config_obj)

                if not result["success"]:
                    st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
                elif result["num_faces"] == 0:
                    st.warning(
                        "🔍 **No Face Detected**: Haar cascade could not detect a frontal face in the image.\n\n"
                        "- Ensure face is front-facing and unobstructed.\n"
                        "- Ensure adequate lighting and contrast."
                    )
                else:
                    # Convert annotated BGR back to RGB for Streamlit display
                    annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)

                    if result.get("warning"):
                        st.caption(f"ℹ️ {result['warning']}")

                    # Persistent SQLite Logging
                    # Prevent duplicate insertion on re-render with session_state tracker
                    run_signature = f"{image_name}_{len(result['faces'])}_{result['faces'][0]['dominant_emotion']}_{result['faces'][0]['confidence']}"
                    if st.session_state.get("last_logged_signature") != run_signature:
                        inserted_ids = db.log_multi_faces(
                            faces=result["faces"],
                            image_name=image_name,
                            source=source_label,
                        )
                        st.session_state["last_logged_signature"] = run_signature
                        st.success(f"💾 **Logged {len(inserted_ids)} face record(s) to SQLite Database** (Record IDs: {inserted_ids})")
                    else:
                        st.caption("💾 Record already saved to SQLite Database.")

        # Results Breakdown Section
        if model_obj is not None and image_to_process is not None and result.get("success") and result["num_faces"] > 0:
            st.markdown("---")
            st.markdown(f"### 📊 Detailed Findings ({result['num_faces']} Face{'s' if result['num_faces'] > 1 else ''} Detected)")

            for idx, face in enumerate(result["faces"]):
                with st.container():
                    st.markdown(
                        f"""
                        <div class="glass-card">
                            <h4 style="margin-top:0; color:#f8fafc;">
                                Face #{idx + 1} — Dominant Emotion: 
                                <span style="color:{EMOTION_COLORS.get(face['dominant_emotion'], '#fff')};">
                                    {EMOTION_EMOJIS.get(face['dominant_emotion'], '')} {face['dominant_emotion']}
                                </span> 
                                ({face['confidence'] * 100:.2f}% Confidence)
                            </h4>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    fc1, fc2 = st.columns([1, 2], gap="medium")

                    with fc1:
                        st.markdown("##### Preprocessed CNN Input (48×48 Grayscale)")
                        st.image(
                            face["face_crop"],
                            caption="Normalized 48x48 tensor input",
                            width=160,
                            clamp=True,
                        )
                        st.caption(f"Bounding Box: `(x={face['bbox'][0]}, y={face['bbox'][1]}, w={face['bbox'][2]}, h={face['bbox'][3]})`")

                    with fc2:
                        st.markdown("##### Emotion Probability Distribution")
                        prob_items = list(face["probabilities"].items())
                        chart_df = pd.DataFrame(
                            {
                                "Emotion": [f"{EMOTION_EMOJIS.get(k, '')} {k}" for k, _ in prob_items],
                                "RawEmotion": [k for k, _ in prob_items],
                                "Probability": [v for _, v in prob_items],
                                "Percentage": [f"{v * 100:.2f}%" for _, v in prob_items],
                            }
                        )

                        st.bar_chart(
                            data=chart_df.set_index("Emotion")["Probability"],
                            color="#6366f1",
                            horizontal=True,
                            use_container_width=True,
                        )

                        with st.expander("View Exact Class Probability Table", expanded=False):
                            st.dataframe(
                                chart_df[["Emotion", "Percentage"]].rename(
                                    columns={"Emotion": "Emotion Category", "Percentage": "Softmax Confidence"}
                                ),
                                hide_index=True,
                                use_container_width=True,
                            )


# ==============================================================================
# TAB 3: MODEL EVALUATION
# ==============================================================================
elif nav_selection == "📊 Model Evaluation":
    st.markdown('<div class="gradient-title">Model Evaluation & Metrics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Quantitative assessment derived strictly from real test data and training history.</div>',
        unsafe_allow_html=True,
    )

    eval_json_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    cm_img_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    curves_img_path = os.path.join(OUTPUT_DIR, "training_curves.png")
    history_json_path = os.path.join(OUTPUT_DIR, "training_history.json")

    has_eval_results = os.path.exists(eval_json_path)
    has_history = os.path.exists(history_json_path)

    if not has_eval_results and not has_history:
        st.markdown(
            """
            <div class="glass-card" style="border-left: 4px solid #f59e0b;">
                <h3 style="color:#fbbf24; margin-top:0;">⚠️ Evaluation Not Yet Executed</h3>
                <p style="color:#cbd5e1; line-height:1.6;">
                    No recorded evaluation artifacts were found in the <code>outputs/</code> directory.
                    In accordance with project integrity standards, metrics and charts are generated strictly 
                    from genuine model evaluations and never fabricated.
                </p>
                <p style="color:#94a3b8;">
                    To generate evaluation artifacts:
                </p>
                <ol style="color:#cbd5e1; line-height:1.7;">
                    <li>Ensure training has produced <code>models/emotion_model.keras</code> and <code>outputs/test_data.npz</code>.</li>
                    <li>Execute the evaluation pipeline: <code>python evaluate.py</code></li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("#### Run Quick Evaluation / Verification")
        if st.button("🚀 Trigger Evaluation Now via `evaluate.py`"):
            from evaluate import evaluate as run_eval

            with st.spinner("Running quantitative model evaluation..."):
                success = run_eval(
                    model_path=MODEL_PATH,
                    config_path=CONFIG_PATH,
                    test_data_path="outputs/test_data.npz",
                    output_dir=OUTPUT_DIR,
                )
            if success:
                st.success("Evaluation completed successfully! Refresh page to see full metrics.")
                st.rerun()
            else:
                st.error("Evaluation failed. Please verify that a trained model and test data exist.")

    else:
        # Load and display real evaluation metrics
        if has_eval_results:
            with open(eval_json_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)

            st.markdown("### 🏆 Held-Out Test Set Performance")
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            m_col1.metric("Overall Accuracy", f"{eval_data.get('accuracy', 0) * 100:.2f}%")
            m_col2.metric("Macro Precision", f"{eval_data.get('macro_precision', 0) * 100:.2f}%")
            m_col3.metric("Macro Recall", f"{eval_data.get('macro_recall', 0) * 100:.2f}%")
            m_col4.metric("Macro F1-Score", f"{eval_data.get('macro_f1', 0) * 100:.2f}%")

            st.caption(f"Evaluated on {eval_data.get('num_test_samples', 0)} real held-out test samples.")

        st.markdown("---")
        ev_col1, ev_col2 = st.columns(2, gap="large")

        with ev_col1:
            st.markdown("#### 🎯 Confusion Matrix (Normalized)")
            if os.path.exists(cm_img_path):
                st.image(cm_img_path, caption="Confusion Matrix on Test Split", use_container_width=True)
            else:
                st.info("Confusion matrix image not yet generated.")

        with ev_col2:
            st.markdown("#### 📈 Training & Validation Curves")
            if os.path.exists(curves_img_path):
                st.image(curves_img_path, caption="Accuracy & Loss Curves", use_container_width=True)
            elif has_history:
                with open(history_json_path, "r", encoding="utf-8") as f:
                    hist = json.load(f)
                epochs_list = list(range(1, len(hist.get("loss", [])) + 1))
                hist_df = pd.DataFrame(
                    {
                        "Epoch": epochs_list,
                        "Training Accuracy": hist.get("accuracy", []),
                        "Validation Accuracy": hist.get("val_accuracy", []),
                    }
                ).set_index("Epoch")
                st.line_chart(hist_df, use_container_width=True)
            else:
                st.info("Training curve plot not yet generated.")

        # Classification Report Table
        if has_eval_results and "classification_report" in eval_data:
            st.markdown("---")
            st.markdown("#### 📋 Detailed Per-Class Classification Report")
            cr = eval_data["classification_report"]
            rows = []
            for label in EMOTION_LABELS:
                if label in cr:
                    rows.append(
                        {
                            "Emotion Category": label,
                            "Precision": f"{cr[label]['precision'] * 100:.2f}%",
                            "Recall": f"{cr[label]['recall'] * 100:.2f}%",
                            "F1-Score": f"{cr[label]['f1-score'] * 100:.2f}%",
                            "Test Samples (Support)": int(cr[label]["support"]),
                        }
                    )
            if rows:
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ==============================================================================
# TAB 4: DATABASE & HISTORY
# ==============================================================================
elif nav_selection == "🗄️ Database & History":
    st.markdown('<div class="gradient-title">Prediction Database & History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="gradient-subtitle">Structured SQLite persistence of all facial emotion scans, confidence distributions, and timestamps.</div>',
        unsafe_allow_html=True,
    )

    stats = db.fetch_db_stats()

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Stored Records", stats["total_records"])
    kpi2.metric(
        "Most Common Emotion",
        f"{EMOTION_EMOJIS.get(stats['top_emotion'], '')} {stats['top_emotion']}",
    )
    kpi3.metric("Average Confidence", f"{stats['avg_confidence'] * 100:.2f}%")
    kpi4.metric("Last Logged Activity", stats["latest_timestamp"].split()[-1] if stats["latest_timestamp"] != "None" else "None")

    st.markdown("---")

    # If database is empty, provide quick seed button
    if stats["total_records"] == 0:
        st.info("💡 The database is currently empty. Run an emotion detection scan or click below to populate sample historical entries.")
        if st.button("🌱 Populate Sample Demonstration Records"):
            num_seeded = db.seed_demo_records()
            st.success(f"Added {num_seeded} sample records to the SQLite database!")
            st.rerun()

    else:
        # Analytics Visualizations
        chart_col1, chart_col2 = st.columns([3, 2], gap="large")

        with chart_col1:
            st.markdown("#### 📊 Historical Emotion Frequency Distribution")
            emo_counts = stats["emotion_counts"]
            if emo_counts:
                hist_chart_df = pd.DataFrame(
                    {
                        "Emotion": [f"{EMOTION_EMOJIS.get(k, '')} {k}" for k in emo_counts.keys()],
                        "Frequency": list(emo_counts.values()),
                    }
                ).set_index("Emotion")

                st.bar_chart(
                    hist_chart_df["Frequency"],
                    color="#8b5cf6",
                    horizontal=True,
                    use_container_width=True,
                )

        with chart_col2:
            st.markdown("#### 📥 Prediction Source Breakdown")
            src_counts = stats["source_counts"]
            if src_counts:
                src_df = pd.DataFrame(
                    {
                        "Source": list(src_counts.keys()),
                        "Scans": list(src_counts.values()),
                    }
                ).set_index("Source")
                st.dataframe(src_df, use_container_width=True)

        st.markdown("---")

        # Database Query Filters
        st.markdown("### 🔍 Query & Filter Records")
        f_col1, f_col2, f_col3 = st.columns([2, 2, 1])

        with f_col1:
            emotion_filter = st.selectbox(
                "Filter by Dominant Emotion:",
                ["All"] + EMOTION_LABELS,
            )

        with f_col2:
            source_filter = st.selectbox(
                "Filter by Source:",
                ["All", "Upload", "Sample"],
            )

        with f_col3:
            record_limit = st.selectbox(
                "Max Records:",
                [25, 50, 100, 200],
                index=1,
            )

        # Fetch records
        records_df = db.fetch_all_predictions(
            limit=record_limit,
            emotion_filter=emotion_filter,
            source_filter=source_filter,
        )

        if records_df.empty:
            st.warning("No records match the selected filter criteria.")
        else:
            # Display formatted columns
            display_df = records_df.copy()
            display_df["confidence_pct"] = (display_df["confidence"] * 100).round(2).astype(str) + "%"

            col_subset = [
                "id",
                "timestamp",
                "image_name",
                "face_index",
                "dominant_emotion",
                "confidence_pct",
                "bbox",
                "source",
            ]
            display_columns = [c for c in col_subset if c in display_df.columns]

            st.dataframe(
                display_df[display_columns].rename(
                    columns={
                        "id": "ID",
                        "timestamp": "Timestamp",
                        "image_name": "Image File",
                        "face_index": "Face #",
                        "dominant_emotion": "Dominant Emotion",
                        "confidence_pct": "Confidence",
                        "bbox": "Bounding Box",
                        "source": "Source",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

            # Export & Management Row
            action_c1, action_c2 = st.columns([1, 1], gap="large")

            with action_c1:
                csv_data = records_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Export Stored Records to CSV",
                    data=csv_data,
                    file_name=f"emotion_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            with action_c2:
                with st.expander("⚠️ Database Maintenance (Clear Records)", expanded=False):
                    st.caption("This action permanently deletes all saved prediction history from SQLite.")
                    confirm_clear = st.checkbox("Confirm deletion of all database records")
                    if st.button("🗑️ Delete All Database Records", disabled=not confirm_clear):
                        deleted = db.clear_all_records()
                        st.success(f"Successfully cleared {deleted} records from database.")
                        st.rerun()

            # Record Probability Breakdown Inspector
            with st.expander("🔎 Inspect Exact 7-Class Probabilities for a Record", expanded=False):
                selected_id = st.selectbox(
                    "Select Record ID:",
                    records_df["id"].tolist(),
                )
                selected_row = records_df[records_df["id"] == selected_id].iloc[0]

                p_cols = {
                    "Angry": selected_row.get("angry_prob", 0.0),
                    "Disgust": selected_row.get("disgust_prob", 0.0),
                    "Fear": selected_row.get("fear_prob", 0.0),
                    "Happy": selected_row.get("happy_prob", 0.0),
                    "Sad": selected_row.get("sad_prob", 0.0),
                    "Surprise": selected_row.get("surprise_prob", 0.0),
                    "Neutral": selected_row.get("neutral_prob", 0.0),
                }

                detail_df = pd.DataFrame(
                    {
                        "Emotion": [f"{EMOTION_EMOJIS[k]} {k}" for k in p_cols.keys()],
                        "Probability": list(p_cols.values()),
                        "Percentage": [f"{v * 100:.2f}%" for v in p_cols.values()],
                    }
                )

                st.markdown(
                    f"**Record #{selected_id}** — Image: `{selected_row['image_name']}` | Logged at: `{selected_row['timestamp']}`"
                )
                st.bar_chart(
                    detail_df.set_index("Emotion")["Probability"],
                    color="#6366f1",
                    horizontal=True,
                    use_container_width=True,
                )
