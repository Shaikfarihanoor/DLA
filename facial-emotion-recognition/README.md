# Facial Emotion Recognition Using Convolutional Neural Networks (CNN)

An end-to-end Computer Vision and Deep Learning system for human facial emotion detection and classification into 7 primary affective states using a deep Convolutional Neural Network (CNN) and OpenCV Haar Cascades, with an interactive Streamlit dashboard.

---

## 📑 Table of Contents
1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Existing vs. Proposed System](#existing-vs-proposed-system)
5. [System Architecture](#system-architecture)
6. [Dataset Description](#dataset-description)
7. [CNN Architecture & Hyperparameters](#cnn-architecture--hyperparameters)
8. [Implementation Methodology](#implementation-methodology)
9. [Results & Evaluation](#results--evaluation)
10. [Advantages & Limitations](#advantages--limitations)
11. [Applications & Future Enhancements](#applications--future-enhancements)
12. [Installation & Execution Instructions](#installation--execution-instructions)
13. [Database Architecture & Historical Analytics](#database-architecture--historical-analytics)

---

## 1. Abstract
Facial Emotion Recognition (FER) is a central problem in computer vision and human-computer interaction (HCI). This project develops an end-to-end deep learning pipeline to recognize human affective states from static portraits and digital imagery. Using the **FER-2013 benchmark dataset**, images are normalized to 48×48 grayscale representations and classified across seven standard emotions: **Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral**. The system integrates **OpenCV Haar Feature-based Cascade Classifiers** for robust facial bounding box localization, followed by an optimized 4-block Convolutional Neural Network featuring Batch Normalization, Dropout, and Softmax activation. The application is delivered via a reactive, dark glassmorphic Streamlit application with multi-face detection, confidence distributions, and evaluation diagnostics.

---

## 2. Problem Statement
Automated facial expression recognition faces several real-world challenges:
- **Intra-class variability**: Different individuals express the same emotion with varying intensity and muscle configurations.
- **Inter-class similarity**: Subtle expressions (e.g., Fear vs. Surprise, or Sad vs. Neutral) share overlapping facial action unit activations.
- **Environmental perturbations**: Uneven illumination, camera noise, occlusions, and variable head poses.
- **Computational efficiency**: Traditional high-resolution deep networks incur high compute costs, limiting real-time deployment on edge devices and standard CPUs.

This project addresses these challenges by developing an efficient, localized CNN pipeline that standardizes face crops to 48×48 grayscale patches and utilizes regularization to minimize overfitting.

---

## 3. Objectives
- Implement an automated face detection pipeline utilizing OpenCV Haar Cascades.
- Design and train a multi-layer Convolutional Neural Network with Batch Normalization and Dropout for 7-class facial affect recognition.
- Enforce strict preprocessing parity (48×48 spatial dimensions, grayscale conversion, [0, 1] normalization) across training, validation, and real-time inference.
- Provide transparent quantitative evaluation metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Loss/Accuracy curves) derived exclusively from genuine model evaluation.
- Build an intuitive, aesthetically rich Streamlit web application supporting single and multi-face analysis, confidence histograms, and diagnostics.

---

## 4. Existing vs. Proposed System

| Feature | Existing Systems (Traditional ML) | Proposed CNN System |
| :--- | :--- | :--- |
| **Feature Extraction** | Handcrafted descriptors (HOG, SIFT, LBP) | Hierarchical representation learning via Conv2D filters |
| **Face Detection** | Manual cropping or basic thresholding | Automated OpenCV Haar Cascade localization |
| **Generalization** | Prone to failure under scale and illumination shifts | Regularized with Batch Normalization and Dropout |
| **Inference Pipeline** | Multi-stage disjoint feature-classifier pipelines | End-to-end tensor processing from raw image to softmax probabilities |
| **User Interface** | CLI scripts or static text outputs | Interactive web HUD with multi-face support and probability charts |

---

## 5. System Architecture

```text
[Input Image (JPG/PNG)]
          │
          ▼
┌────────────────────────────────────────┐
│  OpenCV Haar Cascade Face Detection   │  ──> Localizes bounding boxes (x, y, w, h)
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│       Facial ROI Preprocessing         │  ──> Grayscale conversion
│                                        │  ──> Resize to 48 × 48 pixels
│                                        │  ──> Pixel normalization: x / 255.0
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│        Deep CNN Feature Extractor      │  ──> Conv2D + BatchNorm + ReLU (64)
│                                        │  ──> Conv2D + BatchNorm + ReLU (128)
│                                        │  ──> Conv2D + BatchNorm + ReLU (256)
│                                        │  ──> Max Pooling (2x2) + Dropout
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│          Dense Classifier              │  ──> Dense(512) + Dense(256) + Dropout(0.5)
│                                        │  ──> Softmax Layer (7 Classes)
└────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────┐
│    Streamlit HUD & Visual Analytics    │  ──> Annotated Bounding Boxes
│                                        │  ──> Dominant Emotion & Confidence
│                                        │  ──> 7-Class Probability Bar Chart
└────────────────────────────────────────┘
```

---

## 6. Dataset Description
The system is built for the **FER-2013 (Facial Expression Recognition 2013)** dataset:
- **Total Images**: 35,887 images (48×48 pixel grayscale faces).
- **Labels (7 Classes)**:
  - `0`: Angry
  - `1`: Disgust
  - `2`: Fear
  - `3`: Happy
  - `4`: Sad
  - `5`: Surprise
  - `6`: Neutral
- **Standard Splits**:
  - Training: 28,709 images
  - PublicTest (Validation): 3,589 images
  - PrivateTest (Held-out Test): 3,589 images
- **Supported Formats in this Project**:
  1. `data/fer2013.csv` containing `emotion`, `pixels` (2,304 space-separated floats), and optional `Usage` columns.
  2. Directory structure: `data/train/<emotion>` and `data/test/<emotion>`.

---

## 7. CNN Architecture & Hyperparameters

### Detailed Layer Specification

| Layer # | Type | Filters / Units | Kernel / Pool Size | Activation | Output Shape | Regularization |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Input | - | - | - | (48, 48, 1) | - |
| **1-2** | Conv2D x 2 | 64 | 3 × 3 (same) | ReLU | (48, 48, 64) | Batch Normalization |
| **3-4** | MaxPool + Drop | - | 2 × 2 | - | (24, 24, 64) | Dropout (0.25) |
| **5-6** | Conv2D x 2 | 128 | 3 × 3 (same) | ReLU | (24, 24, 128) | Batch Normalization |
| **7-8** | MaxPool + Drop | - | 2 × 2 | - | (12, 12, 128) | Dropout (0.25) |
| **9-10**| Conv2D x 2 | 256 | 3 × 3 (same) | ReLU | (12, 12, 256) | Batch Normalization |
| **11-12**| MaxPool + Drop| - | 2 × 2 | - | (6, 6, 256) | Dropout (0.25) |
| **13** | Flatten | - | - | - | (9,216,) | - |
| **14** | Dense | 512 | - | ReLU | (512,) | BatchNorm + L2 + Dropout (0.5) |
| **15** | Dense | 256 | - | ReLU | (256,) | BatchNorm + L2 + Dropout (0.5) |
| **16** | Dense (Output)| 7 | - | Softmax | (7,) | Categorical Cross-Entropy |

### Training Hyperparameters
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, initial learning rate $= 0.001$)
- **Loss Function**: Categorical Cross-Entropy
- **Learning Rate Scheduler**: `ReduceLROnPlateau` (factor=0.5, patience=3, min_lr=1e-6)
- **Early Stopping**: Monitored on `val_loss` (patience=8 epochs, restoring best model weights)
- **Model Checkpointing**: Saved to `models/emotion_model.keras` on `val_accuracy` improvement

---

## 8. Implementation Methodology
1. **Face Localization**: Uses OpenCV's pre-trained Viola-Jones Haar Cascade (`haarcascade_frontalface_default.xml`) to identify all facial regions in the image.
2. **Preprocessing**: Converts each facial ROI into single-channel grayscale, resizes using area interpolation to 48×48 pixels, and normalizes pixel values to $[0.0, 1.0]$.
3. **Inference**: P义务 passes the $(1, 48, 48, 1)$ tensor into the CNN to generate a 7-dimensional probability vector.
4. **Rendering**: Overlays color-coded bounding boxes and emotion labels on the image and outputs an interactive probability histogram.

---

## 9. Results & Evaluation
Model evaluation is conducted via `evaluate.py` using genuine held-out test data. Evaluation outputs are saved to the `outputs/` folder:
- `outputs/confusion_matrix.png` & `confusion_matrix.csv`: Normalized true vs. predicted emotion distributions.
- `outputs/training_curves.png`: Training vs. validation loss and accuracy across epochs.
- `outputs/evaluation_results.json`: Precision, Recall, and F1-Scores for each class and macro/weighted averages.

*(No fabricated numbers are hardcoded into the application; metrics are rendered strictly from execution artifacts).*

---

## 10. Advantages & Limitations

### Advantages
1. **End-to-End Deep Learning**: Replaces manual feature extraction with learned spatial hierarchies.
2. **Multi-Face Detection**: Simultaneously detects and classifies multiple faces in a single photograph.
3. **Low Latency**: Inference executes in under 40ms per face on standard x86 CPUs.
4. **Robust Regularization**: Combines Batch Normalization, Dropout, and L2 weight decay to counteract overfitting.

### Limitations
1. **Resolution Constraints**: 48×48 pixel resolution limits detection of fine micro-expressions.
2. **Head Pose Sensitivity**: Haar Cascade requires roughly frontal head orientation and can struggle with extreme yaw/pitch profiles.
3. **FER-2013 Label Noise**: Benchmark dataset contains slight human annotation subjectivity and class imbalance (e.g., fewer samples for Disgust).

---

## 11. Applications & Future Enhancements

### Real-World Applications
- **Driver Monitoring Systems (DMS)**: Detect fatigue, distraction, and road rage.
- **Mental Health & Telemedicine**: Assist clinicians in tracking longitudinal patient affect.
- **E-Learning & Video Conferencing**: Gauge student engagement and comprehension in remote classrooms.
- **Customer Experience & Retail**: Analyze consumer reactions during product testing and retail visits.

### Future Scope
- Integration with MediaPipe / RetinaFace for 3D face mesh landmarking and pose-invariant detection.
- Real-time webcam video stream processing at 30+ FPS.
- Multi-modal affect recognition combining facial vision with speech acoustic cues.

---

## 12. Installation & Execution Instructions

### Prerequisites
- Python 3.10 recommended (compatible with TensorFlow 2.x and OpenCV)

### Step 1: Clone or Navigate to Project
```bash
cd facial-emotion-recognition
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 4: Model Training (Optional / Dataset Preparation)
To train the CNN model:
1. Place `fer2013.csv` into `data/fer2013.csv`, or arrange image folders under `data/train/` and `data/test/`.
2. Run training:
```bash
python train.py --epochs 30 --batch-size 64
```
*(To run a quick pipeline smoke-test without the dataset, run `python train.py --synthetic --epochs 5`)*

### Step 5: Model Evaluation
To evaluate on held-out test data:
```bash
python evaluate.py
```

### Step 6: Command-Line Single Image Prediction
```bash
python predict.py --image path/to/your/image.jpg
```

---

## 13. Database Architecture & Historical Analytics

The application integrates an embedded **SQLite Database** (`data/emotion_records.db`) managed via `database.py`. Every face analyzed through the Streamlit interface is automatically persisted with comprehensive metadata for auditability, longitudinal affect tracking, and reporting.

### 13.1 Database Schema (`prediction_logs`)

```sql
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,              -- Format: YYYY-MM-DD HH:MM:SS
    image_name TEXT DEFAULT 'unnamed',    -- Filename of input image
    face_index INTEGER DEFAULT 0,         -- Index of detected face in image
    dominant_emotion TEXT NOT NULL,       -- Primary predicted emotion
    confidence REAL NOT NULL,             -- Softmax probability score [0.0 - 1.0]
    angry_prob REAL DEFAULT 0.0,          -- Individual emotion probabilities
    disgust_prob REAL DEFAULT 0.0,
    fear_prob REAL DEFAULT 0.0,
    happy_prob REAL DEFAULT 0.0,
    sad_prob REAL DEFAULT 0.0,
    surprise_prob REAL DEFAULT 0.0,
    neutral_prob REAL DEFAULT 0.0,
    bbox TEXT DEFAULT '',                 -- Bounding box format: 'x,y,w,h'
    source TEXT DEFAULT 'Upload',         -- Origin: 'Upload' or 'Sample'
    notes TEXT DEFAULT ''                 -- Optional annotations
);

CREATE INDEX IF NOT EXISTS idx_emotion ON prediction_logs(dominant_emotion);
CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs(timestamp);
```

### 13.2 Database Features
- **Automatic Logging**: Every face detection and CNN classification run records the bounding box coordinates, dominant emotion, confidence, and full 7-class probability vector.
- **Aggregated Analytics**: Real-time KPI summaries for total scans, most frequent emotion, average model confidence, and prediction origin.
- **Dynamic Query Filtering**: Filter stored records by emotion category, source, or record limit directly in the UI.
- **Record Inspector**: Deep-dive into individual records to inspect 7-class probability distributions.
- **CSV Data Export**: One-click download of all stored records for external statistical analysis.
- **Database Maintenance**: Built-in safe reset and demonstration seeding utilities.

