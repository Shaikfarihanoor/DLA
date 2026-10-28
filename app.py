"""
Root entry point launcher for Facial Emotion Recognition Streamlit App.
Allows running `streamlit run app.py` directly from workspace root.
"""
import os
import sys

PROJECT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "facial-emotion-recognition")
if os.path.exists(PROJECT_DIR):
    os.chdir(PROJECT_DIR)
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)

app_path = os.path.join(PROJECT_DIR, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), app_path, "exec")
    exec(code)
