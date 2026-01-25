"""SkiSense configuration settings."""
import os

# =============================================================================
# Path Settings
# =============================================================================
# Get the project root directory (where run.py is located)
# Use current working directory as base (assumes running from project root)
PROJECT_ROOT = os.getcwd()

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# =============================================================================
# Processing Settings
# =============================================================================
DEBUG = False           # True: show all logs, False: suppress logs
SHOW_GUI = False        # True: show preview window, False: headless mode
DEVICE_PREFERENCE = "auto"  # "auto" | "cuda" | "cpu"

# =============================================================================
# Model Settings
# =============================================================================
YOLO_MODEL = "yolov8x.pt"           # YOLOv8 model file
POSE_MODEL = "pose_landmarker.task"  # MediaPipe pose landmarker model

# Model download URL (if not exists)
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
