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
# Detection & Tracking Settings
# =============================================================================
YOLO_CONFIDENCE = 0.3         # YOLO detection confidence threshold (0.0-1.0)
DEEPSORT_MAX_AGE = 30         # Frames to keep track without detection
DEEPSORT_N_INIT = 1           # Frames required to confirm new track
POSE_PRESENCE_CONF = 0.3      # MediaPipe pose presence confidence
POSE_TRACKING_CONF = 0.3      # MediaPipe pose tracking confidence

# =============================================================================
# Model Settings
# =============================================================================
YOLO_MODEL = "yolov8x.pt"           # YOLOv8 model file
POSE_MODEL = "pose_landmarker.task"  # MediaPipe pose landmarker model

# Model download URL (if not exists)
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

# =============================================================================
# Zoom Settings (Center Tracking)
# =============================================================================
ZOOM_ENABLED = True           # True: enable center zoom, False: disable
ZOOM_SCALE = 1.2              # Fixed zoom magnification
ZOOM_SMOOTHING = 0.08         # EMA smoothing factor (0.0-1.0, lower = smoother)
ZOOM_PADDING = 1.0            # Padding multiplier around bounding box
