"""SkiSense configuration settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

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
# Load environment variables from .env file
# =============================================================================
env_path = Path(PROJECT_ROOT) / '.env'
load_dotenv(dotenv_path=env_path, verbose=False)

# =============================================================================
# Helper Functions for Type Conversion
# =============================================================================
def _get_bool(key: str, default: bool) -> bool:
    """Get boolean from environment with fallback to default."""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')

def _get_int(key: str, default: int, min_val: int = None, max_val: int = None) -> int:
    """Get integer from environment with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        result = int(value)
        if min_val is not None and result < min_val:
            return default
        if max_val is not None and result > max_val:
            return default
        return result
    except ValueError:
        return default

def _get_float(key: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """Get float from environment with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        result = float(value)
        if min_val is not None and result < min_val:
            return default
        if max_val is not None and result > max_val:
            return default
        return result
    except ValueError:
        return default

def _get_str(key: str, default: str, valid_options: list = None) -> str:
    """Get string from environment with validation."""
    value = os.getenv(key)
    if value is None:
        return default
    if valid_options and value not in valid_options:
        return default
    return value

# =============================================================================
# Processing Settings
# =============================================================================
DEBUG = _get_bool('SKISENSE_DEBUG', False)           # True: show all logs, False: suppress logs
SHOW_GUI = _get_bool('SKISENSE_SHOW_GUI', False)     # True: show preview window, False: headless mode
# Device preference for PyTorch inference
# - "auto": Auto-detect (priority: MPS > CUDA > CPU)
# - "mps": Force Apple Silicon GPU (macOS only)
# - "cuda": Force NVIDIA GPU (CUDA required)
# - "cpu": Force CPU (no GPU acceleration)
DEVICE_PREFERENCE = _get_str('SKISENSE_DEVICE', 'auto',
                              valid_options=['auto', 'mps', 'cuda', 'cpu'])  # "auto" | "mps" | "cuda" | "cpu"

# =============================================================================
# Detection & Tracking Settings
# =============================================================================
YOLO_CONFIDENCE = _get_float('SKISENSE_YOLO_CONFIDENCE', 0.3, min_val=0.0, max_val=1.0)         # YOLO detection confidence threshold (0.0-1.0)
DEEPSORT_MAX_AGE = _get_int('SKISENSE_DEEPSORT_MAX_AGE', 30, min_val=1)         # Frames to keep track without detection
DEEPSORT_N_INIT = _get_int('SKISENSE_DEEPSORT_N_INIT', 1, min_val=1)           # Frames required to confirm new track
POSE_PRESENCE_CONF = _get_float('SKISENSE_POSE_PRESENCE_CONF', 0.3, min_val=0.0, max_val=1.0)      # MediaPipe pose presence confidence
POSE_TRACKING_CONF = _get_float('SKISENSE_POSE_TRACKING_CONF', 0.3, min_val=0.0, max_val=1.0)      # MediaPipe pose tracking confidence
POSE_DETECTION_CONFIDENCE = _get_float('SKISENSE_POSE_DETECTION_CONFIDENCE', 0.5, min_val=0.0, max_val=1.0)  # MediaPipe pose detection confidence

# =============================================================================
# Model Settings
# =============================================================================
YOLO_MODEL = "yolov8x.pt"           # YOLOv8 model file
POSE_MODEL = "pose_landmarker.task"  # MediaPipe pose landmarker model

# MediaPipe Pose model type selection
POSE_MODEL_TYPE = _get_str('SKISENSE_POSE_MODEL_TYPE', 'full',
                            valid_options=['lite', 'full', 'heavy'])  # lite: fast/low accuracy, full: balanced, heavy: slow/high accuracy

# Model download URL (dynamically generated based on POSE_MODEL_TYPE)
POSE_MODEL_URL = (
    f"https://storage.googleapis.com/mediapipe-models/"
    f"pose_landmarker/pose_landmarker_{POSE_MODEL_TYPE}/float16/latest/"
    f"pose_landmarker_{POSE_MODEL_TYPE}.task"
)

# =============================================================================
# Zoom Settings (Center Tracking)
# =============================================================================
ZOOM_ENABLED = _get_bool('SKISENSE_ZOOM_ENABLED', True)           # True: enable center zoom, False: disable
ZOOM_SCALE = _get_float('SKISENSE_ZOOM_SCALE', 1.2, min_val=1.0, max_val=5.0)              # Fixed zoom magnification
ZOOM_SMOOTHING = _get_float('SKISENSE_ZOOM_SMOOTHING', 0.08, min_val=0.0, max_val=1.0)         # EMA smoothing factor (0.0-1.0, lower = smoother)
ZOOM_PADDING = _get_float('SKISENSE_ZOOM_PADDING', 1.0, min_val=0.5, max_val=5.0)            # Padding multiplier around bounding box
