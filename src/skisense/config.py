"""SkiSense configuration settings."""
import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# Path Settings
# =============================================================================
# Resolve the repository root from this file's location so paths stay stable
# regardless of the caller's current working directory.
# config.py lives at <repo>/src/skisense/config.py, so parents[2] is the repo.
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
INPUT_DIR = os.path.join(PROJECT_ROOT, "input")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# =============================================================================
# Load environment variables from .env file
# =============================================================================
env_path = Path(PROJECT_ROOT) / ".env"
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
TARGET_SELECTION_MODE = _get_str('SKISENSE_TARGET_SELECTION_MODE', 'longest',
                                  valid_options=['longest', 'largest'])  # "longest" | "largest"
POSE_VISIBILITY_THRESHOLD = _get_float('SKISENSE_POSE_VISIBILITY_THRESHOLD', 0.5, min_val=0.0, max_val=1.0)  # Minimum landmark visibility for upper-body joints
POSE_VISIBILITY_THRESHOLD_LEGS = _get_float('SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS', 0.3, min_val=0.0, max_val=1.0)  # Looser threshold for leg joints (often occluded in ski poses)
ROI_PADDING_RATIO = _get_float('SKISENSE_ROI_PADDING', 0.3, min_val=0.0, max_val=1.0)  # ROI bbox expansion ratio for better pose accuracy

# Preprocessing / TTA toggles (opt-in, validated as data-dependent on ski footage)
CLAHE_ENABLED = _get_bool('SKISENSE_CLAHE_ENABLED', False)  # Apply CLAHE to ROI before pose estimation
FLIP_TTA_ENABLED = _get_bool('SKISENSE_FLIP_TTA_ENABLED', False)  # Run pose estimation on horizontal flip too and pick best-visibility landmarks (doubles inference cost)

# =============================================================================
# Model Settings
# =============================================================================
YOLO_MODEL = "yolov8x.pt"           # YOLOv8 model file (person detection)

# YOLO11-Pose model filename
YOLO_POSE_MODEL = _get_str('SKISENSE_YOLO_POSE_MODEL', 'yolo11x-pose.pt')
YOLO_POSE_CONFIDENCE = _get_float('SKISENSE_YOLO_POSE_CONFIDENCE', 0.25,
                                   min_val=0.0, max_val=1.0)

# =============================================================================
# Zoom Settings (Center Tracking)
# =============================================================================
ZOOM_ENABLED = _get_bool('SKISENSE_ZOOM_ENABLED', True)           # True: enable center zoom, False: disable
ZOOM_SCALE = _get_float('SKISENSE_ZOOM_SCALE', 1.2, min_val=1.0, max_val=5.0)              # Fallback zoom magnification
ZOOM_SMOOTHING = _get_float('SKISENSE_ZOOM_SMOOTHING', 0.08, min_val=0.0, max_val=1.0)         # EMA smoothing factor (0.0-1.0, lower = smoother)
ZOOM_PADDING = _get_float('SKISENSE_ZOOM_PADDING', 1.0, min_val=0.5, max_val=5.0)            # Padding multiplier around bounding box
ZOOM_TARGET_AREA_RATIO = _get_float('SKISENSE_ZOOM_TARGET_AREA_RATIO', 0.35, min_val=0.05, max_val=0.95)  # Target skier bbox area ratio in the output frame
ZOOM_MAX_SCALE = _get_float('SKISENSE_ZOOM_MAX_SCALE', 5.0, min_val=1.0, max_val=20.0)  # Maximum dynamic zoom magnification
