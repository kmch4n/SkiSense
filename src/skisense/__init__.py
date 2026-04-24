"""SkiSense - Ski Pose Analysis Tool

A tool for analyzing skiing posture from video or still images using
computer vision and pose estimation.
"""
from .image_processor import process_image
from .main import main, process_video
from .pose_analyzer import COLORS, analyze_ski_pose

__version__ = "1.0.0"
__all__ = [
    "main",
    "process_video",
    "process_image",
    "analyze_ski_pose",
    "COLORS",
]
