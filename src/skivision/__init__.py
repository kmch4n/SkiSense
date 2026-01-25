"""SkiVision - Ski Pose Analysis Tool

A tool for analyzing skiing posture from video using computer vision
and pose estimation.
"""
from .main import main
from .pose_analyzer import analyze_ski_pose, COLORS

__version__ = "1.0.0"
__all__ = ["main", "analyze_ski_pose", "COLORS"]
