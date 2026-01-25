#!/usr/bin/env python
"""SkiSense - Ski Pose Analysis Tool

Usage:
    python run.py [video_file]

Examples:
    python run.py                    # Process input/video.mp4
    python run.py my_ski_video.mp4   # Process input/my_ski_video.mp4

Configuration:
    Edit src/skisense/config.py to change settings.
"""
import sys
from src.skisense import main

if __name__ == "__main__":
    video_file = sys.argv[1] if len(sys.argv) > 1 else None
    main(video_file)
