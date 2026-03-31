#!/usr/bin/env python
"""SkiSense - Ski Pose Analysis Tool

Usage:
    python run.py [options] [video_file]

Options:
    --high    Enable high precision mode (frame interpolation)

Examples:
    python run.py                          # Normal mode
    python run.py --high                   # High precision mode
    python run.py my_ski_video.mp4         # Process specific video
    python run.py --high my_ski_video.mp4  # High precision with specific video

Configuration:
    Edit src/skisense/config.py to change settings.
"""
import sys
import argparse
from src.skisense import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='SkiSense - Ski Pose Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('video_file', nargs='?', default=None,
                        help='Video filename in input/ directory (default: video.mp4)')
    parser.add_argument('--high', action='store_true',
                        help='Enable high precision mode with frame interpolation')

    args = parser.parse_args()
    main(video_file=args.video_file, high_precision=args.high)
