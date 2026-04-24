#!/usr/bin/env python
"""SkiSense - Ski Pose Analysis Tool

Usage:
    python run.py [options] [input_file]

Options:
    --high     Enable high precision mode (frame interpolation, video only)
    --image    Process a single image instead of a video

Examples:
    python run.py                          # Video: default input/video.mp4
    python run.py --high                   # Video: high precision mode
    python run.py my_ski_video.mp4         # Video: specific file
    python run.py skier.jpg --image        # Image: input/skier.jpg

Configuration:
    Edit src/skisense/config.py to change settings.
"""
import argparse

from src.skisense import process_image, process_video

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SkiSense - Ski Pose Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input_file", nargs="?", default=None,
        help="Video or image filename in input/ directory",
    )
    parser.add_argument(
        "--high", action="store_true",
        help="Enable high precision mode with frame interpolation (video only)",
    )
    parser.add_argument(
        "--image", action="store_true",
        help="Process a single image instead of a video",
    )

    args = parser.parse_args()

    if args.image:
        if args.high:
            parser.error("--image cannot be combined with --high")
        process_image(image_file=args.input_file)
    else:
        process_video(video_file=args.input_file, high_precision=args.high)
