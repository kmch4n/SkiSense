"""Generate a side-by-side pose-model comparison from a ski image or video.

Examples:
    python compare.py video.mp4 --frames 30,90
    python compare.py skier.jpg --models sam3d,yolo11,yolo26,rtmw3d
"""

import argparse

from src.skisense.pose_comparison import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="File in input/, or an absolute path")
    parser.add_argument("--models", default="sam3d,yolo11,yolo26,rtmw3d",
                        help="Comma-separated model names")
    parser.add_argument("--frames", default="",
                        help="Comma-separated zero-based video frame numbers; default: 3 evenly spaced frames")
    args = parser.parse_args()
    frames = [int(item.strip()) for item in args.frames.split(",") if item.strip()]
    result = compare(args.input_file,
                     [item.strip() for item in args.models.split(",")],
                     indices=frames)
    print(f"Comparison saved to: {result}")


if __name__ == "__main__":
    main()
