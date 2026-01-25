# SkiSense

Ski pose analysis tool using computer vision and deep learning.

![SkiSense preview](images/skier.gif)

## Overview

SkiSense analyzes ski videos to detect and evaluate skiing posture in real-time. It provides:

- **Person Detection:** YOLOv8x for accurate skier detection
- **Pose Estimation:** MediaPipe Pose Landmarker for joint tracking
- **Pose Analysis:** Real-time evaluation of knee, hip, ankle angles and shoulder tilt
- **Score Display:** Overall posture score (0-100)
- **Object Tracking:** Deep SORT for consistent tracking across frames

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Process default video (input/video.mp4)
python run.py

# Process specific video
python run.py my_ski_video.mp4
```

- Place input videos in `input/` directory
- Output videos are saved in `output/` as `{filename}_pose.mp4`

## Configuration

Edit `src/skisense/config.py` to customize settings:

```python
DEBUG = False           # Show debug logs
SHOW_GUI = False        # Show preview window during processing
DEVICE_PREFERENCE = "auto"  # "auto" | "cuda" | "cpu"
```

## Project Structure

```
SkiSense/
├── src/skisense/      # Source code
│   ├── main.py         # Main processing
│   ├── pose_analyzer.py # Pose analysis
│   └── config.py       # Configuration
├── models/             # Model files (auto-downloaded)
├── input/              # Input videos
├── output/             # Output videos
├── run.py              # Entry point
└── requirements.txt
```

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended for real-time processing)

## Tech Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Person detection
- [MediaPipe](https://developers.google.com/mediapipe) - Pose estimation
- [Deep SORT](https://github.com/nwojke/deep_sort) - Object tracking
- [OpenCV](https://opencv.org/) - Video processing
- [PyTorch](https://pytorch.org/) - Deep learning backend
