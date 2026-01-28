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
- **GPU Acceleration:** Supports NVIDIA CUDA and Apple Silicon MPS

## Installation

### Quick Start (Windows/Linux with NVIDIA GPU)

**Recommended for most users - includes GPU acceleration (1.5-3x faster)**

1. Install CUDA Toolkit 11.8 from [NVIDIA website](https://developer.nvidia.com/cuda-downloads)
2. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```
3. Install dependencies:
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate (Windows)
   .venv\Scripts\activate

   # Activate (Linux)
   source .venv/bin/activate

   # Install PyTorch with CUDA 11.8 support
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # Install other dependencies
   pip install -r requirements.txt
   ```

### macOS with Apple Silicon (M1/M2/M3)

**Automatic GPU acceleration (1.5-3x faster than CPU)**

```bash
# Create virtual environment (Python 3.11+ recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# Install CPU-compatible PyTorch (MPS support is built-in)
pip install torch torchvision
pip install -r requirements.txt
```

SkiSense automatically detects and uses Apple Silicon GPU.

### CPU-Only Installation (No GPU)

For systems without NVIDIA GPU or for testing:

```bash
# Create virtual environment
python -m venv .venv

# Activate
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# Install CPU-only PyTorch
pip install torch torchvision

# Install other dependencies
pip install -r requirements.txt
```

**Note:** PyTorch must be installed separately before `requirements.txt` because the correct version depends on your hardware (CUDA version, macOS, or CPU-only).

### Alternative CUDA Versions

If you have a different CUDA version installed:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then install other dependencies
pip install -r requirements.txt
```

Check your CUDA version with `nvidia-smi` or `nvcc --version`.

## Usage

```bash
# Process default video (input/video.mp4)
python run.py

# Process specific video
python run.py my_ski_video.mp4
```

- Place input videos in `input/` directory
- Output videos are saved in `output/` as timestamped directories containing:
  - `{filename}_pose.mp4` - Annotated video
  - `best_shot.jpg` - Best posture frame

## Configuration

SkiSense can be configured using a `.env` file in the project root. This allows you to customize settings without modifying the code.

### Setup

1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your preferred settings:
   ```bash
   # Enable debug mode and GUI preview
   SKISENSE_DEBUG=true
   SKISENSE_SHOW_GUI=true

   # Adjust detection confidence
   SKISENSE_YOLO_CONFIDENCE=0.4
   ```

3. Run SkiSense (settings from `.env` will be applied):
   ```bash
   python run.py
   ```

### Configuration Options

All configuration options are documented in `.env.example`. Here are some common configurations:

#### Development Mode (Show GUI + Debug Logs)
```bash
SKISENSE_DEBUG=true
SKISENSE_SHOW_GUI=true
```

#### Production Mode (Headless + Minimal Logs)
```bash
SKISENSE_DEBUG=false
SKISENSE_SHOW_GUI=false
```

#### Force CPU Mode (No GPU)
```bash
SKISENSE_DEVICE=cpu
```

#### High Confidence Mode (Fewer False Positives)
```bash
SKISENSE_YOLO_CONFIDENCE=0.5
SKISENSE_POSE_PRESENCE_CONF=0.5
```

### Available Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `SKISENSE_DEBUG` | false | Show detailed logs and warnings |
| `SKISENSE_SHOW_GUI` | false | Show preview window during processing |
| `SKISENSE_DEVICE` | auto | Device preference: auto, mps, cuda, cpu |
| `SKISENSE_YOLO_CONFIDENCE` | 0.3 | Detection confidence (0.0-1.0) |
| `SKISENSE_ZOOM_ENABLED` | true | Enable automatic zoom tracking |
| `SKISENSE_ZOOM_SCALE` | 1.2 | Zoom magnification (1.0-5.0) |

See `.env.example` for complete list and detailed descriptions.

### Device Selection

SkiSense automatically selects the best available device:
1. **Apple Silicon GPU (MPS)** - macOS M1/M2/M3
2. **NVIDIA GPU (CUDA)** - Windows/Linux with CUDA toolkit
3. **CPU** - Fallback for all platforms

Check the output logs to see which device is being used:
```
[timestamp] Using device: MPS
[timestamp] GPU acceleration: enabled (mps)
[timestamp] ========================================
[timestamp] Component configuration:
[timestamp]   - YOLO: MPS GPU (half=False)
[timestamp]   - Deep SORT: CPU (MPS not supported)
[timestamp]   - MediaPipe: CPU
[timestamp] ========================================
```

## Project Structure

```
SkiSense/
├── src/skisense/      # Source code
│   ├── main.py         # Main processing
│   ├── pose_analyzer.py # Pose analysis
│   ├── config.py       # Configuration
│   └── zoom_tracker.py # Zoom tracking
├── models/             # Model files (auto-downloaded)
├── input/              # Input videos
├── output/             # Output videos (timestamped)
├── run.py              # Entry point
└── requirements.txt
```

## Requirements

- Python 3.10+
- **Recommended for real-time processing:**
  - NVIDIA GPU with CUDA 11.8+ (Windows/Linux)
  - Apple Silicon M1/M2/M3 (macOS)

## Performance Comparison

| Hardware | Processing Speed | Notes |
|----------|-----------------|-------|
| NVIDIA GPU (CUDA) | 🚀 Fastest | All components GPU-accelerated |
| Apple Silicon (MPS) | ⚡ Fast | YOLO GPU-accelerated, 1.5-3x faster than CPU |
| CPU | 🐢 Slow | All components CPU-only |

## Tech Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Person detection
- [MediaPipe](https://developers.google.com/mediapipe) - Pose estimation
- [Deep SORT](https://github.com/nwojke/deep_sort) - Object tracking
- [OpenCV](https://opencv.org/) - Video processing
- [PyTorch](https://pytorch.org/) - Deep learning backend

## Features

- ✅ GPU acceleration (NVIDIA CUDA / Apple Silicon MPS)
- ✅ Automatic device detection
- ✅ Real-time pose evaluation
- ✅ Multi-person tracking
- ✅ Best shot extraction
- ✅ Automatic zoom and centering
- ✅ Timestamped output files
- ✅ Headless mode (no GUI required)
