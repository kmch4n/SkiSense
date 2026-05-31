# SkiSense

**English** | [日本語](README_ja.md)

A computer-vision tool that estimates a skier's posture from video and
quantitatively evaluates their form via joint angles.

![SkiSense preview](images/skier.gif)
![SkiSense preview](images/skier2.gif)

## Overview

SkiSense detects a skier in a clip, estimates their pose, and scores key
joint angles (knee, hip, ankle, shoulder tilt) against ideal ranges drawn
from competitive *kihon* (technical) skiing. It is a visualization and
quantitative-feedback tool, not an automatic judge. The annotated video,
the best-scoring frame, and per-joint readouts are written to disk.

## Features

- **Person detection** — YOLOv8x
- **Pose estimation** — selectable backend: SAM 3D Body (3D MHR-21,
  default) or YOLO11-Pose (2D COCO-17). See [Pose backends](#pose-backends)
- **Joint-angle evaluation** — knee / hip / ankle in 3D, shoulder tilt in 2D
- **Overall score** — 0–100 over the angles that can be measured
- **Multi-person tracking** — Deep SORT keeps a stable ID across frames
- **Auto zoom & centering** — keeps the skier's torso centered
- **Best-shot extraction** — saves the highest-scoring frame

## Quick start

PyTorch must be installed first, matching your CUDA build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Only needed for the SAM 3D Body backend (see docs/pose_backends.md):
pip install -r requirements-sam3d.txt
```

Run:

```bash
python run.py video.mp4            # process a video (input/video.mp4 by default)
python run.py --fast video.mp4     # skip per-frame detect/track; one pass per frame
python run.py skier.jpg --image    # process a single image
```

Output is written to `output/YYYYMMDD_HHMMSS/` (`video_pose.mp4`,
`best_shot.jpg`, and a copy of the input).

> The default `sam3d` backend uses gated HuggingFace weights: request
> access to `facebook/sam-3d-body-dinov3` and run `hf auth login` once
> before the first run. To avoid this (or to run on CPU), use the
> `yolo11` backend instead.

## Pose backends

The pose engine is selected in `.env`:

```bash
SKISENSE_POSE_BACKEND=sam3d    # default: SAM 3D Body (3D, CUDA required)
SKISENSE_POSE_BACKEND=yolo11   # YOLO11-Pose (2D, runs on CPU/MPS/CUDA)
```

- **SAM 3D Body** — 3D MHR keypoints + body mesh; view-invariant joint
  angles including the ankle. CUDA required; ~1–2 s/frame.
- **YOLO11-Pose** — 2D COCO-17 keypoints; fast and CPU-capable, but the
  ankle angle is `N/A` (no foot landmark).

Full comparison, settings, and trade-offs:
[`docs/pose_backends.md`](docs/pose_backends.md).

## Architecture

Each frame runs a three-step pipeline:

1. **Detection & tracking** — YOLOv8x detects persons; Deep SORT assigns
   persistent track IDs.
2. **Pose estimation** — the selected backend returns keypoints (3D + 2D
   for SAM 3D Body, 2D for YOLO11-Pose); `pose_analyzer` evaluates joint
   angles.
3. **Rendering** — `ZoomTracker` applies smooth zoom, skeleton/bbox are
   drawn through the single `transform_point_to_zoom()` choke point, and
   the info panel is overlaid.

Key modules: `config.py`, `pose_topology.py`, `backends/`,
`pose_analyzer.py`, `zoom_tracker.py`, `main.py`, `image_processor.py`.

## More (日本語)

- Project background, design rationale, scoring logic, and lessons
  learned (Japanese): [README_ja.md](README_ja.md) /
  [`docs/project_details_ja.md`](docs/project_details_ja.md)

## License

MIT License. See [LICENSE](LICENSE).
