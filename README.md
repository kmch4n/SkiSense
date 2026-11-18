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
- **Pose estimation** — YOLO26-Pose (default), YOLO11-Pose, or SAM 3D Body,
  switchable per run via
  `--pose-backend`. See [Pose backends](#pose-backends)
- **Joint-angle evaluation** — knee / hip in 2D with YOLO, or knee / hip /
  ankle in 3D with SAM 3D Body; shoulder tilt in 2D
- **Overall score** — 0–100 over the angles that can be measured
- **Multi-person tracking** — Deep SORT keeps a stable ID across frames
- **Auto zoom & centering** — keeps the skier's torso centered
- **Best-shot extraction** — saves the highest-scoring frame

## Quick start

Install PyTorch for your device first (CUDA example below; CPU and MPS
installation instructions are on the PyTorch site):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Only needed for the SAM 3D Body backend:
pip install -r requirements-sam3d.txt
# Also clone Meta's sam-3d-body source and authenticate gated weights;
# see docs/sam3d_setup.md for the complete process.
```

Run:

```bash
python run.py video.mp4                     # process a video (input/video.mp4 by default)
python run.py --fast video.mp4              # full-frame pose; longest+zoom may pre-scan with YOLO/Deep SORT
python run.py --pose-backend sam3d video.mp4    # optional 3D pose engine (CUDA required)
python run.py --pose-backend yolo11 video.mp4   # earlier 2D pose engine
python run.py skier.jpg --image             # process a single image
```

Output is written to `output/YYYYMMDD_HHMMSS/` (`video_pose.mp4`,
`best_shot.jpg`, and a copy of the input).

To compare pose models on the same skier frames, see
[Pose model comparison](docs/pose_comparison.md). For example:

```bash
python compare.py video.mp4 --frames 30,90
```

The comparison writes side-by-side images and `results.json` under
`output/comparison_YYYYMMDD_HHMMSS/`. It does not change the active
processing backend.

> `sam3d` uses gated Hugging Face weights: request access to
> `facebook/sam-3d-body-dinov3` and run `hf auth login` before its first
> run. The default `yolo26` downloads public weights and supports CPU.

## Pose backends

Three pose engines are available for regular runs. The default `yolo26`
uses 2D COCO-17 landmarks and needs no gated model access. `yolo11` remains
available for earlier 2D results; `sam3d` adds 3D body and foot landmarks.

Per run, on the command line:

```bash
python run.py --pose-backend yolo26    # 2D engine (default)
python run.py --pose-backend yolo11    # earlier 2D engine
python run.py --pose-backend sam3d     # 3D engine (CUDA required)
```

Or as a persistent default in `.env`:

```bash
SKISENSE_POSE_BACKEND=yolo26   # change to yolo11 or sam3d when needed
```

The CLI flag wins over `.env`. The active engine is printed on the
`- Pose:` line of the startup banner.

- **SAM 3D Body** — estimated camera-space 3D MHR joints + body mesh, including ankle angles. CUDA and gated weights are required; viewpoint and occlusion errors remain possible. Latency depends on the machine and model.
- **YOLO26-Pose** — default 2D COCO-17 engine; public weights and CPU support.
- **YOLO11-Pose** — 2D COCO-17 keypoints; fast and CPU-capable, but the
  ankle angle is `N/A` (no foot landmark).

YOLO26 also has no foot landmark, so its ankle angle is `N/A`.
Scores across the 2D and 3D engines are **not** comparable: their angles
are measured differently, and YOLO averages 5 items where `sam3d` averages
7. Full comparison, settings, and trade-offs:
[`docs/pose_backends.md`](docs/pose_backends.md).

## Architecture

The standard video mode follows three steps (the fast mode differs, and
`longest` target selection may first scan the whole video):

1. **Detection & tracking** — YOLOv8x detects persons; Deep SORT can assign
   persistent track IDs for target selection.
2. **Pose estimation** — the selected backend returns keypoints (3D + 2D
   for SAM 3D Body, 2D for YOLO11/YOLO26); `pose_analyzer` evaluates joint
   angles.
3. **Rendering** — when enabled, `ZoomTracker` applies immediate bbox-based
   dynamic crop/zoom (the smoothing configuration is not currently applied).
   The selected bbox and skeleton use `transform_point_to_zoom()`, then the
   information panel is overlaid. Zoom-off overlay limitations: [#15](https://github.com/kmch4n/SkiSense/issues/15).

Key modules: `config.py`, `pose_topology.py`, `backends/`,
`pose_analyzer.py`, `zoom_tracker.py`, `main.py`, `image_processor.py`.

## Specifications and development process

The proposed [specification baseline](docs/spec/README.md) distinguishes current behavior, intended behavior, and verified results. For optional SAM 3D Body source/weights, see the [setup guide](docs/sam3d_setup.md). Known limitations are tracked in GitHub Issues.

## More (日本語)

- Project background, design rationale, scoring logic, and lessons
  learned (Japanese): [README_ja.md](README_ja.md) /
  [`docs/project_details_ja.md`](docs/project_details_ja.md)

## License

MIT License. See [LICENSE](LICENSE).
