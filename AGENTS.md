# AGENTS.md

Guidance for AI coding agents (Codex, Claude Code, and others) working in
this repository. `.claude/CLAUDE.md` imports this file, so keep shared
instructions here.

## Overview

SkiSense is a computer-vision tool that detects and evaluates skiing
posture from video or still images. The current scope is posture
visualization and quantitative feedback, not full automatic judging.

The runtime stack is YOLOv8x for person detection, Deep SORT for
tracking, and a selectable pose backend. The backend is chosen via
`run.py --pose-backend` (per run) or `SKISENSE_POSE_BACKEND` (persistent
default):

- **YOLO26-Pose** (`yolo26`, default): 2D COCO-17, CPU/MPS/CUDA.
- **SAM 3D Body** (`sam3d`): 3D MHR-21, CUDA required.
- **YOLO11-Pose** (`yolo11`): 2D COCO-17, kept to reproduce earlier output.

See `docs/pose_backends.md` and `docs/pose_comparison.md`.

## Language

- Respond to the user in Japanese.
- Keep code, identifiers, comments, docstrings, and commit messages in
  English.
- Preserve each document's existing language:
  - `README.md` is English; `README_ja.md` is Japanese.
  - `docs/project_details_ja.md` holds the full Japanese narrative
    (background, design, architecture, scoring logic, lessons learned,
    roadmap).
  - `docs/pose_backends.md` is English (canonical);
    `docs/pose_backends_ja.md` is its Japanese version.
  - `docs/sam3d_setup.md` and `docs/pose_comparison.md` are English.
  - `docs/spec/README.md` is English; the other `docs/spec/*_ja.md`
    files are Japanese.
- When editing docs, keep each EN/JA pair in sync.

## Working Mode

- Treat discussion, review, diagnosis, and design consultation as
  read-only unless the user explicitly asks for implementation.
- Do not run `git add`, `git commit`, or `git push` unless the user
  explicitly asks.
- Prefer surgical edits. Do not rewrite large files unless the structure
  is genuinely being improved.

## Specification-Driven Changes

`docs/spec/` is the shared change contract for agents and developers.
Read `docs/spec/README.md` and `docs/spec/change_protocol_ja.md` before
changing behavior.

- Keep as-is behavior, intended behavior, and verified evidence separate.
  Code reading alone does not make a model's accuracy or real-video
  behavior "verified".
- Update the relevant `docs/spec/` files and
  `docs/spec/verification_matrix_ja.md` in the same PR as any behavioral
  change.
- State compatibility and comparability impacts when changing scoring,
  coordinate frames, units, model defaults, or emitted JSON.
- Fill in `.github/pull_request_template.md`, including the tests that
  were **not** run.

## Code Style

- Use 4 spaces for indentation. Do not use tabs.
- Keep type hints when adding new Python code.
- Follow the existing style: small helpers, direct orchestration, and
  English docstrings.
- No linter/formatter runs in CI, but follow Black + Ruff + mypy
  conventions.

## Commands

```bash
# Run with default video (input/video.mp4)
python run.py

# Run with a specific video in input/
python run.py my_ski_video.mp4

# Run with a still image
python run.py skier.jpg --image

# High precision mode placeholder (not implemented, Issue #4)
python run.py --high

# Fast mode: skip per-frame YOLOv8 + Deep SORT and run the backend on the
# full frame. Under sam3d there is no person detection (Issue #16).
python run.py --fast

# Main-target selection for zoom (default: longest)
python run.py --target-mode largest

# Switch pose engine for a single run (overrides SKISENSE_POSE_BACKEND)
python run.py --pose-backend sam3d
python run.py --pose-backend yolo11 skier.jpg --image

# Compare pose models on identical frames and detection boxes
python compare.py video.mp4 --frames 30,90
python compare.py skier.jpg --models sam3d,yolo11,yolo26,rtmw3d

# Unit tests (unittest, no extra framework)
python -m unittest discover -s tests -v

# CI static checks (.github/workflows/static-checks.yml, stdlib only)
python -m compileall -q run.py compare.py src tests
python -m unittest discover -s tests -p 'test_docs_links.py' -v

# Install dependencies (PyTorch must be installed first, separately)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA
pip install -r requirements.txt

# Install SAM 3D Body dependencies (see docs/sam3d_setup.md)
# detectron2 / MoGe are intentionally skipped; the backend uses
# YOLOv8x for detection and the default FOV.
pip install -r requirements-sam3d.txt
```

The optional RTMW3D comparison column also needs `rtmlib` and ONNX
Runtime; see `docs/pose_comparison.md`.

## Repo Map

- Entry CLIs: `run.py` (analysis), `compare.py` (model comparison)
- Package exports: `src/skisense/__init__.py`
- Video pipeline: `src/skisense/main.py`
- Single-image pipeline: `src/skisense/image_processor.py`
- Pose scoring: `src/skisense/pose_analyzer.py`
- Joint topologies: `src/skisense/pose_topology.py`
- Pose backends: `src/skisense/backends/`
- Model comparison: `src/skisense/pose_comparison.py`,
  `src/skisense/compare_rtmw3d.py`
- Zoom / centering: `src/skisense/zoom_tracker.py`
- Environment config: `src/skisense/config.py`, `.env.example`
- Tests: `tests/`
- Input / output / weights: `input/`, `output/`, `models/` (git-ignored)
- Specification: `docs/spec/`
- CI and PR template: `.github/`
- Private notes: `notes/` (git-ignored; do not link from tracked docs)
- Vendored SAM 3D Body: `external/sam-3d-body/` (git-ignored)

## Architecture

### Processing Pipeline

The video processing follows a strict 3-step pipeline per frame:

1. **Detection & Tracking**: YOLOv8x detects persons; Deep SORT assigns
   persistent track IDs.
2. **Pose Estimation**: Each detected person's bbox is forwarded to the
   selected backend. SAM 3D Body returns MHR keypoints (3D +
   2D-projected) plus the full body mesh; YOLO26/YOLO11-Pose return 2D
   COCO-17 keypoints. `analyze_ski_pose()` reads the backend's
   `PoseTopology` and evaluates knee/hip/ankle in 3D when `is_3d` is
   set, otherwise in image space; shoulder tilt is always 2D.
3. **Rendering**: ZoomTracker applies zoom; landmarks/bboxes are
   drawn in zoom-transformed coordinates using the 2D keypoints; the
   info panel is overlaid.

All coordinate transforms must go through `transform_point_to_zoom()`
when zoom is enabled. If overlays drift, inspect this function first.
SAM 3D Body's 2D keypoints are normalised to the padded ROI before being
attached to landmarks so existing draw helpers keep working unchanged.

### Module Responsibilities

- **config.py**: Loads settings from `.env` with `python-dotenv`. All
  env vars are prefixed with `SKISENSE_`. `PROJECT_ROOT` is resolved
  from the file location, so paths do not depend on the working
  directory. `AVAILABLE_POSE_BACKENDS` is the single source of truth for
  backend names.
- **pose_topology.py**: Defines `MHR_BODY` (SAM 3D Body) and `COCO_17`
  (YOLO26/YOLO11-Pose). `MHR_BODY` covers body + feet + wrists with
  `has_foot=True` and `is_3d=True`; `COCO_17` is 2D with
  `has_foot=False`.
- **pose_analyzer.py**: Pure computation module.
  `analyze_ski_pose(..., topology=...)` reads joint indices and `is_3d`
  from the given `PoseTopology`. For a 3D topology it uses camera-space
  coords via `get_landmark_point_3d` for knee/hip/ankle angles; for a 2D
  topology it falls back to image-plane coords.
- **backends/**: Pose backend abstraction. `get_backend(backend=...)`
  resolves the engine in the order explicit argument (`run.py
  --pose-backend`) → `SKISENSE_POSE_BACKEND` → `yolo26`; valid names
  live in `AVAILABLE_BACKENDS`. `yolo11_backend.py` wraps Ultralytics
  YOLO-Pose for both `yolo26` and `yolo11` (different checkpoints);
  `sam3d_backend.py` wraps Meta's `SAM3DBodyEstimator`. All normalise
  outputs into the shared `PoseBackend` contract, expose `display_name`,
  and implement `estimate_full_frame` for `--fast`.
- **pose_comparison.py**: Runs several backends on the same frames and
  detection boxes and writes a side-by-side comparison.
  **compare_rtmw3d.py** is a comparison-only RTMW3D adapter; its 3D
  frame is not validated, so it returns no score.
- **_logging.py**: Central `SuppressStderr` context manager and
  Ultralytics log-level overrides. Imported eagerly by `main.py` and
  every backend so native library noise is suppressed uniformly when
  `SKISENSE_DEBUG=false`.
- **zoom_tracker.py**: Stateful tracker that selects the main target and
  applies immediate bbox-based zoom and centering with a 30-frame
  detection timeout. The legacy scale/smoothing/padding arguments are
  accepted but unused; no EMA is applied.
- **main.py**: Video orchestration: device resolution, model
  initialization, frame loop, video I/O. It is orchestration-heavy;
  read it before changing pipeline behavior.
- **image_processor.py**: Single-image analysis flow; reuses helpers
  from `main.py`.

### Device Handling

Device resolution happens in `resolve_device()` with priority
MPS > CUDA > CPU when set to `auto`.

- YOLO models use `half=True` only on CUDA.
- **SAM 3D Body requires CUDA**. The backend raises at construction if
  given `mps` or `cpu`, and the message points at another backend.
- YOLO26/YOLO11-Pose run on CPU / MPS / CUDA.
- Deep SORT GPU acceleration is CUDA-only.

### Configuration

See `.env.example` for the full surface. Important settings:

- `SKISENSE_DEBUG`: enables verbose logging and disables stderr
  suppression
- `SKISENSE_SHOW_GUI`: enables the OpenCV preview window
- `SKISENSE_DEVICE`: auto/mps/cuda/cpu (effectively cuda for SAM 3D Body)
- `SKISENSE_POSE_BACKEND`: `yolo26` (default), `sam3d`, or `yolo11`.
  Overridden per run by `run.py --pose-backend`.
- `SKISENSE_YOLO26_POSE_MODEL` / `SKISENSE_YOLO_POSE_MODEL`: checkpoint
  filenames for YOLO26 / YOLO11-Pose, cached under `models/`
- `SKISENSE_SAM3D_HF_REPO`: `facebook/sam-3d-body-dinov3` (default) or
  `facebook/sam-3d-body-vith` (smaller, faster)
- `SKISENSE_SAM3D_USE_HAND_REFINE`: enable the hand decoder pass;
  default `false` since SkiSense does not score hands
- `SKISENSE_YOLO_CONFIDENCE`: YOLOv8x person-detection threshold
  (default `0.3`). `SKISENSE_YOLO_POSE_CONFIDENCE` (default `0.25`) is
  the separate person/instance threshold passed to YOLO-Pose as
  `conf=`; neither filters individual joints.
- `SKISENSE_POSE_VISIBILITY_THRESHOLD`: minimum keypoint confidence for
  upper-body joints (default `0.5`). SAM 3D Body does not expose
  per-joint confidence and fills `1.0`, so this is inert under `sam3d`.
- `SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS`: looser threshold for hip /
  knee / ankle (default `0.3`). Same `sam3d` caveat.
- `SKISENSE_ROI_PADDING`: ROI bbox expansion ratio before pose
  estimation (default `0.3`).
- `SKISENSE_ZOOM_TARGET_AREA_RATIO` / `SKISENSE_ZOOM_MAX_SCALE` drive the
  dynamic zoom. `SKISENSE_ZOOM_SCALE`, `SKISENSE_ZOOM_SMOOTHING`, and
  `SKISENSE_ZOOM_PADDING` are accepted for compatibility but unused.

### Model Weights

- YOLOv8x person-detection weights are expected at `models/yolov8x.pt`.
- Ultralytics downloads YOLO-Pose weights on first run and caches them
  under `models/`.
- SAM 3D Body weights are gated on HuggingFace:
  1. Visit https://huggingface.co/facebook/sam-3d-body-dinov3 and click
     **Request Access**.
  2. Once approved, generate a read token and run `hf auth login`.
  3. The backend downloads the checkpoint through `huggingface_hub` on
     first construction; later runs reuse the local cache.
- The vendored Meta repository lives at `external/sam-3d-body/` (clone
  with `git clone https://github.com/facebookresearch/sam-3d-body.git
  external/sam-3d-body`). The backend adds it to `sys.path` at
  construction time.

### Output

Video mode writes to `output/YYYYMMDD_HHMMSS/`:

- `video_pose.mp4` (`video_pose_fast.mp4` with `--fast`): annotated video
- `best_shot.jpg`: highest-scoring frame (only saved when a score is positive)
- `video.mp4`: copy of the input video

Image mode writes `image_pose.jpg` and `image.jpg` to the same layout.
`compare.py` writes to `output/comparison_YYYYMMDD_HHMMSS/`.

## Known Issues

`docs/spec/behavior_ja.md` is the authoritative as-is baseline. Notable
issues:

- `--high` only logs its intent; frame interpolation is not implemented
  (#4).
- With `SKISENSE_ZOOM_ENABLED=false`, target bbox/skeleton drawing is
  missing and target selection differs (#15).
- `--fast --pose-backend sam3d` has no person detector and uses a single
  whole-image bbox (#16).
- Video output does not check `VideoWriter.isOpened()` (#17).
- `process_video()` in `main.py` still contains some duplicated inline
  steps even though several helpers have been extracted.

## Validation Expectations

- Run `python -m unittest discover -s tests -v` after changing backend
  selection, comparison, config, or docs. CI runs only the stdlib static
  checks listed under Commands.
- If you change pipeline behavior, run the smallest validation that
  matches the change. Check for import/runtime errors, config/path
  regressions, output file generation, and overlay placement.
- If full video inference is too heavy to run, say so explicitly and
  limit claims to what you actually checked.
