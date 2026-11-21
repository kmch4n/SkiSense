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

- Respond to the repository owner in Japanese unless the request specifies otherwise.
- Write code, identifiers, comments, docstrings, commit messages, PR titles and PR bodies in English.
- All files under `docs/spec/` must be **English-only**. Do not introduce translated `*_ja.md` specification duplicates.
- `README.md` and `README_ja.md` remain English/Japanese user introductions. `docs/pose_backends.md` and `docs/pose_backends_ja.md` remain operational guides in their respective languages. `docs/project_details_ja.md` is historical background, not the normative engineering contract.
- Maintain links from translated operational/background material to the canonical English specifications; do not maintain conflicting copied rules.

## Working Mode

- Treat discussion, review, diagnosis, and design consultation as
  read-only unless the user explicitly asks for implementation.
- Do not run `git add`, `git commit`, or `git push` unless the user
  explicitly asks.
- Prefer surgical edits. Do not rewrite large files unless the structure
  is genuinely being improved.

## Specification-First Development — Mandatory

`docs/spec/README.md` is the canonical map; `docs/spec/change-protocol.md` is the **accepted, binding workflow**. Read these files and the subject-specific specs **before modifying behavior**. The specification suite documents the implementation baseline at `main@f95d68c`; this anchor identifies the initial audit, not a promise that every future commit will keep identical behavior.

**Required order: read current source/specs → update the English specs to define intended change → define acceptance tests → change code → verify → update traceability → PR review.** Do not edit Python runtime code first and backfill the specification afterward. The specification change is the first logical stage of work, even when both ship in one PR.

- State current implementation (**observed as-is**) separately from agreed target behavior (**accepted to-be**) and actual evidence (**verified**). A known bug must not be relabelled as intended behavior to justify the existing code.
- For any feature, bug fix or compatibility-affecting refactor, update the affected `docs/spec/` contracts **first**. Include mode matrix, inputs/outputs, coordinate frames/units, selection behavior, errors, missing data, performance, compatibility and model constraints as appropriate.
- Define regression/acceptance tests **before** implementing the new behavior. Record which checks require CPU/mock, MPS/CUDA, gated checkpoints or real ski footage; do not claim unperformed runs.
- Update `docs/spec/verification-matrix.md` and all affected README/help/settings documentation in the **same PR**. Identify unimplemented acceptance criteria and track them explicitly.
- Changes to score thresholds, scoring semantics, backend defaults, coordinate frames, units or comparison `results.json` require versioning/compatibility analysis; never claim cross-backend score comparability without validation.
- Use `.github/pull_request_template.md`. A behavior-changing PR without a preceding spec change and a verification plan is **not ready for merge**.
- For documentation-only corrections, inspect the relevant source and update observed claims directly; do not fabricate a model test or new behavior.
- Preserve concise, purpose-driven commits matching repository history (e.g. `[📝] Document ...`); avoid one commit per file. PR titles/descriptions and all canonical specs must be English.

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
- Specification: `docs/spec/` (canonical English contract suite)
- CI and PR template: `.github/`
- Private notes: `notes/` (git-ignored; do not link from tracked docs)
- Vendored SAM 3D Body: `external/sam-3d-body/` (git-ignored)

## Architecture

### Processing Pipeline

Normal video processing follows three main stages (see
`docs/spec/video-processing.md` for mode-specific behavior):

1. **Detection & Tracking**: YOLOv8x detects persons per frame. Deep SORT
   runs in a separate pre-scan for `longest` with zoom enabled, or per
   frame for active `largest` regardless of zoom. With zoom disabled,
   `largest` still incurs tracking work but its tracks are unused for
   target selection; `longest` performs no tracking. Fast mode uses
   full-frame backend inference, with only the conditional pre-scan.
2. **Pose Estimation**: Each detected person's bbox is forwarded to the
   selected backend. SAM 3D Body returns MHR keypoints (3D +
   2D-projected) plus the full body mesh; YOLO26/YOLO11-Pose return 2D
   COCO-17 keypoints. `analyze_ski_pose()` reads the backend's
   `PoseTopology` and evaluates knee/hip/ankle in 3D when `is_3d` is
   set, otherwise in image space; shoulder tilt is always 2D.
3. **Rendering**: When enabled, ZoomTracker applies zoom. The selected
   landmarks/bbox are drawn in output coordinates and the info panel is
   overlaid. Normal video with zoom disabled omits the target skeleton
   and bbox (Issue #15).

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

## Known Issues and Evidence

The current as-is behavior and pending validation are described in
`docs/spec/video-processing.md`, `docs/spec/pose-scoring.md`,
`docs/spec/output-and-failure-semantics.md`, and
`docs/spec/verification-matrix.md`.

- `--high` only logs intent; interpolation is not implemented ([#4](https://github.com/kmch4n/SkiSense/issues/4)).
- Raw shoulder tilt may wrap to ~180° and be misgraded ([#12](https://github.com/kmch4n/SkiSense/issues/12)).
- Normal video with zoom disabled omits target bbox/skeleton and changes selection ([#15](https://github.com/kmch4n/SkiSense/issues/15)).
- `--fast --pose-backend sam3d` has no internal person detector and uses an upstream whole-image fallback ([#16](https://github.com/kmch4n/SkiSense/issues/16)).
- Video output does not verify `VideoWriter.isOpened()` ([#17](https://github.com/kmch4n/SkiSense/issues/17)).
- Full scoring and behavioral integration tests remain insufficient, even though former tracking Issue [#18](https://github.com/kmch4n/SkiSense/issues/18) was closed with PR #20. Verify coverage, not issue state.

No source inspection, static CI job or mocked unit test proves skiing-form accuracy, SAM3D Windows/CUDA installation compatibility or real-video performance.

## Validation Expectations

- Run `python -m unittest discover -s tests -v` after changing backend
  selection, comparison, config, or docs. CI runs only the stdlib static
  checks listed under Commands.
- If you change pipeline behavior, run the smallest validation that
  matches the change. Check for import/runtime errors, config/path
  regressions, output file generation, and overlay placement.
- If full video inference is too heavy to run, say so explicitly and
  limit claims to what you actually checked.
