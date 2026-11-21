# CLI and configuration contract

**Status:** Observed as-is. **Code:** `run.py`, `compare.py`, `src/skisense/config.py`, `src/skisense/backends/__init__.py`, `src/skisense/main.py`.

## 1. `run.py` command surface

```text
python run.py [--high] [--fast] [--target-mode {longest,largest}]
              [--pose-backend {yolo26,sam3d,yolo11}] [--image] [input_file]
```

| Argument | Default / current behavior | Validation and limits |
| --- | --- | --- |
| `input_file` | `video.mp4` for video; `image.jpg` for image, under `input/` | Actual path made by `os.path.join(INPUT_DIR, filename)`; absolute input overrides prefix |
| `--image` | False; calls `process_image` if present | Cannot combine with `--high`, `--fast`, or `--target-mode` |
| `--fast` | False; invokes backend full-frame inference in video mode | Cannot combine with `--high`; may still require pre-scan for longest/zoom-on |
| `--high` | False; logs planned frame interpolation but **does not implement it** | Video-only; Issue [#4](https://github.com/kmch4n/SkiSense/issues/4) |
| `--target-mode` | `None`, resolved to `SKISENSE_TARGET_SELECTION_MODE` (default `longest`) | `longest` or `largest`; semantics vary with zoom/mode ([video-processing](video-processing.md)) |
| `--pose-backend` | `None`, resolved from `SKISENSE_POSE_BACKEND` (default `yolo26`) | Known values are validated by argparse; constructor errors still possible |

Argument combinations forbidden by the CLI are rejected with `argparse.parser.error` (exit code 2). Other execution-time failures are handled inconsistently and may return without a nonzero exit code. `--high` and `--fast` are not equivalent accuracy/performance levels. Models and package imports occur even for requesting CLI help.

Examples:

```bash
python run.py
python run.py ski.mp4 --target-mode longest
python run.py ski.mp4 --fast --pose-backend yolo26
python run.py skier.png --image --pose-backend sam3d
python compare.py ski.mp4 --frames 30,90 --models yolo11,yolo26
```

## 2. `compare.py` command surface

```text
python compare.py input_file [--models sam3d,yolo11,yolo26,rtmw3d] [--frames 30,90]
```

`input_file` is required; relative paths resolve under `input/`, absolute paths are used directly. The default models are `sam3d,yolo11,yolo26,rtmw3d`. Models must be known, unique, and nonempty; failures raise `ValueError`. `--frames` is a comma-separated list of **zero-based** video frame numbers; omitted/empty chooses three evenly spaced interior frames. The CLI converts tokens to integers without friendly validation. For images, frame selection is ignored and index `0` is used. See [model-comparison.md](model-comparison.md).

## 3. Settings precedence and parsing

`config.py` derives `PROJECT_ROOT` from `Path(__file__).resolve().parents[2]`, then calls `load_dotenv` on `<root>/.env`. Existing process environment variables have precedence over `.env` by the default `python-dotenv` behavior. Settings are parsed **once on module import**, not refreshed dynamically.

Backend precedence: explicit `get_backend(backend=...)` / CLI `--pose-backend` (trimmed and lowercased) > parsed `SKISENSE_POSE_BACKEND` > `yolo26`. Empty explicit names fall back to config; invalid explicit names raise `ValueError`. Invalid environment backend values silently fall back to `yolo26`.

Boolean parser accepts `true`, `1`, `yes`, `on` (case-insensitive) as true; every other set string becomes false. Numeric parse failures and out-of-range values silently use defaults. Device and target-mode string options are case-sensitive in `.env`; backend value is lowercased/trimmed. No schema validation error is emitted for invalid environment input.

## 4. Complete supported `SKISENSE_` environment surface

Values and ranges below are the **current code** defaults, not recommended tuning.

| Variable | Default | Accepted range / effect |
| --- | --- | --- |
| `SKISENSE_DEBUG` | `false` | Boolean; verbose Ultralytics logging and disable stderr/warning suppression when true |
| `SKISENSE_SHOW_GUI` | `false` | Boolean; OpenCV preview and `q` interruption in video mode |
| `SKISENSE_DEVICE` | `auto` | `auto`, `mps`, `cuda`, `cpu` |
| `SKISENSE_YOLO_CONFIDENCE` | `0.3` | Float [0,1]; separate YOLOv8x person-detection confidence |
| `SKISENSE_DEEPSORT_MAX_AGE` | `30` | Integer >=1; tracker age in frames |
| `SKISENSE_DEEPSORT_N_INIT` | `1` | Integer >=1; frames required for confirmed track |
| `SKISENSE_TARGET_SELECTION_MODE` | `longest` | `longest`, `largest`; overridden by video CLI |
| `SKISENSE_POSE_VISIBILITY_THRESHOLD` | `0.5` | Float [0,1]; non-leg landmark visibility |
| `SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS` | `0.3` | Float [0,1]; hip/knee/ankle/foot landmark visibility |
| `SKISENSE_ROI_PADDING` | `0.3` | Float [0,1]; bbox expansion on each side for ROI-mode inference |
| `SKISENSE_POSE_BACKEND` | `yolo26` | `yolo26`, `sam3d`, `yolo11` |
| `SKISENSE_SAM3D_HF_REPO` | `facebook/sam-3d-body-dinov3` | Or `facebook/sam-3d-body-vith`; gated weights |
| `SKISENSE_SAM3D_USE_HAND_REFINE` | `false` | Boolean; `full` vs `body` inference type |
| `SKISENSE_YOLO_POSE_MODEL` | `yolo11x-pose.pt` | String filename for YOLO11 pose |
| `SKISENSE_YOLO26_POSE_MODEL` | `yolo26x-pose.pt` | String filename for YOLO26 pose |
| `SKISENSE_YOLO_POSE_CONFIDENCE` | `0.25` | Float [0,1]; Ultralytics pose **person/instance** threshold, not per-joint visibility |
| `SKISENSE_CLAHE_ENABLED` | `false` | Boolean; optional **ROI-only** CLAHE (YOLO pose backend) |
| `SKISENSE_FLIP_TTA_ENABLED` | `false` | Boolean; extra horizontally flipped **ROI-only** YOLO pose pass |
| `SKISENSE_ZOOM_ENABLED` | `true` | Boolean; controls crop/zoom and currently also target-selection behavior |
| `SKISENSE_ZOOM_SCALE` | `1.2` | Float [1,5]; **accepted but unused** legacy setting |
| `SKISENSE_ZOOM_SMOOTHING` | `0.08` | Float [0,1]; **accepted but unused**, no EMA |
| `SKISENSE_ZOOM_PADDING` | `1.0` | Float [0.5,5]; **accepted but unused** legacy setting |
| `SKISENSE_ZOOM_TARGET_AREA_RATIO` | `0.35` | Float [0.05,0.95]; target bbox proportion in output area |
| `SKISENSE_ZOOM_MAX_SCALE` | `5.0` | Float [1,20]; upper bound on dynamic zoom |

`YOLO_MODEL = "yolov8x.pt"` is a fixed Python constant for person detection, **not** an environment variable. `MODEL_DIR`, `INPUT_DIR`, and `OUTPUT_DIR` derive from the repository root; they are not configurable by `.env`.

## 5. Device resolution

`resolve_device(preference)` chooses MPS > CUDA > CPU for `auto` (tries a small MPS tensor allocation). Forced MPS/CUDA preferences that are unavailable log a warning and **fall back to CPU**, not a hard failure. Forced CPU stays on CPU. YOLO uses half precision on CUDA only; Deep SORT's embedder GPU flag is CUDA-only. `Sam3dBackend` rejects initialization unless the resolved device is exactly `cuda`, so an unavailable forced CUDA device will ultimately error there.

## 6. Operational dependencies

Install a matching PyTorch and torchvision pair separately, then `requirements.txt` (OpenCV, NumPy, Ultralytics, Deep SORT, tqdm, python-dotenv). YOLOv8x detector weights are loaded from `models/yolov8x.pt` (no explicit auto-download setup in `load_yolo_model`); YOLO pose weights are downloaded into `models/` when absent. `sam3d` additionally needs `requirements-sam3d.txt`, a compatible `external/sam-3d-body/` checkout, CUDA and approved Hugging Face access. RTMW3D comparison needs `rtmlib` and ONNX Runtime. Detailed installation caveats: [../sam3d_setup.md](../sam3d_setup.md).

## 7. Compatibility rules

Treat changes to default backend, thresholds, argument semantics, device fallback, environment parsing, input-path handling or supported model names as **public behavior changes**. Specify them first, then update tests, CLI help, `.env.example` and operational README docs. Changing a backend or angle coordinate system changes scoring comparability; see [pose-scoring.md](pose-scoring.md).
