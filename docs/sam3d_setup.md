# SAM 3D Body setup (optional CUDA backend)

SkiSense's default `yolo26` backend does **not** need these steps. The `sam3d` backend uses Meta's SAM 3D Body source and gated Hugging Face weights. This guide describes the source-inspected integration at `main@f95d68c`; a real CUDA installation has **not** been exercised as part of the documentation audit.

## Prerequisites

- NVIDIA GPU with a compatible CUDA/PyTorch installation. `sam3d` currently rejects CPU and MPS.
- Python environment appropriate for the SAM 3D Body dependencies (the supplemental requirements file describes Python 3.11 and CUDA 12.x).
- Git, Internet access for model weights, a Hugging Face account, and approval for the gated checkpoint.

## Installation

From the SkiSense repository root:

```bash
# Install PyTorch appropriate for your CUDA build (example only):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -r requirements-sam3d.txt

# SkiSense searches for the upstream source at this exact path:
git clone https://github.com/facebookresearch/sam-3d-body.git external/sam-3d-body

# Obtain approval for the selected checkpoint on Hugging Face, then:
hf auth login
```

The `src/skisense/backends/sam3d_backend.py` loader adds `external/sam-3d-body` to Python's import path when that checkout exists. **The upstream repository's `main` branch can change; pin a known-compatible commit when establishing a reproducible environment.** Merely installing the requirements does **not** install the upstream `sam_3d_body` module. If you maintain the package in another environment location, make it importable through an appropriate Python path/package installation.

By default SkiSense requests `facebook/sam-3d-body-dinov3` (gated). It also accepts `facebook/sam-3d-body-vith` via `SKISENSE_SAM3D_HF_REPO`. Consult the model licenses and access requirements before redistribution.

## Run

```bash
python run.py --pose-backend sam3d video.mp4
python run.py --pose-backend sam3d skier.jpg --image
```

To use a CPU/MPS-capable model without gated weights:

```bash
python run.py --pose-backend yolo26 video.mp4
```

## Limitations and verification

- CUDA and pretrained weights are mandatory for current SAM3D execution.
- In the standard mode, YOLOv8x supplies person bboxes. **Do not assume `--fast --pose-backend sam3d` performs separate person detection**: SkiSense disables the estimator's human detector and sends no bbox in that path. Track [Issue #16](https://github.com/kmch4n/SkiSense/issues/16).
- MHR-based 3D angles are inferred from monocular imagery; they are not ground-truth physical measurements.
- No successful CUDA end-to-end run is claimed here. Meta's upstream installation guide lists Detectron2 for its reference workflow; this SkiSense integration deliberately omits that detector by setting `human_detector=None`, but optional dependency behavior and Windows compatibility have not been validated across upstream revisions. Test one input video and one image on the target host; record device, versions, upstream commit, model repository and output artifacts.

For supported 2D/3D model selection, see [pose backend documentation](pose_backends.md).
