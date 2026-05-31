# Pose Estimation Backends

**English** | [日本語](pose_backends_ja.md)

SkiSense supports two pose estimation engines, selected via
`SKISENSE_POSE_BACKEND` in `.env`:

```bash
SKISENSE_POSE_BACKEND=sam3d    # default: SAM 3D Body (3D, high accuracy)
SKISENSE_POSE_BACKEND=yolo11   # YOLO11-Pose (2D, light and fast)
```

The person detection (YOLOv8x), tracking (Deep SORT), zoom, and scoring
pipeline are shared; only the pose estimation step swaps out. Each backend
declares its own topology (`pose_topology.py`), and `analyze_ski_pose`
plus the drawing helpers follow it automatically.

## Comparison

| Aspect | SAM 3D Body | YOLO11-Pose |
|---|---|---|
| `SKISENSE_POSE_BACKEND` | `sam3d` (default) | `yolo11` |
| Topology | MHR-21 (body + feet + wrists) | COCO-17 |
| Dimensions | 3D (camera space) + 2D projection | 2D image space only |
| Joint angles | Knee / hip / ankle in 3D (view-invariant) | Knee / hip in 2D |
| Ankle angle | Yes (foot landmarks available) | N/A (no foot in COCO-17) |
| Shoulder tilt | 2D image plane | 2D image plane |
| Device | **CUDA required** | CPU / MPS / CUDA |
| Speed (rough) | ~1–2 s/frame (RTX 4060) | a few ms/ROI |
| VRAM (rough) | ~3.4 GB (FP16, body-only) | ~2–3 GB |
| Weights | HuggingFace gated (request + `hf auth login`) | Ultralytics auto-download |
| License | SAM License (Apache-2.0-like, commercial OK) | AGPL-3.0 (Ultralytics) |
| Mesh output | Yes (`pred_vertices`) | No |

## SAM 3D Body (`sam3d`)

Meta's single-image 3D human mesh recovery model, released 2025-11-19. It
uses the MHR (Momentum Human Rig) parametric model, yielding true,
view-invariant joint angles. SkiSense consumes the first 63 of the MHR70
keypoints (body, feet, wrists).

- **Strengths**: Knee/hip/ankle angles stay undistorted under oblique
  viewpoints; robust to inside-leg occlusion. Evaluates ankle flex
  (knee→ankle→toe) and also produces a body mesh.
- **Weaknesses**: CUDA required. A few seconds per frame, so video is an
  offline batch. Gated weights require a one-time HuggingFace access
  request.
- **Settings**:
  - `SKISENSE_SAM3D_HF_REPO`: `facebook/sam-3d-body-dinov3` (default, 840M)
    or `facebook/sam-3d-body-vith` (631M, lighter / lower VRAM)
  - `SKISENSE_SAM3D_USE_HAND_REFINE`: hand decoder. Off by default since
    SkiSense does not score hands (roughly halves VRAM and latency).
- **Setup**: see [`../notes/sam3d_setup.md`](../notes/sam3d_setup.md)
  (detectron2 / MoGe are not needed).

## YOLO11-Pose (`yolo11`)

Ultralytics' 2D pose model. Estimates COCO-17 keypoints per ROI.

- **Strengths**: Runs on CPU. A few ms per frame, suitable for quick
  checks, near-real-time use, or machines without CUDA. Weights
  auto-download.
- **Weaknesses**: Being 2D, joint angles suffer projection distortion at
  oblique viewpoints. COCO-17 has no foot landmark, so the **ankle angle
  is N/A** (excluded from the score).
- **Settings**:
  - `SKISENSE_YOLO_POSE_MODEL`: `yolo11x-pose.pt` (default); also n/s/m/l/x
  - `SKISENSE_YOLO_POSE_CONFIDENCE`: keypoint confidence threshold (default 0.25)
  - `SKISENSE_CLAHE_ENABLED`: apply CLAHE to the ROI (default false)
  - `SKISENSE_FLIP_TTA_ENABLED`: horizontal-flip TTA (default false, 2x cost)

## Choosing a backend

- **Precise form analysis / production visualization** → `sam3d`. The
  view-invariant 3D angles, ankle evaluation, and mesh pay off when CUDA
  is available and processing time is acceptable.
- **Quick checks / no-CUDA machines / first-pass batch screening** → `yolo11`.

## Known limitations

- `yolo11` cannot produce an ankle angle (COCO-17 has no foot landmark);
  the info panel shows `N/A`.
- `sam3d` requires CUDA. Selecting `sam3d` on an environment where
  `SKISENSE_DEVICE` does not resolve to CUDA raises a `RuntimeError` at
  backend construction. Switch to `yolo11` or run on a CUDA machine.
