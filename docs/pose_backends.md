# Pose Estimation Backends

**English** | [日本語](pose_backends_ja.md)

SkiSense supports two pose estimation engines. `yolo11` is the
pre-migration engine and stays fully supported, so switching back to the
older 2D behaviour is always a one-flag operation.

**Per run — CLI flag** (highest priority):

```bash
python run.py --pose-backend yolo11 video.mp4     # legacy 2D engine
python run.py --pose-backend sam3d  video.mp4     # 3D engine
python run.py --pose-backend yolo11 skier.jpg --image
```

**Persistent default — `.env`:**

```bash
SKISENSE_POSE_BACKEND=sam3d    # default: SAM 3D Body (3D, high accuracy)
SKISENSE_POSE_BACKEND=yolo11   # YOLO11-Pose (2D, light and fast)
```

Resolution order is `--pose-backend` → `SKISENSE_POSE_BACKEND` → `sam3d`.
The selected engine is echoed on the `- Pose:` line of the startup banner,
so you can confirm which one is active before a long run.

The person detection (YOLOv8x), tracking (Deep SORT), zoom, and scoring
pipeline are shared; only the pose estimation step swaps out. Each backend
declares its own topology (`pose_topology.py`), and `analyze_ski_pose`
plus the drawing helpers follow it automatically. `--fast`, `--high`, and
`--target-mode` work with either engine.

## Comparison

| Aspect | SAM 3D Body | YOLO11-Pose |
|---|---|---|
| Backend name | `sam3d` (default) | `yolo11` (legacy fallback) |
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

Ultralytics' 2D pose model, and the engine SkiSense used before the SAM
3D Body migration. Estimates COCO-17 keypoints per ROI. It is retained
deliberately — not deprecated — so results from the older pipeline stay
reproducible and CUDA-less machines remain usable.

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
- **Reproducing pre-migration output** → `yolo11`. It is the same engine
  and the same COCO-17 topology as before the SAM 3D Body migration.

## Scores are not comparable across backends

Switching backends changes the measured numbers, not just their
precision, so do not compare a `sam3d` score against a `yolo11` score:

- Knee / hip angles come from 3D camera-space vectors under `sam3d` and
  from projected image-plane coordinates under `yolo11`. The same posture
  yields different degrees at oblique viewpoints.
- The overall score is the mean over the joints that could be measured.
  `yolo11` cannot measure the two ankle angles, so its mean is taken over
  5 items instead of 7.

Fix the backend for a given comparison — for example when reviewing a
season's clips as a set.

## Known limitations

- `yolo11` cannot produce an ankle angle (COCO-17 has no foot landmark);
  the info panel shows `N/A`.
- `sam3d` requires CUDA. Selecting `sam3d` on an environment where
  `SKISENSE_DEVICE` does not resolve to CUDA raises a `RuntimeError` at
  backend construction, naming the `--pose-backend yolo11` fallback.
- `sam3d` does not expose per-joint confidence, so the backend fills
  visibility with `1.0` for every joint it uses. As a consequence
  `SKISENSE_POSE_VISIBILITY_THRESHOLD` and
  `SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS` have no effect under `sam3d`;
  occluded joints are still scored from the model's estimate. Both
  settings remain effective under `yolo11`.
