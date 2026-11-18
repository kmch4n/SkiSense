# Pose Estimation Backends

**English** | [日本語](pose_backends_ja.md)

SkiSense supports three pose estimation engines in regular runs. `yolo26`
is the default: its public weights run on CPU, MPS, or CUDA without a gated
model account. `yolo11` preserves the earlier 2D pipeline, while `sam3d`
provides 3D body and foot landmarks.

**Per run — CLI flag** (highest priority):

```bash
python run.py --pose-backend yolo26 video.mp4     # default 2D engine
python run.py --pose-backend yolo11 video.mp4     # earlier 2D engine
python run.py --pose-backend sam3d video.mp4      # 3D engine, CUDA required
python run.py --pose-backend yolo26 skier.jpg --image
```

**Persistent default — `.env`:**

```bash
SKISENSE_POSE_BACKEND=yolo26   # change to yolo11 or sam3d when needed
```

Resolution order is `--pose-backend` → `SKISENSE_POSE_BACKEND` → `yolo26`.
The selected engine is echoed on the `- Pose:` line of the startup banner,
so you can confirm which one is active before a long run.

The person detection (YOLOv8x), tracking (Deep SORT), zoom, and scoring
pipeline are shared; only the pose estimation step swaps out. Each backend
declares its own topology (`pose_topology.py`), and `analyze_ski_pose`
plus the drawing helpers follow it automatically. `--fast`, `--high`, and
`--target-mode` work with each regular-run engine.

## Comparison

| Aspect | YOLO26-Pose | YOLO11-Pose | SAM 3D Body |
|---|---|---|---|
| Backend name | `yolo26` (default) | `yolo11` | `sam3d` |
| Topology | COCO-17 | COCO-17 | MHR-21 (body + feet + wrists) |
| Dimensions | 2D image space | 2D image space | 3D (camera space) + 2D projection |
| Joint angles | Knee / hip in 2D | Knee / hip in 2D | Knee / hip / ankle in 3D |
| Ankle angle | N/A | N/A | Yes |
| Shoulder tilt | 2D image plane | 2D image plane | 2D image plane |
| Device | CPU / MPS / CUDA | CPU / MPS / CUDA | **CUDA required** |
| Weights | Ultralytics auto-download | Ultralytics auto-download | Hugging Face gated (request + `hf auth login`) |
| License | Ultralytics AGPL-3.0 | Ultralytics AGPL-3.0 | SAM License |
| Mesh output | No | No | Yes (`pred_vertices`) |

The default reflects simpler setup and the results observed on a small
sample, not a validated ski-pose accuracy ranking. Compare your own footage
with [`compare.py`](pose_comparison.md) before changing a workflow.

## YOLO26-Pose (`yolo26`)

The default regular-run model uses `yolo26x-pose.pt` and COCO-17 body
landmarks. Set `SKISENSE_YOLO26_POSE_MODEL` to choose another YOLO26 pose
checkpoint. Ultralytics downloads the weights on first use. It shares the
existing 2D scoring pipeline with YOLO11, including its view dependence
and missing ankle angle. `ultralytics>=8.4.215` is required.

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

- **Start with no model access request** → `yolo26`.
- **3D body and foot inspection** → `sam3d` when CUDA and gated weights are
  available. Validate its output on the footage being evaluated.
- **Reproducing pre-migration output** → `yolo11`. It is the same engine
  and the same COCO-17 topology as before the SAM 3D Body migration.

## Scores are not comparable across backends

Switching backends changes the measured numbers, not just their
precision, so do not compare a `sam3d` score against a YOLO score:

- Knee / hip angles come from 3D camera-space vectors under `sam3d` and
  from projected image-plane coordinates under `yolo11` or `yolo26`. The same posture
  yields different degrees at oblique viewpoints.
- The overall score is the mean over the joints that could be measured.
  YOLO cannot measure the two ankle angles, so its mean is taken over
  5 items instead of 7.

Fix the backend for a given comparison — for example when reviewing a
season's clips as a set.

## Known limitations

- YOLO11 and YOLO26 cannot produce an ankle angle (COCO-17 has no foot landmark);
  the info panel shows `N/A`.
- `sam3d` requires CUDA. Selecting `sam3d` on an environment where
  `SKISENSE_DEVICE` does not resolve to CUDA raises a `RuntimeError` at
  backend construction, naming the `--pose-backend yolo26` alternative.
- `sam3d` does not expose per-joint confidence, so the backend fills
  visibility with `1.0` for every joint it uses. As a consequence
  `SKISENSE_POSE_VISIBILITY_THRESHOLD` and
  `SKISENSE_POSE_VISIBILITY_THRESHOLD_LEGS` have no effect under `sam3d`;
  occluded joints are still scored from the model's estimate. Both
  settings remain effective under YOLO11 and YOLO26.

RTMW3D is available in [`compare.py`](pose_comparison.md) for visual
comparison. Its 3D coordinate frame and scoring behavior are not yet
validated for the regular pipeline, so it is not a `run.py` backend.
