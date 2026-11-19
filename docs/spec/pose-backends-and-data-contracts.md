# Pose backends and data contracts

**Status:** Observed as-is. **Code:** `src/skisense/backends/`, `pose_topology.py`, `pose_analyzer.py`, `preprocessing.py`.

## 1. Supported engines and selection

| Name | Constructor | Checkpoint default | Topology | Coordinate basis | Hardware |
| --- | --- | --- | --- | --- | --- |
| `yolo26` (default) | `Yolo11Backend` with alternate model filename/display name | `yolo26x-pose.pt` | `COCO_17`, 17 joints | 2D, padded ROI-normalized | CPU, MPS, CUDA |
| `yolo11` | `Yolo11Backend` | `yolo11x-pose.pt` | `COCO_17`, 17 joints | 2D, padded ROI-normalized | CPU, MPS, CUDA |
| `sam3d` | `Sam3dBackend` | `facebook/sam-3d-body-dinov3` | `MHR_BODY`, 63 entries | 2D ROI-normalized + inferred 3D camera coordinates | CUDA **only** |
| `rtmw3d` | `Rtmw3dComparisonBackend` | rtmlib default | `WHOLEBODY_133` | Projected 2D full-frame; raw 3D unverified | Comparison only |

`AVAILABLE_POSE_BACKENDS` from `config.py` drives `backends.AVAILABLE_BACKENDS`, the CLI choice list and `get_backend` resolution. RTMW3D is **not** in this list and must not be introduced as a normal analysis backend without specifying a validated coordinate/score contract.

`get_backend(running_mode, device, use_gpu, device_str, backend)` constructs the concrete model lazily; `running_mode` exists for interface compatibility and is not used in the dispatch decision. The named YOLO26 engine reuses the class named `Yolo11Backend` and changes `model_filename` and `display_name`.

## 2. Abstract and operational interfaces

`PoseBackend` formally requires:

```python
estimate(frame, bbox, timestamp_ms=None) -> (landmark_entry | None, analysis | None)
close() -> None  # base implementation is a no-op
```

Both regular concrete classes also implement `estimate_full_frame(frame) -> list[(entry, analysis)]`. **This method is not declared abstract in `PoseBackend`**, so `process_video` checks `hasattr` in fast mode. The signatures are Python interfaces, not strict runtime-enforced schemas. Both backends are stateless per frame; timestamp is accepted but unused.

### Landmark entry (normal ROI mode)

```python
{
  "landmarks": list[object],                # .x/.y/.visibility; backend-specific extras
  "bbox": (px, py, pw, ph),                # padded ROI, original-frame pixels
  "detection_bbox": (x, y, w, h),         # original detector bbox
  "torso_center": (cx, cy) | None,         # ROI-relative pixels
  "shoulder_center": (cx, cy),             # ROI-relative pixels
  "hip_center": (cx, cy),                  # ROI-relative pixels
  "topology": PoseTopology,                # COCO_17 or MHR_BODY
}
```

The backend's `estimate` returns `(None, None)` when it cannot produce a usable person/analysis. `analysis` has the schema in [pose-scoring.md](pose-scoring.md). The entry's `bbox` is **not** the same as `detection_bbox` in ROI mode. In full-frame YOLO mode `bbox=(0,0,frame_width,frame_height)` and `detection_bbox` is a per-person box; SAM3D's fast path uses the model-provided box as both. Consumers must use the correct key for scoring, target association or rendering.

Landmarks use normalized **padded-ROI** `x/y`, not absolute frame pixels. Compute frame pixels as `bbox.x + landmark.x*bbox.w` and similarly for y. Visibility is a YOLO per-keypoint model score or a SAM3D **synthetic placeholder**, not a calibrated cross-backend probability. SAM3D landmark shims additionally expose `x3d/y3d/z3d` and a `z` depth alias; YOLO shims have `z=0`. Full-frame YOLO entries can also carry `confidence` (detected-person confidence).

### SAM3D-only entry fields

Normal ROI inference additionally attaches `joints_3d`, `mesh_vertices`, `mhr_params` (`body_pose`, `shape`, `scale`) and `camera_params` (`focal_length`, `translation`) from the upstream output. Fast inference attaches `joints_3d` and `mesh_vertices`, **not** the parameter dictionaries. These extras are kept in memory: `run.py` does not serialize a mesh or 3D angle file. Compare export serializes available `joints_3d` only.

## 3. Topology contract

`PoseTopology` has `name`, `num_landmarks`, semantic `indices`, drawing `connections`, `leg_indices`, `lr_swap_pairs`, `has_foot`, and `is_3d`.

| Semantic joint | COCO-17 index | MHR_BODY index |
| --- | ---: | ---: |
| Left / right shoulder | 5 / 6 | 5 / 6 |
| Left / right hip | 11 / 12 | 9 / 10 |
| Left / right knee | 13 / 14 | 11 / 12 |
| Left / right ankle | 15 / 16 | 13 / 14 |
| Left / right big toe | Not available | 15 / 18 |
| Left / right elbow | 7 / 8 (render edges) | 7 / 8 |
| Left / right wrist | 9 / 10 (render edges) | 62 / 41 |

`COCO_17`: `name='coco17'`, `num_landmarks=17`, `has_foot=False`, `is_3d=False`. `MHR_BODY`: `name='mhr_body'`, `num_landmarks=63`, `has_foot=True`, `is_3d=True`. MHR includes head/torso, arms and feet; indices between 21 and 62 include hand-finger slots largely unused by the application. The UI nickname **"MHR-21" does not mean the returned array contains 21 entries**.

`lr_swap_pairs` permits YOLO flip-TTA coordinate restoration. `leg_indices` selects the lower visibility threshold. The drawing connections are backend-specific and should not be replaced with a shared hard-coded skeleton.

## 4. YOLO11/YOLO26 ROI inference

`Yolo11Backend.estimate(frame, bbox)`:

1. Reject source bboxes with nonpositive width/height; compute padding `int(width*ROI_PADDING_RATIO)` and `int(height*ROI_PADDING_RATIO)`, clip padded crop to image boundaries; reject zero-sized or empty ROI.
2. Optionally apply CLAHE **only on that ROI**.
3. Invoke Ultralytics pose with `conf=YOLO_POSE_CONFIDENCE`, `device=cpu/mps/0` and `half=True` only on CUDA.
4. In a multi-person ROI, associate the requested original box with the pose box by greatest intersection-over-union; when no overlap exists, use closest box center; when no requested target is supplied, use greatest model confidence.
5. Convert `keypoints.xyn` and `keypoints.conf` to 17 normalized landmark shims; if keypoint confidence is missing, set all visibility to 1.0.
6. If flip TTA is enabled, run one more inference on a horizontally flipped ROI, swap left/right indices, invert x, and independently retain the more visible landmark at each index. This **does not** enforce a physically coherent set of joints.
7. Analyze 2D skeleton and return the padded ROI entry with original `detection_bbox`.

`estimate_full_frame` instead performs one YOLO-Pose call over the entire frame and emits all analysable people, each with normalized full-frame coordinates and a detector bbox. **Full-frame YOLO path does not apply ROI CLAHE or flip TTA**. Missing or degenerate pose boxes can fall back to a full-frame bbox.

Weight handling: first look for `models/<pose checkpoint>`. If absent, move a same-named checkpoint found in the process working directory or temporarily change `os.getcwd()` to `models/` to trigger Ultralytics download. This global working-directory mutation is not thread-safe. Closing the backend drops its model reference; it does not guarantee immediate GPU memory reclamation.

## 5. SAM 3D Body inference

`Sam3dBackend` rejects a resolved device other than `cuda`. It inserts `external/sam-3d-body/` into `sys.path` when present, imports `sam_3d_body` directly, loads gated Hugging Face weights with `load_sam_3d_body_hf`, and creates `SAM3DBodyEstimator` with `human_detector=None`, `human_segmentor=None`, `fov_estimator=None`. Hand refinement selects inference type `full`; default `body` skips it. External setup details: [../sam3d_setup.md](../sam3d_setup.md).

In ROI mode, SkiSense still passes the **entire RGB frame**, plus a **single padded `xyxy` box**, to the estimator. It converts the first result's projected 2D joints to padded-ROI normalized `x/y` and first 63 MHR joints to camera-space 3D; unsupported/unused hand slots receive visibility 0.0, used joints receive visibility 1.0. It requires 2D and 3D arrays with at least 63 rows. It then scores with the MHR topology.

In full-frame mode, no box or detector is supplied. This must **not** be described as multi-person detection: the current upstream fallback can operate on the whole image as one box ([#16](https://github.com/kmch4n/SkiSense/issues/16)). The code processes whatever upstream `bbox` items are returned. Changes to this behavior require a new detection/association specification and GPU integration evidence.

Model-based 3D coordinates are inferred from a single RGB image, not directly measured. SAM3D exposes no per-joint visibility probability through this integration: the fixed 1.0 visibility of used points means both configured thresholds have no practical filtering effect for those joints, including when they are occluded.

## 6. RTMW3D comparison adapter

`Rtmw3dComparisonBackend` uses `rtmlib.Wholebody3d(backend='onnxruntime', device=...)`; it expects arrays shaped `(N,133,2)` plus `(N,133)` confidence. It finds a person near the target box based on visible torso points (falling back to 23 body points), rejects distant matches, emits projected full-frame `WHOLEBODY_133` 2D landmarks, optionally records raw model 3D coordinates, and returns `(entry, None)`. **No skiing score** is produced: the 3D coordinate frame is unverified and cannot be treated as equivalent to MHR.

## 7. Compatibility and acceptance criteria for extensions

A new backend must specify topology counts and names, ordering, left/right swaps, coordinate origin/units/frame, bbox semantics, visibility meaning, failure behavior, full-frame capability and resource cleanup. It must provide deterministic normalization tests, model-independent analyzer tests and separate actual-model validation. A backend switching change must address score comparability, weight licensing, device fallback, install documentation and any serializable JSON format impacts.
