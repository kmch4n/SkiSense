# Pose-model comparison workflow

**Status:** Observed as-is. **Code:** `compare.py`, `src/skisense/pose_comparison.py`, `compare_rtmw3d.py`. **Related operations:** [../pose_comparison.md](../pose_comparison.md).

## 1. Purpose and non-goals

The comparison command visually places model predictions on the **same source frames**, with one reference person detector bbox for association. It is an exploratory visualization and metadata export, **not** a normalized accuracy evaluation, speed benchmark or fair cross-model score competition. No aggregate SkiSense score is computed or saved. This workflow does **not** change the default normal-run backend.

Available columns: `sam3d`, `yolo11`, `yolo26`, `rtmw3d`; default includes all four. The list must be nonempty, contain unique known names and is processed **in user-supplied order**.

## 2. Sampling and target detection

`compare(input_file,models,indices=None,output_dir=None)` resolves relative files under `input/` and accepts absolute paths. JPEG, PNG and WebP suffixes are read as one image with sample index `0`. Other suffixes are opened as video by OpenCV.

For video, the user may supply **zero-based** frame indices. Without them, three interior indices are chosen as `int(total_frames*i/(count+1))` for `i=1..count`, clamped to the last index, deduplicated and sorted. Seeking and reading is done **before loading inference backends**; invalid indices or unreadable frames raise `ValueError`. No samples is an error.

After device resolution, one shared YOLOv8x detector processes each sample and chooses the bbox with maximum area. If a sampled frame lacks a person, that frame is skipped with a diagnostic; if all samples lack a person, `compare` raises `ValueError`. Different sampled frames may refer to **different people**, since the comparator does not perform Deep SORT or cross-frame subject tracking. Always inspect the original cyan rectangle.

## 3. Model evaluation lifecycle

1. Create `output/comparison_YYYYMMDD_HHMMSS/` (unless an explicit `output_dir` was passed programmatically).
2. Record input basename, device, selected models, checkpoint identifiers, installed package versions, selection rule and interpretation caveat.
3. For each selected frame, record a reference-box original column.
4. For each model **sequentially**, construct one backend, run `backend.estimate(frame, shared_bbox)` over all chosen frames, and capture per-frame elapsed seconds. This uses ROI estimation for YOLO/SAM3D; RTMW3D runs its wholebody model on the full frame and selects by spatial proximity.
5. On a successful entry, serialize pose, draw visible skeleton on a **copy** of original frame, and save `frame_NNNNNN_<model>.jpg`. If no pose is returned, save an unannotated per-model JPEG with `status=no_pose`.
6. A per-frame exception records `status=error` and the exception text, but processing other frames/models continues. A model-construction exception marks that model `error` for all frames; the others continue.
7. Close each backend in `finally` and clear CUDA cache when CUDA is active.
8. Build `frame_NNNNNN_comparison.jpg` with a labelled original column and one column per model, using a fixed 560-pixel square display canvas; write `results.json` and a simple `index.html` linking comparison sheets.

The fixed square display crop is created from the target bbox with side `min(3*max(w,h),max(frame_width,frame_height))`, clamped inside the image and resized to fit 560×560. This **visual crop is independent of normal-mode dynamic zoom**. Column timing includes the per-frame estimate call only; cold-model initialization occurs separately outside that timer, but first-estimate GPU warmup and runtime variability are still possible.

## 4. Machine-readable `results.json` schema

The root object contains:

```json
{
  "input": "ski.mp4",
  "device": "cpu",
  "package_versions": {"torch": "...", "ultralytics": "..."},
  "models": ["yolo11", "yolo26"],
  "model_checkpoints": {
    "sam3d": "facebook/sam-3d-body-dinov3",
    "yolo11": "yolo11x-pose.pt",
    "yolo26": "yolo26x-pose.pt",
    "rtmw3d": "rtmlib Wholebody3d default checkpoint"
  },
  "selection": "largest YOLOv8x person bbox per frame",
  "interpretation": "...",
  "frames": {
    "30": {
      "target_bbox": [12, 34, 200, 360],
      "models": {
        "yolo11": {
          "status": "ok",
          "seconds": 0.123,
          "topology": "coco17",
          "bbox": [0, 0, 400, 600],
          "landmarks": [{"x": 123.0, "y": 234.0, "visibility": 0.9}],
          "visibility_semantics": "model_keypoint_score",
          "overlay": "frame_000030_yolo11.jpg"
        }
      },
      "comparison": "frame_000030_comparison.jpg"
    }
  }
}
```

This is a **schema illustration**, not an actual result or validated timing. `package_versions` contains only packages that can be resolved at runtime; the listed packages are `torch`, `ultralytics`, `rtmlib`, `onnxruntime-gpu`. `model_checkpoints` includes all four known model identifiers even if fewer are requested.

The `frames` keys are stringified zero-based indices. Per-model outcomes:

| Status | Fields |
| --- | --- |
| `ok` | `status`, elapsed `seconds`, serialized `topology`, `bbox`, `landmarks`, `visibility_semantics`, `overlay`; optionally `joints_3d` and `coordinate_system` |
| `no_pose` | `status`, `seconds`, `overlay` |
| `error` | `status`, exception `error`; no per-model overlay filename is guaranteed |

Each serialized landmark contains `x` and `y` in **absolute original-frame pixels** (nullable for nonfinite input) and numeric `visibility` (nonfinite becomes 0.0). When `joints_3d` is a two-dimensional array with exactly three columns, it is emitted as a list of 3D points with nonfinite components replaced by `null`. Its `coordinate_system` defaults to `mhr_camera_space` unless the entry supplies another label. RTMW3D explicitly labels unverified model coordinates. **These coordinate labels do not constitute calibration, alignment or proof of metric comparability**.

The `visibility_semantics` value is `placeholder_one_for_used_joints` for SAM3D or `model_keypoint_score` for the others. JSON is written with `allow_nan=False`, UTF-8 and indentation; errors in unrelated serialization fields can still abort the export. The CLI does not write per-model angle evaluations or an overall score.

## 5. Interpretation and change control

A visual overlap on a handful of frames cannot establish pose-estimation accuracy, optimal model choice, throughput or physical joint-angle validity. Repeat with diverse views, occlusions, turn phases and skier sizes; pin model checkpoints, source versions and environment. Changes to frame sampling, target association, `results.json` keys, missing-data representation, crop conventions or timing definitions require spec-first agreement and compatibility tests. Do not compare RTMW3D raw 3D coordinates with MHR or assign a skiing score to the adapter without a validated transformation.
