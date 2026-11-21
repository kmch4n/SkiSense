# Single-image processing specification

**Status:** Observed as-is. **Code:** `src/skisense/image_processor.py`; shared helpers from `main.py`. **Entry:** `process_image(image_file=None, pose_backend_name=None)`.

## 1. Input and initialization

1. Default filename is `image.jpg`; resolve it under repository `input/`.
2. Resolve device as for video, enable CUDA cuDNN benchmark when CUDA was selected, construct chosen backend and print device/backend banner.
3. Only then check whether the image exists and whether `cv2.imread` decodes it. On either failure, log and **return**; backend cleanup is not called on these early returns.
4. Create `output/YYYYMMDD_HHMMSS/`; copy the original input **verbatim** to `image.jpg`, regardless of its source extension.
5. Load fixed YOLOv8x person detector and prepare its CUDA/MPS/CPU inference arguments.

CLI `--image` cannot be combined with `--high`, `--fast` or `--target-mode`. It **can** use any regular pose backend, subject to CUDA/gated-weight requirements for SAM3D. The image flow does not use Deep SORT, a pre-scan, target tracking, video timestamps or a dynamic zoom.

## 2. Detection and analysis

- Detect persons using the shared YOLOv8x helper with `SKISENSE_YOLO_CONFIDENCE`.
- Select the single largest bbox by `w*h` (`_pick_largest_bbox`). If no detection exists, keep the input frame with no pose overlay or score.
- If a target exists, draw its detector bbox first; call `pose_backend.estimate(frame, target_bbox)`, where the backend pads the ROI and returns normalized landmarks and an analysis.
- On successful pose estimation, render the topology-specific skeleton and info panel. On failure, retain the detector bbox only.
- Use identity `zoom_info = {'center': (width/2,height/2), 'scale': 1.0, 'crop': (0,0,width,height)}` so rendering helpers convert ROI points into absolute image pixels.

Although `SKISENSE_ZOOM_ENABLED` is a video setting, still-image output does not use the `ZoomTracker`. ROI CLAHE and horizontal-flip TTA, when enabled, apply to YOLO pose backends in ROI mode; SAM3D ignores those settings.

## 3. Output, errors and cleanup

The code calls `cv2.imwrite(image_pose.jpg, output_frame, [cv2.IMWRITE_JPEG_QUALITY,95])`; it does **not check the returned success boolean**. The input image copy is named `image.jpg` even if the original is a PNG or WebP, so its extension may not identify its real format. A collision is possible when multiple runs create the same timestamped output folder in one second. Errors from copying, model loading or drawing are not caught into a structured result. The backend is closed on the normal final path only.

`process_image` returns no structured object or explicit exit status. There is no best-shot selection, image confidence file, exported joint-angle JSON or per-person output list. See [output-and-failure-semantics.md](output-and-failure-semantics.md).

## 4. Requirements for changes

Any change to largest-person selection, image input handling, output naming, confidence threshold, backend choice or error reporting requires an updated specification **before** implementation, including acceptance tests for no detections, corrupt files, non-JPEG inputs, multiple people and failed image writes. Shared drawing/pose changes must also update [pose-backends-and-data-contracts.md](pose-backends-and-data-contracts.md), [pose-scoring.md](pose-scoring.md), or [zoom-and-rendering.md](zoom-and-rendering.md), as applicable.
