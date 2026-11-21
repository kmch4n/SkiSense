# Video processing specification

**Status:** Observed as-is. **Code:** `src/skisense/main.py`, `zoom_tracker.py`, backend implementations. **Entry:** `process_video(video_file=None, high_precision=False, fast_mode=False, target_mode=None, pose_backend_name=None)`.

## 1. Run initialization

1. Resolve default input name `video.mp4`, then target mode from config (default `longest`); invalid programmatic target modes raise `ValueError`.
2. Resolve CPU/MPS/CUDA preference; CUDA enables `torch.backends.cudnn.benchmark`.
3. Construct the chosen pose backend **before opening/validating input**. Backend initialization may load/download weights and can fail for inaccessible SAM3D dependencies or non-CUDA device.
4. Open `input/<video_file>` via OpenCV. An unopened capture is logged and the function returns after closing the pose backend.
5. Read FPS, width, height and total frame count. Check FPS > 0, width > 0, height > 0; failure logs and returns. Total frame count is **not required positive**; FPS is not explicitly checked for finiteness.
6. Create `output/YYYYMMDD_HHMMSS/`, copy input to `video.mp4`, construct a `mp4v` `VideoWriter` for original FPS and dimensions. **The writer's `isOpened()` status is not checked**.
7. When enabled, construct a `ZoomTracker`; configure YOLO device/precision and load a person detector only when needed.
8. When `target_mode=longest` and zoom is enabled, perform a full-video target-selection pre-scan; if it finds no confirmed tracks, switch active mode to `largest`.
9. For normal video with active `largest` (including a failed `longest` pre-scan fallback), initialize Deep SORT for per-frame tracking **regardless of zoom state**. With `longest` and zoom disabled, no tracker is created. Begin the rendering loop.

`high_precision=True` **only logs** a future interpolation feature. The frame iteration, sampling, output FPS and estimation behavior are unchanged ([#4](https://github.com/kmch4n/SkiSense/issues/4)).

## 2. YOLOv8x person detection

`load_yolo_model` loads `models/yolov8x.pt` using Ultralytics YOLO. `run_yolo_detection` requests class ID 0 (person), confidence `SKISENSE_YOLO_CONFIDENCE`, CUDA half precision when applicable, and returns a list of integer `(x, y, w, h)` original-frame bboxes. Width/height <= 0 detections are skipped. Input bbox coordinates are not comprehensively clipped at this stage; ROI backends clamp their padded crop.

Deep SORT receives `([x,y,w,h], 1.0, 'person')` for each bbox, uses MobileNet embedding, configured `max_age`/`n_init` and CUDA embedding only on CUDA. It returns tracks with stable IDs *subject to the tracker model's behavior*, not guaranteed person identity.

## 3. Longest-track pre-scan

`build_longest_track_plan` reopens the entire video with a **separate** capture and fresh tracker, processes every readable frame, records every confirmed valid track bbox by **one-based frame number**, and gathers:

- `frames_seen` (primary descending criterion);
- mean bbox area (secondary descending criterion);
- mean distance from frame center (tertiary ascending criterion).

`choose_longest_track` selects max of `(frames_seen, average_area, -average_center_distance)`; the selected track ID and its frame→bbox map become `TargetTrackPlan`. If no confirmed track exists, the plan is `None` and normal/fast processing falls back to `largest`. IDs from pre-scan are used only through that recorded bbox map; there is **no second-pass track-ID association** to the pre-scan tracker.

This pre-scan runs **only if zoom is enabled**; the default `longest` flag does not have its intended semantics with zoom disabled ([#15](https://github.com/kmch4n/SkiSense/issues/15)). The pre-scan incurs a full separate detection/tracking pass, even in fast mode.

## 4. Normal per-frame algorithm

For each successfully decoded frame (`current_frame` starts at 1):

1. Run YOLOv8x detection. If the active mode is `largest`, update Deep SORT on every frame, even with zoom disabled.
2. Compute `timestamp_ms = int(current_frame * 1000 / fps)`. This is based on the **one-based** frame index, not media presentation timestamps.
3. Call `pose_backend.estimate(frame, detection_bbox, timestamp_ms=...)` once **per detected person**. Retain only non-`None` `(entry, analysis)` pairs.
4. With zoom **enabled**:
   - `longest` with a plan: get that frame's selected-track bbox; otherwise use `zoom_tracker.select_main_target(tracks, rects)`.
   - Match retained poses to the chosen bbox: exact `entry.detection_bbox` equality wins; otherwise highest bbox IoU **strictly greater than 0.05**; no match yields `None` analysis.
   - Use the selected entry's ROI-relative torso center converted to full-frame coordinates, if available; process target bbox through `zoom_tracker.process_bbox`.
5. With zoom **disabled**: use the uncropped frame and identity transform. If poses exist, take the **first pose's analysis** but do **not** set `target_entry` or `target_bbox`. Any Deep SORT tracks computed in active `largest` mode are discarded for target selection/rendering; the tracking cost is still incurred. This causes the current bbox/skeleton overlay omission and inconsistent target selection ([#15](https://github.com/kmch4n/SkiSense/issues/15)).
6. Draw selected target overlay and info panel if an analysis is available; update `latest_pose_analysis` to the **current frame's** analysis or `None`.
7. If this frame has an analysis with score **strictly greater** than the best seen so far, retain a **copy of the annotated frame**.
8. Call `out.write(output_frame)`; optionally show GUI and stop if user presses `q`.

A person can be detected but have no usable pose; in that case its bbox may still drive zoom, while the panel is absent. Score zero can be remembered as a candidate but will **not be saved as best_shot.jpg** by final export.

## 5. Fast per-frame algorithm

`process_fast_frame` calls `pose_backend.estimate_full_frame(frame)`. It does not invoke per-frame YOLOv8x, Deep SORT or the ROI `estimate` method.

- With a pre-scanned `target_bbox`, select the pose whose `detection_bbox` matches by exact equality or IoU > 0.05. If no pose matches, the target bbox may still be zoomed/drawn but no pose score is available.
- Without a pre-scanned bbox, choose the pose with the largest `detection_bbox` area. Ties follow the backend's result order.
- If zoom is enabled, pass the chosen bbox (or pose bbox when no plan bbox exists) and the selected torso center to `ZoomTracker.process_bbox`.
- If zoom is disabled, use identity coordinates **and still draw the selected full-frame pose**. This differs from the normal zoom-off bug.
- Only the selected primary person's bbox/skeleton/panel is drawn, although the backend may return multiple people.

YOLO26/YOLO11 pose models can detect people during full-frame inference. **SAM3D does not have an internal human detector configured**, and calls the upstream estimator without bboxes; do not claim multi-person detection for that path ([#16](https://github.com/kmch4n/SkiSense/issues/16)). The base class does not formally declare `estimate_full_frame`; `process_video` checks for its presence at runtime.

## 6. Exit, best shot and lifecycle

On loop termination or GUI interruption, save `best_shot.jpg` only if a best frame exists **and `best_score > 0`**. The first frame attaining the maximum wins (strict `>` comparison). The file is rendered/cropped exactly as the output video frame, not a clean source frame. Release capture, writer and pose backend; destroy OpenCV windows if GUI enabled. There is no explicit post-export MP4 validation, no measured inference quality criterion, no per-frame CSV/JSON score history and no transactional cleanup on unexpected exceptions.

Output definitions and failure behaviors: [output-and-failure-semantics.md](output-and-failure-semantics.md). Zoom tracking and crop mathematics: [zoom-and-rendering.md](zoom-and-rendering.md).

## 7. Mode-specific acceptance considerations

Every future video behavior change must define behavior for normal vs fast, zoom on vs off, `longest` vs `largest`, no detector bboxes, zero pose results, disappearing target, multi-person collisions, target bbox/pose mismatch, invalid video metadata, missing weights and failed writer creation. Declare whether frame numbering, track identity, score history, GUI and best-shot tie handling change. Add mock/CPU tests first and retain separate real-footage acceptance criteria.
