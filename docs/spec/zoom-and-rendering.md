# Zoom, target continuity and rendering

**Status:** Observed as-is. **Code:** `src/skisense/zoom_tracker.py`, drawing helpers in `main.py`, `pose_topology.py`.

## 1. Responsibilities and coordinate spaces

The render path has three distinct coordinate systems:

1. **ROI-local pixel coordinates**: `landmark.x*bbox.width` and `landmark.y*bbox.height`, relative to padded `entry["bbox"]`.
2. **Original source-frame pixels**: ROI-local points plus that bbox's `x/y` offset.
3. **Output-frame normalized coordinates**: `((source_x-crop_x1)/crop_width, (source_y-crop_y1)/crop_height)`, returned by `transform_point_to_zoom` if the point is inside crop.

The drawing helpers convert normalized output coordinates to output pixels by multiplying by current output `frame.width/height`. In the identity case with `zoom_info=None`, `transform_point_to_zoom` instead returns **absolute frame pixels**; the current call sites normally pass identity `zoom_info` rather than `None`. Do not change this implicit distinction without updating every caller and tests.

## 2. Target selection state

`ZoomTracker` holds `main_track_id`, `last_valid_target = (center_x,center_y,scale)`, `frames_since_detection`, and `smooth_center_x/y/scale`. Despite the historical names, **no exponential smoothing** is performed; each current valid bbox immediately sets the new center and scale.

`select_main_target(tracks,rects)` ignores `rects` for selection, filters confirmed Deep SORT tracks, retains the previous `main_track_id` when available, otherwise picks the confirmed track with the largest bbox area and saves its ID. `process_frame` uses this method; the main pipeline usually calls `process_bbox` after handling selection separately.

The `longest` pre-scan plan is implemented in `main.py`, **not** in `ZoomTracker`. See [video-processing.md](video-processing.md). Target selection is currently coupled incorrectly to `ZOOM_ENABLED` ([#15](https://github.com/kmch4n/SkiSense/issues/15)).

## 3. Dynamic scale and missing-target timeout

Given a detected bbox `(x,y,w,h)`, frame size `W,H`, target area ratio `r`, and max scale `s_max`:

```text
bbox_area   = max(1.0, float(w*h))
target_area = W*H*r
scale       = clip(sqrt(target_area/bbox_area), 1.0, s_max)
```

The preferred center is the analysis `torso_center` converted to full-frame coordinates when a matched pose exists; otherwise the bbox midpoint. There is no temporal interpolation or velocity model.

When a target bbox is missing, `frames_since_detection` increments. For counts **1 through 29**, `ZoomTracker` reuses `last_valid_target` if set; on count **30** or later, it reverts to frame center and scale 1.0. A new valid detection resets the counter and saved target. This is **frame-count dependent**, not a fixed number of milliseconds. The `reset()` method clears smooth center/scale and the selected track ID, but does not clear `last_valid_target` or `frames_since_detection`; callers needing a pristine tracker must account for this.

Legacy `zoom_scale`, `smoothing` and `padding` constructor/config values are stored but **unused by the active zoom calculation**. Only `target_area_ratio` and `max_scale` control scale. Do not document legacy parameters as functional.

## 4. Crop and output frame

`apply_zoom(frame, center_x, center_y, scale)` computes an axis-aligned crop of size `(W/scale, H/scale)`, centered on the selected point, clamps it fully inside the input frame instead of padding with edge pixels, rounds start down/end up to integer pixel boundaries, and resizes the crop to exactly `(W,H)` with `cv2.INTER_LINEAR`. It returns the resized BGR frame plus the **rounded original-frame crop** `(x1,y1,x2,y2)`. Empty crops fall back to the original frame and the full-frame crop.

`process_bbox` is the normal processing entry for an already selected bbox and a full-frame center. `process_frame` accepts tracks and optionally a **ROI-relative** center; if it selects a track, it offsets that center by bbox position. Those center contracts differ and must not be confused.

## 5. Overlay behavior

- `draw_target_overlay` uses `entry["detection_bbox"]` for its rectangle if a pose entry exists, otherwise the selected target bbox. It renders only the **selected** target.
- `draw_bbox_on_zoomed_frame` computes transformed top-left and bottom-right corners; if **either corner is outside the crop**, the entire rectangle is omitted. Partial-clipping of a rectangle is not implemented.
- `draw_landmarks_on_zoomed_frame` uses `PoseTopology.connections` and visibility thresholds to draw green BGR lines and red BGR dots. Each connection is skipped if either endpoint is unavailable or outside the crop. Individual points outside the crop are skipped; other points may still render.
- `draw_info_panel` places a semitransparent black panel at top-left, scales dimensions/fonts by `frame_width/1280`, renders knees/hips/shoulder/ankles, and places the score below. Metric status controls color: good green, warning yellow, bad red, unavailable white. The panel returns without drawing if `analysis is None`.
- The panel displays **`N/A` from evaluations**, not the sentinel `0.0` stored in `angles`. Score text is green when >=70, yellow for 40..69 and red below 40; this **UI color threshold is separate from the per-angle good/warning/bad thresholds**.
- No timestamp, pose-confidence ribbon, loss-of-track label, turn-phase marker or per-person table is rendered.

With **normal video and zoom disabled**, `main.py` currently never sets target entry/bbox, so it renders the info panel from the first pose but **no bbox or skeleton**. Fast video with zoom disabled still renders its selected target. Still images always use an identity crop and draw the largest detected target. This asymmetry is tracked as [#15](https://github.com/kmch4n/SkiSense/issues/15).

## 6. Acceptance considerations

Any change to crop math must preserve testable correspondence between source pixels, padded ROI, output crop and panel position. Test targets at corners and boundaries, aspect-ratio extremes, scale=1 and scale=max, lost targets over 30 frames, partial bbox clipping, no pose, zoom on/off and frame dimensions. Define whether smooth motion, target switching and overlay persistence are required **before** coding. Never infer that `SKISENSE_ZOOM_SMOOTHING` works from its name.
