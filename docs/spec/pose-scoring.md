# Joint-angle measurements and scoring

**Status:** Observed heuristic; **not scientifically validated**. **Code:** `src/skisense/pose_analyzer.py`, `pose_topology.py`, backend shims. **Tracked validation work:** [#19](https://github.com/kmch4n/SkiSense/issues/19).

## 1. Scope and inputs

`analyze_ski_pose(landmarks, width, height, visibility_threshold=0.5, visibility_threshold_legs=None, topology=MHR_BODY)` receives an ordered landmark list and **padded-ROI dimensions**. If the list has fewer items than `topology.num_landmarks`, it returns `None`. If `visibility_threshold_legs` is omitted, it equals the upper-body threshold.

For each required joint, `visibility >= threshold` is accepted. The threshold is `visibility_threshold_legs` for indices in `topology.leg_indices` and `visibility_threshold` otherwise. Missing visibility invalidates the *particular metric*, not necessarily the entire pose.

For the 2D `COCO_17` topology, all angle vectors come from `(int(lm.x*width), int(lm.y*height))`. For `MHR_BODY`, knee/hip/ankle vectors use `(x3d, y3d, z3d)` when present; `_angle_point` silently falls back to 2D if a landmark has no `x3d` attribute. **Shoulder tilt is always measured in projected 2D coordinates**. The 2D coordinates are integer-truncated before angle computation.

## 2. Geometric definition

For `calculate_angle(a,b,c)`, define `u=a-b`, `v=c-b`. The returned angle at vertex `b`, in degrees, is:

```text
theta = degrees(arccos(clip(dot(u,v)/(norm(u)*norm(v)), -1, 1)))
```

If either vector has zero length, the function returns `0.0`, not missing. The calculation does **not** explicitly reject NaN/Infinity. A `0.0` angle from degeneracy can be classified as bad; nonfinite values may propagate to grades rather than be excluded.

`calculate_horizontal_angle(left_shoulder,right_shoulder)` returns `degrees(atan2(delta_y,delta_x))` directly; it does not reduce to an undirected line angle. OpenCV image y increases downward, so the sign follows image-space convention rather than an independent gravity reference. Reversing left-to-right shoulder direction can produce near ±180° for a nearly horizontal segment. This is the existing [#12](https://github.com/kmch4n/SkiSense/issues/12) defect.

## 3. Evaluated metrics and exact thresholds

| Metric (key) | Three-point geometry or reference | Good: 100 | Warning: 50 | Bad: 0 |
| --- | --- | --- | --- | --- |
| `left_knee` / `right_knee` | hip–knee–ankle, vertex knee | [90°,120°] | [80°,90°) or (120°,140°] | All other computable values |
| `left_hip` / `right_hip` | shoulder–hip–knee, vertex hip | [100°,130°] | [90°,100°) or (130°,150°] | All other computable values |
| `shoulder_tilt` | `atan2` left→right shoulder image vector | abs(angle) <=10° | 10° < abs(angle) <=20° | abs(angle) >20° |
| `left_ankle` / `right_ankle` | knee–ankle–big toe, vertex ankle | [70°,90°] | (90°,110°] | All other computable values |

At most **seven** items are graded: two knees, two hips, one shoulder tilt, two ankles. In COCO-17, the topology has no foot landmark and both ankle metrics are **always unavailable**. Thus, YOLO models can grade at most five items; SAM3D can nominally grade seven. Additional occlusion/visibility restrictions reduce the count.

Labels are:
- Good `{"status":"good","label":"Good"}`;
- Knee/hip/ankle caution `{"status":"warning","label":"Caution"}`;
- Shoulder caution `{"status":"warning","label":"Tilted"}`;
- Bad knee/hip/ankle `{"status":"bad","label":"Fix"}`;
- Bad shoulder `{"status":"bad","label":"Over-tilt"}`;
- Unavailable `{"status":"info","label":"N/A"}`.

## 4. Overall heuristic score

For each metric with a non-`None` computed angle, add `100` for good, `50` for warning, `0` for bad; exclude `N/A` from both numerator and denominator. Then:

```text
score = int(sum(metric_points) / number_of_graded_metrics)
score = 0 if number_of_graded_metrics == 0
```

The returned integer is the **arithmetic mean with Python `int` truncation**. There is no separate weighting, temporal smoothing, confidence weighting, phase detection, symmetry penalty, adjudicator calibration, or normalization for camera angle. A `0` score may mean either all graded metrics were bad or none were available; inspect `evaluations` to distinguish those cases.

## 5. Return-value contract

```python
{
  "angles": {
    "left_knee": float, "right_knee": float,
    "left_hip": float, "right_hip": float,
    "shoulder_tilt": float,
    "left_ankle": float, "right_ankle": float
  },
  "evaluations": {
    "<each angle key>": {"status": "good|warning|bad|info", "label": str}
  },
  "score": int,
  "shoulder_center": (float, float),
  "hip_center": (float, float),
  "torso_center": (float, float)
}
```

The `angles` mapping stores **`0.0` for unavailable values**, so callers **must use `evaluations[key].status == 'info'`** to recognize `N/A`; a bare `0.0` is not reliable evidence of a measured angle. A too-short landmark list or an exception during analysis yields `None` (exceptions are swallowed with no diagnostic). Individual invalid joint geometry is not otherwise rejected robustly.

`shoulder_center` and `hip_center` are averages of 2D ROI-pixel points, regardless of visibility. `torso_center` is their midpoint if both shoulders and hips are visible; otherwise hip center if both hips are visible; otherwise shoulder center. It is used as a rendering/zoom target, not a biomechanical center of mass.

## 6. Model comparability and interpretation

A numerical match between YOLO and SAM3D does **not** imply equivalent posture. YOLO uses projected 2D knee and hip angles; SAM3D uses inferred camera-frame 3D vectors and includes two additional ankle metrics. SAM3D's used-joint visibility is fixed at 1.0, so occlusion does not suppress its metrics. Even YOLO11 and YOLO26 can differ due to checkpoints, confidence behavior and person association.

There is no **versioned, expert-validated scoring algorithm**, no demonstrated correspondence with ski-race/SAJ judging scores, and no defined camera-view or turn-phase applicability. The code evaluates the same fixed thresholds for every frame regardless of turn stage, piste slope or camera motion. `best_shot.jpg` is the rendered frame with the highest available heuristic score >0, which is not necessarily the technically best skiing moment.

## 7. Proposed acceptance work (not implemented)

Issue [#19](https://github.com/kmch4n/SkiSense/issues/19) must define an explicit scoring version; clear missing-data behavior (including zero vectors and NaN), angle units and coordinate frames; stable score inputs; model and view applicability; expert-labelled validation footage; and benchmark/error criteria **before** implementation. Regression tests should cover all threshold boundaries, 0/1/5/7 valid metrics, left-right swaps, mirrored frames, occlusions, missing feet, zero vectors, nonfinite values and per-backend equivalence where mathematically justified. A scientific claim requires actual evidence, not just tests against the existing constants.
