"""SkiSense - Ski Pose Analysis Tool

Analyzes ski videos to detect and evaluate skiing posture.
"""
# =============================================================================
# Configuration + shared stderr suppression (must come before heavy imports)
# =============================================================================
from .config import (
    SHOW_GUI, DEVICE_PREFERENCE,
    MODEL_DIR, INPUT_DIR, OUTPUT_DIR,
    YOLO_MODEL,
    TARGET_SELECTION_MODE,
    ZOOM_ENABLED, ZOOM_SCALE, ZOOM_SMOOTHING, ZOOM_PADDING,
    ZOOM_TARGET_AREA_RATIO, ZOOM_MAX_SCALE,
    YOLO_CONFIDENCE, DEEPSORT_MAX_AGE, DEEPSORT_N_INIT,
    POSE_VISIBILITY_THRESHOLD, POSE_VISIBILITY_THRESHOLD_LEGS,
)
from ._logging import DEBUG, SuppressStderr
from .backends import get_backend
from .pose_topology import COCO_17, PoseTopology
from .zoom_tracker import ZoomTracker

import inspect
import os
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

with SuppressStderr():
    import cv2
    from deep_sort_realtime.deepsort_tracker import DeepSort
    from ultralytics import YOLO
    from .pose_analyzer import COLORS


def log_message(message: str):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


# Module-level alias for callers that import the active skeleton edges.
POSE_CONNECTIONS = COCO_17.connections
BBox = Tuple[int, int, int, int]


@dataclass
class TrackStats:
    """Aggregated visibility stats for one Deep SORT track."""

    track_id: int
    frames_seen: int = 0
    first_frame: Optional[int] = None
    last_frame: Optional[int] = None
    area_sum: float = 0.0
    center_distance_sum: float = 0.0

    def update(
        self,
        frame_number: int,
        bbox: BBox,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Record one visible frame for this track."""
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        frame_center_x = frame_width / 2
        frame_center_y = frame_height / 2

        self.frames_seen += 1
        self.first_frame = frame_number if self.first_frame is None else self.first_frame
        self.last_frame = frame_number
        self.area_sum += max(0, w) * max(0, h)
        self.center_distance_sum += float(
            np.hypot(center_x - frame_center_x, center_y - frame_center_y)
        )

    @property
    def average_area(self) -> float:
        """Return the average bbox area for tie-breaking."""
        if self.frames_seen == 0:
            return 0.0
        return self.area_sum / self.frames_seen

    @property
    def average_center_distance(self) -> float:
        """Return average distance from the frame center."""
        if self.frames_seen == 0:
            return float("inf")
        return self.center_distance_sum / self.frames_seen


@dataclass
class TargetTrackPlan:
    """Frame-by-frame bboxes for the selected main skier track."""

    track_id: int
    stats: TrackStats
    frame_bboxes: Dict[int, BBox] = field(default_factory=dict)

    def bbox_for_frame(self, frame_number: int) -> Optional[BBox]:
        """Return the selected track bbox for one frame if visible."""
        return self.frame_bboxes.get(frame_number)


def visibility_threshold_for(index: int, topology: PoseTopology = COCO_17) -> float:
    """Return the visibility threshold that applies to a landmark index."""
    if index in topology.leg_indices:
        return POSE_VISIBILITY_THRESHOLD_LEGS
    return POSE_VISIBILITY_THRESHOLD


def resolve_device(preference: str):
    """Resolve device preference to actual device.

    Returns:
        tuple: (torch.device, bool, str)
            - device: PyTorch device object
            - use_gpu: True if GPU (CUDA/MPS) is available
            - device_str: "cuda" | "mps" | "cpu"
    """
    preference = preference.lower().strip()

    if preference == "cpu":
        return torch.device("cpu"), False, "cpu"

    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), True, "cuda"
        log_message("Warning: CUDA forced but not available. Falling back to CPU.")
        return torch.device("cpu"), False, "cpu"

    if preference == "mps":
        if not hasattr(torch.backends, 'mps'):
            log_message("Warning: MPS not supported in this PyTorch version. Falling back to CPU.")
            return torch.device("cpu"), False, "cpu"

        if torch.backends.mps.is_available():
            try:
                device = torch.device("mps")
                _ = torch.zeros(1, device=device)
                return device, True, "mps"
            except Exception as e:
                log_message(f"Warning: MPS device creation failed: {e}. Falling back to CPU.")
                return torch.device("cpu"), False, "cpu"

        log_message("Warning: MPS forced but not available. Falling back to CPU.")
        return torch.device("cpu"), False, "cpu"

    # Auto: MPS > CUDA > CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            _ = torch.zeros(1, device=device)
            return device, True, "mps"
        except Exception:
            pass

    if torch.cuda.is_available():
        return torch.device("cuda:0"), True, "cuda"

    return torch.device("cpu"), False, "cpu"


def transform_point_to_zoom(point, bbox, zoom_info):
    """Transform a point from ROI coordinates to zoomed frame coordinates.

    Args:
        point: (x, y) in ROI coordinates
        bbox: (x, y, w, h) bounding box in original frame
        zoom_info: dict with 'crop' key containing (x1, y1, x2, y2)

    Returns:
        (x, y) in zoomed frame coordinates, or None if outside zoom region
    """
    if zoom_info is None:
        x, y, w, h = bbox
        return (point[0] + x, point[1] + y)

    bx, by, bw, bh = bbox
    frame_x = bx + point[0]
    frame_y = by + point[1]

    x1, y1, x2, y2 = zoom_info['crop']
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w == 0 or crop_h == 0:
        return None

    zoom_x = (frame_x - x1) / crop_w
    zoom_y = (frame_y - y1) / crop_h

    if zoom_x < 0 or zoom_x > 1 or zoom_y < 0 or zoom_y > 1:
        return None
    return (zoom_x, zoom_y)


def draw_landmarks_on_zoomed_frame(frame, landmarks, bbox, zoom_info, topology: PoseTopology = COCO_17):
    """Draw pose landmarks on a (possibly zoomed) frame.

    Args:
        frame: Zoomed frame (will be modified in-place).
        landmarks: List of landmark objects (``.x``, ``.y``, ``.visibility``).
        bbox: (x, y, w, h) padded ROI bbox in original frame coordinates.
        zoom_info: Dict with zoom crop info.
        topology: Pose topology describing skeleton edges and leg indices.
    """
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox

    def _threshold_for(idx: int) -> float:
        if idx in topology.leg_indices:
            return POSE_VISIBILITY_THRESHOLD_LEGS
        return POSE_VISIBILITY_THRESHOLD

    for start_idx, end_idx in topology.connections:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        start_lm = landmarks[start_idx]
        end_lm = landmarks[end_idx]

        if (start_lm.visibility < _threshold_for(start_idx)
                or end_lm.visibility < _threshold_for(end_idx)):
            continue

        start_roi = (start_lm.x * bw, start_lm.y * bh)
        end_roi = (end_lm.x * bw, end_lm.y * bh)

        start_zoom = transform_point_to_zoom(start_roi, bbox, zoom_info)
        end_zoom = transform_point_to_zoom(end_roi, bbox, zoom_info)

        if start_zoom and end_zoom:
            start_point = (int(start_zoom[0] * w), int(start_zoom[1] * h))
            end_point = (int(end_zoom[0] * w), int(end_zoom[1] * h))
            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    for idx, landmark in enumerate(landmarks):
        if landmark.visibility < _threshold_for(idx):
            continue

        roi_point = (landmark.x * bw, landmark.y * bh)
        zoom_point = transform_point_to_zoom(roi_point, bbox, zoom_info)

        if zoom_point:
            px = int(zoom_point[0] * w)
            py = int(zoom_point[1] * h)
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)


def draw_bbox_on_zoomed_frame(frame, bbox, zoom_info):
    """Draw bounding box on zoomed frame.

    Args:
        frame: Zoomed frame (will be modified in-place)
        bbox: (x, y, w, h) bounding box in original frame
        zoom_info: dict with zoom information
    """
    h, w = frame.shape[:2]
    x, y, bw, bh = bbox

    tl = transform_point_to_zoom((0, 0), bbox, zoom_info)
    br = transform_point_to_zoom((bw, bh), bbox, zoom_info)

    if tl and br:
        pt1 = (int(tl[0] * w), int(tl[1] * h))
        pt2 = (int(br[0] * w), int(br[1] * h))
        cv2.rectangle(frame, pt1, pt2, (255, 0, 0), 2)


def draw_target_overlay(frame, target_bbox, target_entry, zoom_info):
    """Draw only the selected skier overlay on the output frame."""
    if target_entry is not None:
        overlay_bbox = target_entry.get("detection_bbox", target_entry["bbox"])
    else:
        overlay_bbox = target_bbox

    if overlay_bbox is not None:
        draw_bbox_on_zoomed_frame(frame, overlay_bbox, zoom_info)

    if target_entry is not None:
        draw_landmarks_on_zoomed_frame(
            frame,
            target_entry["landmarks"],
            target_entry["bbox"],
            zoom_info,
            topology=target_entry.get("topology", COCO_17),
        )


def draw_info_panel(frame, analysis):
    """Draw pose analysis info panel on top-left of frame with adaptive scaling."""
    if analysis is None:
        return

    angles = analysis['angles']
    evals = analysis['evaluations']
    score = analysis['score']

    h, w = frame.shape[:2]

    scale_factor = w / 1280.0

    panel_x = int(10 * scale_factor)
    panel_y = int(10 * scale_factor)
    line_height = int(25 * scale_factor)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale_factor
    thickness = max(1, int(2 * scale_factor))

    panel_height = line_height * 10 + int(20 * scale_factor)
    panel_width = int(280 * scale_factor)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    text_padding = int(10 * scale_factor)
    column2_offset = int(150 * scale_factor)

    def _format_metric(label: str, key: str, fmt: str) -> str:
        evaluation = evals[key]
        if evaluation['status'] == 'info' or evaluation['label'] == 'N/A':
            return f"{label}: N/A"
        return f"{label}: {angles[key]:{fmt}}"

    y_pos = panel_y + line_height
    cv2.putText(frame, "Pose Analysis", (panel_x + text_padding, y_pos),
                font, 0.7 * scale_factor, (255, 255, 255), thickness)

    y_pos += line_height
    color = COLORS[evals['left_knee']['status']]
    cv2.putText(frame, _format_metric("L Knee", "left_knee", ".0f"),
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_knee']['status']]
    cv2.putText(frame, _format_metric("R Knee", "right_knee", ".0f"),
                (panel_x + column2_offset, y_pos), font, font_scale, color, thickness)

    y_pos += line_height
    color = COLORS[evals['left_hip']['status']]
    cv2.putText(frame, _format_metric("L Hip", "left_hip", ".0f"),
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_hip']['status']]
    cv2.putText(frame, _format_metric("R Hip", "right_hip", ".0f"),
                (panel_x + column2_offset, y_pos), font, font_scale, color, thickness)

    y_pos += line_height
    color = COLORS[evals['shoulder_tilt']['status']]
    cv2.putText(frame, _format_metric("Shoulder Tilt", "shoulder_tilt", ".1f"),
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)

    y_pos += line_height
    color = COLORS[evals['left_ankle']['status']]
    cv2.putText(frame, _format_metric("L Ankle", "left_ankle", ".0f"),
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_ankle']['status']]
    cv2.putText(frame, _format_metric("R Ankle", "right_ankle", ".0f"),
                (panel_x + column2_offset, y_pos), font, font_scale, color, thickness)

    y_pos += line_height + text_padding
    score_color = COLORS['good'] if score >= 70 else (COLORS['warning'] if score >= 40 else COLORS['bad'])
    cv2.putText(frame, f"Score: {score}/100",
                (panel_x + text_padding, y_pos), font, 0.8 * scale_factor, score_color, thickness)


def load_yolo_model(device, use_gpu: bool):
    """Load YOLO weights for person detection and move to the target device."""
    yolo_model_path = os.path.join(MODEL_DIR, YOLO_MODEL)

    with SuppressStderr():
        yolo_model = YOLO(yolo_model_path)
        if use_gpu:
            yolo_model.to(device)
    return yolo_model


def create_deepsort_tracker(device_str: str) -> DeepSort:
    """Create a Deep SORT tracker with the project defaults."""
    tracker_kwargs = {
        "max_age": DEEPSORT_MAX_AGE,
        "n_init": DEEPSORT_N_INIT,
        "nms_max_overlap": 1.0,
        "embedder": "mobilenet",
        "half": (device_str == "cuda"),
        "bgr": True,
    }
    if "embedder_gpu" in inspect.signature(DeepSort).parameters:
        tracker_kwargs["embedder_gpu"] = (device_str == "cuda")
    return DeepSort(**tracker_kwargs)


def run_yolo_detection(yolo_model, frame, yolo_device, yolo_half: bool) -> list[BBox]:
    """Run YOLO person detection on a frame.

    Returns:
        List of (x, y, w, h) bboxes in absolute frame coordinates.
    """
    with SuppressStderr():
        results = yolo_model(
            frame, classes=[0], conf=YOLO_CONFIDENCE,
            verbose=DEBUG, device=yolo_device, half=yolo_half,
        )

    rects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                continue
            rects.append((int(x1), int(y1), int(w), int(h)))
    return rects


def ltrb_to_xywh(ltrb: Tuple[float, ...], width: int, height: int) -> Optional[BBox]:
    """Convert Deep SORT ltrb coordinates to a clipped xywh bbox."""
    left, top, right, bottom = ltrb
    x1 = max(0, min(width, int(round(left))))
    y1 = max(0, min(height, int(round(top))))
    x2 = max(0, min(width, int(round(right))))
    y2 = max(0, min(height, int(round(bottom))))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def choose_longest_track(stats_by_id: Dict[int, TrackStats]) -> Optional[TrackStats]:
    """Choose the most likely main skier from accumulated track stats."""
    if not stats_by_id:
        return None
    return max(
        stats_by_id.values(),
        key=lambda stats: (
            stats.frames_seen,
            stats.average_area,
            -stats.average_center_distance,
        ),
    )


def build_longest_track_plan(
    video_path: str,
    width: int,
    height: int,
    total_frames: int,
    yolo_model,
    yolo_device,
    yolo_half: bool,
    device_str: str,
) -> Optional[TargetTrackPlan]:
    """Pre-scan the video and lock onto the longest visible person track."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_message(f"Warning: could not open video for target selection: {video_path}")
        return None

    tracker = create_deepsort_tracker(device_str)
    stats_by_id: Dict[int, TrackStats] = {}
    bboxes_by_track: Dict[int, Dict[int, BBox]] = {}
    frame_number = 0

    pbar = tqdm(total=total_frames, desc="Selecting target", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        pbar.update(1)

        rects = run_yolo_detection(yolo_model, frame, yolo_device, yolo_half)
        detections = [([x, y, w, h], 1.0, 'person') for (x, y, w, h) in rects]
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            bbox = ltrb_to_xywh(track.to_ltrb(), width, height)
            if bbox is None:
                continue

            track_id = int(track.track_id)
            stats = stats_by_id.setdefault(track_id, TrackStats(track_id=track_id))
            stats.update(frame_number, bbox, width, height)
            bboxes_by_track.setdefault(track_id, {})[frame_number] = bbox

    pbar.close()
    cap.release()

    selected_stats = choose_longest_track(stats_by_id)
    if selected_stats is None:
        log_message("Warning: no confirmed tracks found; falling back to largest-person zoom.")
        return None

    coverage = selected_stats.frames_seen / max(1, total_frames) * 100
    log_message(
        "Target track selected: "
        f"id={selected_stats.track_id}, "
        f"frames={selected_stats.frames_seen}/{total_frames} ({coverage:.1f}%)"
    )
    return TargetTrackPlan(
        track_id=selected_stats.track_id,
        stats=selected_stats,
        frame_bboxes=bboxes_by_track.get(selected_stats.track_id, {}),
    )


def identity_zoom_info(width: int, height: int) -> dict:
    """Return zoom metadata that maps frame coordinates directly."""
    return {
        'center': (width / 2, height / 2),
        'scale': 1.0,
        'crop': (0, 0, width, height),
    }


def bbox_area(bbox) -> int:
    """Return bbox area for target selection."""
    if bbox is None:
        return 0
    return max(0, int(bbox[2])) * max(0, int(bbox[3]))


def pose_result_for_bbox(
    pose_results: list[tuple[dict, dict]],
    bbox: Optional[BBox],
) -> Optional[tuple[dict, dict]]:
    """Find the pose result that best matches a detector/tracker bbox."""
    if bbox is None:
        return None

    best_result = None
    best_iou = 0.0
    for result in pose_results:
        entry, _analysis = result
        entry_bbox = entry.get("detection_bbox")
        if entry_bbox == bbox:
            return result
        score = bbox_iou(entry_bbox, bbox)
        if score > best_iou:
            best_iou = score
            best_result = result

    if best_result is not None and best_iou > 0.05:
        return best_result
    return None


def select_primary_fast_pose(
    pose_results: list[tuple[dict, dict]],
    target_bbox: Optional[BBox] = None,
) -> Optional[tuple[dict, dict]]:
    """Select the target pose result from full-frame pose results."""
    if not pose_results:
        return None
    if target_bbox is not None:
        return pose_result_for_bbox(pose_results, target_bbox)
    return max(
        pose_results,
        key=lambda item: bbox_area(item[0].get("detection_bbox")),
    )


def frame_center_from_entry(entry, key: str = "torso_center"):
    """Return a pose center converted into full-frame coordinates."""
    center = entry.get(key) or entry.get("torso_center") or entry.get("shoulder_center")
    if center is None:
        return None
    x, y, _w, _h = entry["bbox"]
    return (x + center[0], y + center[1])


def bbox_iou(a, b) -> float:
    """Return intersection-over-union for two xywh bboxes."""
    if a is None or b is None:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_w = max(0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0, min(ay2, by2) - max(ay, by))
    inter_area = inter_w * inter_h
    union_area = aw * ah + bw * bh - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def process_fast_frame(
    frame,
    pose_backend,
    zoom_tracker,
    width: int,
    height: int,
    target_bbox: Optional[BBox] = None,
):
    """Process one frame with full-frame YOLO11-Pose only.

    This skips the separate YOLOv8 detector, Deep SORT, and ROI pose calls.
    """
    pose_results = pose_backend.estimate_full_frame(frame)
    primary = select_primary_fast_pose(pose_results, target_bbox)
    primary_entry = primary[0] if primary is not None else None
    current_analysis = primary[1] if primary is not None else None

    zoom_bbox = target_bbox
    if zoom_tracker is not None:
        if zoom_bbox is None and primary_entry is not None:
            zoom_bbox = primary_entry.get("detection_bbox")
        target_center = (
            frame_center_from_entry(primary_entry)
            if primary_entry is not None else None
        )
        output_frame, zoom_info = zoom_tracker.process_bbox(
            frame,
            zoom_bbox,
            target_center,
        )
    else:
        output_frame = frame.copy()
        zoom_info = identity_zoom_info(width, height)

    draw_target_overlay(output_frame, zoom_bbox, primary_entry, zoom_info)
    draw_info_panel(output_frame, current_analysis)
    return output_frame, current_analysis


def process_video(
    video_file: str = None,
    high_precision: bool = False,
    fast_mode: bool = False,
    target_mode: Optional[str] = None,
):
    """Main processing function for video input.

    Args:
        video_file: Video filename in input/ directory. Defaults to "video.mp4".
        high_precision: If True, use frame interpolation for higher accuracy.
        fast_mode: If True, use full-frame YOLO11-Pose without Deep SORT.
        target_mode: "longest" locks zoom to the longest visible track.
    """
    if video_file is None:
        video_file = "video.mp4"
    if target_mode is None:
        target_mode = TARGET_SELECTION_MODE
    target_mode = target_mode.lower().strip()
    if target_mode not in ("longest", "largest"):
        raise ValueError(f"Unsupported target_mode: {target_mode}")

    DEVICE, USE_CUDA, DEVICE_STR = resolve_device(DEVICE_PREFERENCE)

    if DEVICE_STR == "cuda":
        torch.backends.cudnn.benchmark = True

    log_message(f"Using device: {DEVICE_STR.upper()}")
    if USE_CUDA:
        log_message(f"GPU acceleration: enabled ({DEVICE})")

    log_message("=" * 40)
    log_message("Component configuration:")

    if DEVICE_STR == "cuda":
        log_message("  - YOLO: CUDA GPU (half=True)")
    elif DEVICE_STR == "mps":
        log_message("  - YOLO: MPS GPU (half=False)")
    else:
        log_message("  - YOLO: CPU")

    if fast_mode and target_mode == "longest" and ZOOM_ENABLED:
        log_message("  - Deep SORT: first-pass target selection")
        log_message("  - Fast mode: full-frame YOLO11-Pose, ROI preprocessing/TTA skipped")
    elif fast_mode:
        log_message("  - Deep SORT: disabled (fast mode)")
        log_message("  - Fast mode: full-frame YOLO11-Pose, ROI preprocessing/TTA skipped")
    elif DEVICE_STR == "cuda":
        log_message("  - Deep SORT: CUDA GPU")
    else:
        log_message("  - Deep SORT: CPU" + (" (MPS not supported)" if DEVICE_STR == "mps" else ""))

    log_message("  - Pose: YOLO11-Pose")
    log_message(f"  - Target selection: {target_mode}")
    log_message("=" * 40)

    pose_backend = get_backend(
        running_mode="video",
        device=DEVICE,
        use_gpu=USE_CUDA,
        device_str=DEVICE_STR,
    )

    video_path = os.path.join(INPUT_DIR, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_message(f"Error: Could not open video: {video_path}")
        pose_backend.close()
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    invalid_fields = []
    if not fps or fps <= 0:
        invalid_fields.append(f"fps={fps}")
    if width <= 0:
        invalid_fields.append(f"width={width}")
    if height <= 0:
        invalid_fields.append(f"height={height}")
    if invalid_fields:
        log_message(
            "Error: invalid video metadata "
            f"({', '.join(invalid_fields)}) for {video_path}"
        )
        cap.release()
        pose_backend.close()
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    output_video_name = "video_pose_fast.mp4" if fast_mode else "video_pose.mp4"
    output_video_path = os.path.join(output_dir, output_video_name)
    best_shot_path = os.path.join(output_dir, "best_shot.jpg")
    input_copy_path = os.path.join(output_dir, "video.mp4")

    shutil.copy2(video_path, input_copy_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    zoom_tracker = None
    if ZOOM_ENABLED:
        zoom_tracker = ZoomTracker(
            frame_width=width,
            frame_height=height,
            zoom_scale=ZOOM_SCALE,
            smoothing=ZOOM_SMOOTHING,
            padding=ZOOM_PADDING,
            target_area_ratio=ZOOM_TARGET_AREA_RATIO,
            max_scale=ZOOM_MAX_SCALE,
        )

    if fast_mode and not hasattr(pose_backend, "estimate_full_frame"):
        log_message("Error: selected pose backend does not support fast mode.")
        cap.release()
        out.release()
        pose_backend.close()
        return

    if DEVICE_STR == "cuda":
        yolo_device, yolo_half = 0, True
    elif DEVICE_STR == "mps":
        yolo_device, yolo_half = "mps", False
    else:
        yolo_device, yolo_half = "cpu", False

    yolo_model = None
    if not fast_mode or (ZOOM_ENABLED and target_mode == "longest"):
        yolo_model = load_yolo_model(DEVICE, USE_CUDA)

    start_time = time.time()
    target_plan = None
    active_target_mode = target_mode
    if ZOOM_ENABLED and target_mode == "longest":
        log_message("主対象選択: 最長出演 track を事前解析しています")
        target_plan = build_longest_track_plan(
            video_path,
            width,
            height,
            total_frames,
            yolo_model,
            yolo_device,
            yolo_half,
            DEVICE_STR,
        )
        if target_plan is None:
            active_target_mode = "largest"

    tracker = None
    if not fast_mode and active_target_mode == "largest":
        tracker = create_deepsort_tracker(DEVICE_STR)

    latest_pose_analysis = None
    best_score = -1
    best_frame = None
    best_frame_number = 0

    log_message("処理を開始します")
    if high_precision:
        log_message("高精度モード: フレーム補間を使用（将来実装予定）")
    if fast_mode and active_target_mode == "longest":
        log_message("高速モード: full-frame姿勢推定と事前選択した主対象bboxを使用")
    elif fast_mode:
        log_message("高速モード: YOLOv8検出、Deep SORT、ROI前処理/TTAを省略")
    log_message("処理中...")

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        pbar.update(1)

        if fast_mode:
            target_bbox = (
                target_plan.bbox_for_frame(current_frame)
                if target_plan is not None and active_target_mode == "longest" else None
            )
            output_frame, latest_pose_analysis = process_fast_frame(
                frame,
                pose_backend,
                zoom_tracker,
                width,
                height,
                target_bbox,
            )
        else:
            rects = run_yolo_detection(yolo_model, frame, yolo_device, yolo_half)

            tracks = []
            if tracker is not None:
                detections = [([x, y, w, h], 1.0, 'person') for (x, y, w, h) in rects]
                tracks = tracker.update_tracks(detections, frame=frame)

            # Step 1: Pose estimation through the selected backend.
            pose_results = []
            timestamp_ms = int(current_frame * 1000 / fps)
            for bbox in rects:
                entry, analysis = pose_backend.estimate(frame, bbox, timestamp_ms=timestamp_ms)
                if entry is not None and analysis is not None:
                    pose_results.append((entry, analysis))

            output_frame = frame.copy()
            current_pose_analysis = None
            target_bbox = None
            target_entry = None
            if zoom_tracker is not None:
                if active_target_mode == "longest" and target_plan is not None:
                    target_bbox = target_plan.bbox_for_frame(current_frame)
                else:
                    target_bbox = zoom_tracker.select_main_target(tracks, rects)

                target_result = pose_result_for_bbox(pose_results, target_bbox)
                target_entry = target_result[0] if target_result is not None else None
                current_pose_analysis = target_result[1] if target_result is not None else None
                target_center = (
                    frame_center_from_entry(target_entry)
                    if target_entry is not None else None
                )
                output_frame, zoom_info = zoom_tracker.process_bbox(
                    frame,
                    target_bbox,
                    target_center,
                )
            else:
                zoom_info = identity_zoom_info(width, height)
                if pose_results:
                    current_pose_analysis = pose_results[0][1]

            latest_pose_analysis = current_pose_analysis

            draw_target_overlay(output_frame, target_bbox, target_entry, zoom_info)
            draw_info_panel(output_frame, latest_pose_analysis)

        if latest_pose_analysis and latest_pose_analysis['score'] > best_score:
            best_score = latest_pose_analysis['score']
            best_frame = output_frame.copy()
            best_frame_number = current_frame

        out.write(output_frame)

        if SHOW_GUI:
            cv2.imshow("Ski Video - Pose & Tracking", output_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                log_message("処理が中断されました")
                break

    pbar.close()

    if best_frame is not None and best_score > 0:
        cv2.imwrite(best_shot_path, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        log_message(f"ベストショット: フレーム {best_frame_number} (スコア: {best_score}/100)")
    else:
        log_message("スコアが記録されませんでした")

    end_time = time.time()
    elapsed_seconds = int(end_time - start_time)
    minutes = elapsed_seconds // 60
    seconds = elapsed_seconds % 60
    elapsed_str = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"

    log_message(f"出力先: {output_dir}")
    log_message(f"処理が完了しました ({elapsed_str})")

    cap.release()
    out.release()
    pose_backend.close()
    if SHOW_GUI:
        cv2.destroyAllWindows()


# Backward-compatible alias
main = process_video


if __name__ == "__main__":
    process_video()
