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
    ZOOM_ENABLED, ZOOM_SCALE, ZOOM_SMOOTHING, ZOOM_PADDING,
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
from datetime import datetime

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

    # Clamp to frame boundaries so skeleton lines still render even when
    # an endpoint falls outside the zoom region.
    zoom_x = max(0, min(1, zoom_x))
    zoom_y = max(0, min(1, zoom_y))
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


def run_yolo_detection(yolo_model, frame, yolo_device, yolo_half: bool):
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
            rects.append((int(x1), int(y1), int(w), int(h)))
    return rects


def process_video(video_file: str = None, high_precision: bool = False):
    """Main processing function for video input.

    Args:
        video_file: Video filename in input/ directory. Defaults to "video.mp4".
        high_precision: If True, use frame interpolation for higher accuracy.
    """
    if video_file is None:
        video_file = "video.mp4"

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

    if DEVICE_STR == "cuda":
        log_message("  - Deep SORT: CUDA GPU")
    else:
        log_message("  - Deep SORT: CPU" + (" (MPS not supported)" if DEVICE_STR == "mps" else ""))

    log_message("  - Pose: YOLO11-Pose")
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

    output_video_path = os.path.join(output_dir, "video_pose.mp4")
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
        )

    yolo_model = load_yolo_model(DEVICE, USE_CUDA)

    # Deep SORT tracker: GPU acceleration is CUDA-only, not supported on MPS.
    tracker_kwargs = {
        "max_age": DEEPSORT_MAX_AGE,
        "n_init": DEEPSORT_N_INIT,
        "nms_max_overlap": 1.0,
        "embedder": "mobilenet",
        "half": (DEVICE_STR == "cuda"),
        "bgr": True,
    }
    if "embedder_gpu" in inspect.signature(DeepSort).parameters:
        tracker_kwargs["embedder_gpu"] = (DEVICE_STR == "cuda")
    tracker = DeepSort(**tracker_kwargs)

    latest_pose_analysis = None
    best_score = -1
    best_frame = None
    best_frame_number = 0

    start_time = time.time()

    log_message("処理を開始します")
    if high_precision:
        log_message("高精度モード: フレーム補間を使用（将来実装予定）")
    log_message("処理中...")

    if DEVICE_STR == "cuda":
        yolo_device, yolo_half = 0, True
    elif DEVICE_STR == "mps":
        yolo_device, yolo_half = "mps", False
    else:
        yolo_device, yolo_half = "cpu", False

    pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1
        pbar.update(1)

        rects = run_yolo_detection(yolo_model, frame, yolo_device, yolo_half)

        detections = [([x, y, w, h], 1.0, 'person') for (x, y, w, h) in rects]
        tracks = tracker.update_tracks(detections, frame=frame)

        # Step 1: Pose estimation through the selected backend.
        landmarks_data = []
        timestamp_ms = int(current_frame * 1000 / fps)
        for bbox in rects:
            entry, analysis = pose_backend.estimate(frame, bbox, timestamp_ms=timestamp_ms)
            if entry is not None and analysis is not None:
                latest_pose_analysis = analysis
                landmarks_data.append(entry)

        # Step 2: Apply zoom tracking if enabled.
        output_frame = frame.copy()
        if zoom_tracker is not None:
            shoulder_center = landmarks_data[0]['shoulder_center'] if landmarks_data else None
            output_frame, zoom_info = zoom_tracker.process_frame(
                frame, tracks, rects, shoulder_center,
            )
        else:
            zoom_info = {
                'center': (width / 2, height / 2),
                'scale': 1.0,
                'crop': (0, 0, width, height),
            }

        # Step 3: Draw on the zoomed frame.
        for (x, y, w, h) in rects:
            draw_bbox_on_zoomed_frame(output_frame, (x, y, w, h), zoom_info)

        for data in landmarks_data:
            draw_landmarks_on_zoomed_frame(
                output_frame, data['landmarks'], data['bbox'], zoom_info,
                topology=data.get('topology', COCO_17),
            )

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
