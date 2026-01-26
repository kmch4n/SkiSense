"""SkiSense - Ski Pose Analysis Tool

Analyzes ski videos to detect and evaluate skiing posture.
"""
# =============================================================================
# Configuration Import (must be first)
# =============================================================================
from .config import (
    DEBUG, SHOW_GUI, DEVICE_PREFERENCE,
    MODEL_DIR, INPUT_DIR, OUTPUT_DIR,
    YOLO_MODEL, POSE_MODEL, POSE_MODEL_URL,
    ZOOM_ENABLED, ZOOM_SCALE, ZOOM_SMOOTHING, ZOOM_PADDING,
    YOLO_CONFIDENCE, DEEPSORT_MAX_AGE, DEEPSORT_N_INIT,
    POSE_PRESENCE_CONF, POSE_TRACKING_CONF
)
from .zoom_tracker import ZoomTracker

# =============================================================================
# Suppress warnings when DEBUG is False
# =============================================================================
import os
import warnings

if not DEBUG:
    # Must be set before importing TensorFlow/MediaPipe
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['GLOG_minloglevel'] = '3'  # Suppress Google logging (used by MediaPipe)
    os.environ['GRPC_VERBOSITY'] = 'ERROR'
    warnings.filterwarnings('ignore')

    # Suppress absl logging (used by TensorFlow/MediaPipe)
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)

    import logging
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from .pose_analyzer import analyze_ski_pose, COLORS
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
import torch
import inspect
import tempfile
import shutil
import atexit


# Pose connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]


def resolve_device(preference: str):
    """Resolve device preference to actual device."""
    preference = preference.lower().strip()
    if preference == "cpu":
        return torch.device("cpu"), False, "cpu"
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0"), True, "cuda"
        print("Warning: CUDA forced but not available. Falling back to CPU.")
        return torch.device("cpu"), False, "cpu"
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
        # No zoom: ROI coords + bbox offset
        x, y, w, h = bbox
        return (point[0] + x, point[1] + y)

    # ROI coordinates -> Frame coordinates
    bx, by, bw, bh = bbox
    frame_x = bx + point[0]
    frame_y = by + point[1]

    # Frame coordinates -> Zoom coordinates
    x1, y1, x2, y2 = zoom_info['crop']
    crop_w = x2 - x1
    crop_h = y2 - y1

    if crop_w == 0 or crop_h == 0:
        return None

    # Get frame dimensions from zoom_info (inferred from scale)
    # The zoomed frame is resized back to original dimensions
    # So we need to map: (frame_x, frame_y) -> position in (x1, y1, x2, y2) -> (0, w) x (0, h)
    zoom_x = (frame_x - x1) / crop_w
    zoom_y = (frame_y - y1) / crop_h

    # Clamp to frame boundaries (always return valid coordinates)
    # This allows skeleton lines to be drawn even if endpoints are outside zoom region
    zoom_x = max(0, min(1, zoom_x))
    zoom_y = max(0, min(1, zoom_y))
    return (zoom_x, zoom_y)


def draw_landmarks_on_zoomed_frame(frame, landmarks, bbox, zoom_info):
    """Draw pose landmarks on zoomed frame.

    Args:
        frame: Zoomed frame (will be modified in-place)
        landmarks: MediaPipe pose landmarks
        bbox: (x, y, w, h) bounding box in original frame
        zoom_info: dict with zoom information
    """
    h, w = frame.shape[:2]
    bx, by, bw, bh = bbox

    # Transform and draw connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_lm = landmarks[start_idx]
            end_lm = landmarks[end_idx]

            # Get ROI coordinates
            start_roi = (start_lm.x * bw, start_lm.y * bh)
            end_roi = (end_lm.x * bw, end_lm.y * bh)

            # Transform to zoom coordinates
            start_zoom = transform_point_to_zoom(start_roi, bbox, zoom_info)
            end_zoom = transform_point_to_zoom(end_roi, bbox, zoom_info)

            if start_zoom and end_zoom:
                start_point = (int(start_zoom[0] * w), int(start_zoom[1] * h))
                end_point = (int(end_zoom[0] * w), int(end_zoom[1] * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    # Draw landmarks
    for landmark in landmarks:
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

    # Transform bbox corners
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

    # Get frame dimensions
    h, w = frame.shape[:2]

    # Adaptive scaling based on frame width
    # Base resolution: 1280px (720p standard)
    scale_factor = w / 1280.0

    # Panel settings (scaled)
    panel_x = int(10 * scale_factor)
    panel_y = int(10 * scale_factor)
    line_height = int(25 * scale_factor)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale_factor
    thickness = max(1, int(2 * scale_factor))

    # Draw semi-transparent background
    panel_height = line_height * 10 + int(20 * scale_factor)
    panel_width = int(280 * scale_factor)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Scaled text offsets
    text_padding = int(10 * scale_factor)
    column2_offset = int(150 * scale_factor)

    # Title
    y_pos = panel_y + line_height
    cv2.putText(frame, "Pose Analysis", (panel_x + text_padding, y_pos),
                font, 0.7 * scale_factor, (255, 255, 255), thickness)

    # Knee angles
    y_pos += line_height
    color = COLORS[evals['left_knee']['status']]
    cv2.putText(frame, f"L Knee: {angles['left_knee']:.0f}",
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_knee']['status']]
    cv2.putText(frame, f"R Knee: {angles['right_knee']:.0f}",
                (panel_x + column2_offset, y_pos), font, font_scale, color, thickness)

    # Hip angles
    y_pos += line_height
    color = COLORS[evals['left_hip']['status']]
    cv2.putText(frame, f"L Hip: {angles['left_hip']:.0f}",
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_hip']['status']]
    cv2.putText(frame, f"R Hip: {angles['right_hip']:.0f}",
                (panel_x + column2_offset, y_pos), font, font_scale, color, thickness)

    # Shoulder tilt
    y_pos += line_height
    color = COLORS[evals['shoulder_tilt']['status']]
    cv2.putText(frame, f"Shoulder Tilt: {angles['shoulder_tilt']:.1f}",
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)

    # Ankle angles
    y_pos += line_height
    color = COLORS[evals['left_ankle']['status']]
    cv2.putText(frame, f"L Ankle: {angles['left_ankle']:.0f}",
                (panel_x + text_padding, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_ankle']['status']]
    cv2.putText(frame, f"R Ankle: {angles['right_ankle']:.0f}",
                (panel_x + column2_offset, y_pos), font, font_scale, color, thickness)

    # Score
    y_pos += line_height + text_padding
    score_color = COLORS['good'] if score >= 70 else (COLORS['warning'] if score >= 40 else COLORS['bad'])
    cv2.putText(frame, f"Score: {score}/100",
                (panel_x + text_padding, y_pos), font, 0.8 * scale_factor, score_color, thickness)


def main(video_file: str = None):
    """Main processing function.

    Args:
        video_file: Video filename in input/ directory. Defaults to "video.mp4".
    """
    # Set default video file
    if video_file is None:
        video_file = "video.mp4"

    # Resolve device
    DEVICE, USE_CUDA, DEVICE_STR = resolve_device(DEVICE_PREFERENCE)
    if USE_CUDA:
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA runtime: {torch.version.cuda}, torch: {torch.__version__}")
    else:
        print("Using CPU (CUDA not available).")
        print(f"CUDA runtime: {torch.version.cuda}, torch: {torch.__version__}")

    # Setup MediaPipe Pose Landmarker (Tasks API)
    model_path = os.path.join(MODEL_DIR, POSE_MODEL)

    # Download model if not exists
    if not os.path.exists(model_path):
        import urllib.request
        print("Downloading pose landmarker model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        urllib.request.urlretrieve(POSE_MODEL_URL, model_path)
        print("Model downloaded successfully.")

    # Copy model to temp directory to avoid non-ASCII path issues with MediaPipe
    temp_dir = tempfile.mkdtemp()
    temp_model_path = os.path.join(temp_dir, POSE_MODEL)
    shutil.copy2(model_path, temp_model_path)
    atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

    # PoseLandmarker configuration
    base_options = python.BaseOptions(model_asset_path=temp_model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=POSE_PRESENCE_CONF,
        min_tracking_confidence=POSE_TRACKING_CONF,
        num_poses=1
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    # Input video path
    video_path = os.path.join(INPUT_DIR, video_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    # Get video properties for output
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    # Define output filename: same as input with '_pose' appended before the extension
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f"{base_name}_pose.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize ZoomTracker if enabled
    zoom_tracker = None
    if ZOOM_ENABLED:
        zoom_tracker = ZoomTracker(
            frame_width=width,
            frame_height=height,
            zoom_scale=ZOOM_SCALE,
            smoothing=ZOOM_SMOOTHING,
            padding=ZOOM_PADDING
        )

    # YOLO model for person detection
    yolo_model_path = os.path.join(MODEL_DIR, YOLO_MODEL)
    yolo_model = YOLO(yolo_model_path)
    if USE_CUDA:
        yolo_model.to(DEVICE)

    # Tracker setup
    tracker_kwargs = {
        "max_age": DEEPSORT_MAX_AGE,
        "n_init": DEEPSORT_N_INIT,
        "nms_max_overlap": 1.0,
        "embedder": "mobilenet",
        "half": USE_CUDA,
        "bgr": True
    }
    if "embedder_gpu" in inspect.signature(DeepSort).parameters:
        tracker_kwargs["embedder_gpu"] = USE_CUDA
    tracker = DeepSort(**tracker_kwargs)

    # Store latest pose analysis results for display
    latest_pose_analysis = None

    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps:.1f}")
    print(f"Output: {output_path}")
    if zoom_tracker is not None:
        print(f"Zoom tracking: enabled (scale: {ZOOM_SCALE}x)")
    else:
        print("Zoom tracking: disabled")
    print("-" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        # Show progress
        if current_frame % 30 == 0 or current_frame == total_frames:
            progress = (current_frame / total_frames) * 100
            sys.stdout.write(f"\rProcessing: {current_frame}/{total_frames} ({progress:.1f}%)")
            sys.stdout.flush()

        # YOLO person detection
        results = yolo_model(
            frame,
            classes=[0],
            conf=YOLO_CONFIDENCE,
            verbose=DEBUG,
            device=0 if USE_CUDA else "cpu",
            half=USE_CUDA
        )  # class 0 = person
        rects = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w, h = x2 - x1, y2 - y1
                rects.append((x1, y1, w, h))

        # Deep SORT format: ([x, y, w, h], confidence, class)
        detections = [([x, y, w, h], 1.0, 'person') for (x, y, w, h) in rects]

        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        # Create centroid dictionary from confirmed tracks
        objects = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            cx = int((ltrb[0] + ltrb[2]) / 2)
            cy = int((ltrb[1] + ltrb[3]) / 2)
            objects[track_id] = (cx, cy)

        # Step 1: Pose estimation only (no drawing)
        landmarks_data = []
        for (x, y, w, h) in rects:
            # Crop the region of interest (ROI)
            roi = frame[y:y+h, x:x+w]
            # Convert from BGR to RGB for MediaPipe
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            # Create mp.Image object for Tasks API
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

            # Run pose estimation
            detection_result = pose_landmarker.detect(mp_image)

            if detection_result.pose_landmarks:
                # Analyze ski pose
                roi_h, roi_w = roi_rgb.shape[:2]
                analysis = analyze_ski_pose(detection_result.pose_landmarks[0], roi_w, roi_h)
                if analysis:
                    latest_pose_analysis = analysis

                    # Store landmarks and metadata for later drawing
                    landmarks_data.append({
                        'landmarks': detection_result.pose_landmarks[0],
                        'bbox': (x, y, w, h),
                        'shoulder_center': analysis.get('shoulder_center')
                    })

        # Step 2: Apply zoom tracking if enabled
        output_frame = frame.copy()
        zoom_info = None
        if zoom_tracker is not None:
            # Use shoulder center from first detected person
            shoulder_center = landmarks_data[0]['shoulder_center'] if landmarks_data else None
            output_frame, zoom_info = zoom_tracker.process_frame(
                frame, tracks, rects, shoulder_center
            )

        # Step 3: Draw on zoomed frame
        for data in landmarks_data:
            # Draw landmarks
            draw_landmarks_on_zoomed_frame(
                output_frame, data['landmarks'], data['bbox'], zoom_info
            )
            # Draw bounding box
            draw_bbox_on_zoomed_frame(output_frame, data['bbox'], zoom_info)

        # Draw pose analysis info panel (on zoomed frame)
        draw_info_panel(output_frame, latest_pose_analysis)

        # Write frame to output video
        out.write(output_frame)

        # Show preview window if GUI is enabled
        if SHOW_GUI:
            cv2.imshow("Ski Video - Pose & Tracking", output_frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("\n\nProcessing cancelled by user.")
                break

    # Cleanup
    print(f"\n\nProcessing complete!")
    print(f"Output saved to: {output_path}")

    cap.release()
    out.release()
    pose_landmarker.close()
    if SHOW_GUI:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
