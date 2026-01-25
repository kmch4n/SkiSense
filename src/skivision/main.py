"""SkiVision - Ski Pose Analysis Tool

Analyzes ski videos to detect and evaluate skiing posture.
"""
# =============================================================================
# Configuration Import (must be first)
# =============================================================================
from .config import (
    DEBUG, SHOW_GUI, DEVICE_PREFERENCE,
    MODEL_DIR, INPUT_DIR, OUTPUT_DIR,
    YOLO_MODEL, POSE_MODEL, POSE_MODEL_URL
)

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


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks on image using OpenCV."""
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    h, w = annotated_image.shape[:2]

    for pose_landmarks in pose_landmarks_list:
        # Draw connections (green lines)
        for connection in POSE_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks):
                start = pose_landmarks[start_idx]
                end = pose_landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(annotated_image, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks (red circles)
        for landmark in pose_landmarks:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(annotated_image, (cx, cy), 3, (0, 0, 255), -1)

    return annotated_image


def draw_info_panel(frame, analysis):
    """Draw pose analysis info panel on top-left of frame."""
    if analysis is None:
        return

    angles = analysis['angles']
    evals = analysis['evaluations']
    score = analysis['score']

    # Panel settings
    panel_x = 10
    panel_y = 10
    line_height = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Draw semi-transparent background
    panel_height = line_height * 10 + 20
    panel_width = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    y_pos = panel_y + line_height
    cv2.putText(frame, "Pose Analysis", (panel_x + 10, y_pos),
                font, 0.7, (255, 255, 255), thickness)

    # Knee angles
    y_pos += line_height
    color = COLORS[evals['left_knee']['status']]
    cv2.putText(frame, f"L Knee: {angles['left_knee']:.0f}",
                (panel_x + 10, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_knee']['status']]
    cv2.putText(frame, f"R Knee: {angles['right_knee']:.0f}",
                (panel_x + 150, y_pos), font, font_scale, color, thickness)

    # Hip angles
    y_pos += line_height
    color = COLORS[evals['left_hip']['status']]
    cv2.putText(frame, f"L Hip: {angles['left_hip']:.0f}",
                (panel_x + 10, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_hip']['status']]
    cv2.putText(frame, f"R Hip: {angles['right_hip']:.0f}",
                (panel_x + 150, y_pos), font, font_scale, color, thickness)

    # Shoulder tilt
    y_pos += line_height
    color = COLORS[evals['shoulder_tilt']['status']]
    cv2.putText(frame, f"Shoulder Tilt: {angles['shoulder_tilt']:.1f}",
                (panel_x + 10, y_pos), font, font_scale, color, thickness)

    # Ankle angles
    y_pos += line_height
    color = COLORS[evals['left_ankle']['status']]
    cv2.putText(frame, f"L Ankle: {angles['left_ankle']:.0f}",
                (panel_x + 10, y_pos), font, font_scale, color, thickness)
    color = COLORS[evals['right_ankle']['status']]
    cv2.putText(frame, f"R Ankle: {angles['right_ankle']:.0f}",
                (panel_x + 150, y_pos), font, font_scale, color, thickness)

    # Score
    y_pos += line_height + 10
    score_color = COLORS['good'] if score >= 70 else (COLORS['warning'] if score >= 40 else COLORS['bad'])
    cv2.putText(frame, f"Score: {score}/100",
                (panel_x + 10, y_pos), font, 0.8, score_color, thickness)


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
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
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

    # YOLO model for person detection
    yolo_model_path = os.path.join(MODEL_DIR, YOLO_MODEL)
    yolo_model = YOLO(yolo_model_path)
    if USE_CUDA:
        yolo_model.to(DEVICE)

    # Tracker setup
    tracker_kwargs = {
        "max_age": 20,
        "n_init": 2,
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
            conf=0.5,
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

        # Process each bounding box for posture detection
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
                # Draw landmarks (RGB format)
                annotated_roi_rgb = draw_landmarks_on_image(roi_rgb, detection_result)
                # Convert back to BGR and place in frame
                annotated_roi_bgr = cv2.cvtColor(annotated_roi_rgb, cv2.COLOR_RGB2BGR)
                frame[y:y+h, x:x+w] = annotated_roi_bgr

                # Analyze ski pose
                roi_h, roi_w = roi_rgb.shape[:2]
                analysis = analyze_ski_pose(detection_result.pose_landmarks[0], roi_w, roi_h)
                if analysis:
                    latest_pose_analysis = analysis

            # Draw bounding box for visualization (blue)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Draw pose analysis info panel
        draw_info_panel(frame, latest_pose_analysis)

        # Write frame to output video
        out.write(frame)

        # Show preview window if GUI is enabled
        if SHOW_GUI:
            cv2.imshow("Ski Video - Pose & Tracking", frame)
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
