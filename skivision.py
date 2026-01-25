import os
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from noise_filter import filter_noise_rectangles
from pose_analyzer import analyze_ski_pose, COLORS
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

# Setup MediaPipe Pose Landmarker (Tasks API)
MODEL_PATH = "pose_landmarker.task"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    import urllib.request
    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    print("Downloading pose landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully.")

# PoseLandmarker configuration
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    min_pose_detection_confidence=0.1,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    num_poses=1
)
pose_landmarker = vision.PoseLandmarker.create_from_options(options)


# Pose connections (33 landmarks)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32)
]


def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw pose landmarks on image using OpenCV"""
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


# Store latest pose analysis results for display
latest_pose_analysis = None


def draw_info_panel(frame, analysis):
    """Draw pose analysis info panel on top-left of frame"""
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


# Input video path
video_path = "video/your_video_file_here.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties for output
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define output filename: same as input with '_pose' appended before the extension
base_name = os.path.splitext(os.path.basename(video_path))[0]
output_filename = f"{base_name}_pose.mp4"
output_path = os.path.join(os.path.dirname(video_path), output_filename)

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Background subtractor and tracker setup
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)
tracker = DeepSort(
    max_age=20,
    n_init=2,
    nms_max_overlap=1.0,
    embedder='mobilenet',
    half=True,
    bgr=True
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Create foreground mask and reduce noise with morphological operations
    fgmask = fgbg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    # Find contours on the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 30 or h < 50:
                continue
            rects.append((x, y, w, h))

    # Filter out detections in the noisy zones
    rects = filter_noise_rectangles(frame, rects, left_ignore_pct=0.2, right_ignore_pct=0.1, top_ignore_pct=0.1)

    # Deep SORT用フォーマットに変換: ([x, y, w, h], confidence, class)
    detections = [([x, y, w, h], 1.0, 'person') for (x, y, w, h) in rects]

    # Deep SORTでトラッキング更新
    tracks = tracker.update_tracks(detections, frame=frame)

    # 確認済みトラックから重心辞書を作成
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

    # Show the frame and write it to the output video
    cv2.imshow("Ski Video - Pose & Tracking", frame)
    out.write(frame)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
pose_landmarker.close()
cv2.destroyAllWindows()
