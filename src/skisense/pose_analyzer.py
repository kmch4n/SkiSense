import numpy as np
import math

# MediaPipe Pose Landmark indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26
LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32

# Landmarks that belong to the lower body and are commonly occluded in ski poses.
# These use a looser visibility threshold when evaluating joint angles.
LEG_LANDMARK_INDICES = frozenset({
    LEFT_HIP, RIGHT_HIP,
    LEFT_KNEE, RIGHT_KNEE,
    LEFT_ANKLE, RIGHT_ANKLE,
    29, 30,  # left/right heel
    LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX,
})

# Color definitions (BGR format)
COLORS = {
    'good': (0, 255, 0),      # Green
    'warning': (0, 255, 255), # Yellow
    'bad': (0, 0, 255),       # Red
    'info': (255, 255, 255)   # White
}


def calculate_angle(a, b, c):
    """
    Calculate angle at point b formed by points a-b-c

    Parameters:
        a: Point A (x, y)
        b: Point B (x, y) - vertex
        c: Point C (x, y)

    Returns:
        Angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Handle zero-length vectors
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def calculate_horizontal_angle(p1, p2):
    """
    Calculate angle between line p1-p2 and horizontal

    Returns:
        Angle in degrees (positive = right side higher)
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))


def get_landmark_point(landmarks, index, width, height):
    """
    Get pixel coordinates from landmark

    Parameters:
        landmarks: List of pose landmarks
        index: Landmark index
        width: Image width
        height: Image height

    Returns:
        (x, y) pixel coordinates
    """
    landmark = landmarks[index]
    return (int(landmark.x * width), int(landmark.y * height))


def is_landmark_visible(landmarks, index: int, threshold: float) -> bool:
    """Check if a landmark meets the visibility threshold."""
    return landmarks[index].visibility >= threshold


def evaluate_knee_angle(angle):
    """
    Evaluate knee bend angle

    Good skiing posture: 90-120 degrees
    """
    if 90 <= angle <= 120:
        return 'good', "Good"
    elif 80 <= angle < 90 or 120 < angle <= 140:
        return 'warning', "Caution"
    else:
        return 'bad', "Fix"


def evaluate_hip_angle(angle):
    """
    Evaluate hip angle (forward lean)

    Good skiing posture: 100-130 degrees
    """
    if 100 <= angle <= 130:
        return 'good', "Good"
    elif 90 <= angle < 100 or 130 < angle <= 150:
        return 'warning', "Caution"
    else:
        return 'bad', "Fix"


def evaluate_shoulder_tilt(angle):
    """
    Evaluate shoulder tilt

    Good balance: 0-10 degrees
    """
    abs_angle = abs(angle)
    if abs_angle <= 10:
        return 'good', "Good"
    elif abs_angle <= 20:
        return 'warning', "Tilted"
    else:
        return 'bad', "Over-tilt"


def evaluate_ankle_angle(angle):
    """
    Evaluate ankle angle (forward pressure on boots)

    Good skiing posture: 70-90 degrees
    """
    if 70 <= angle <= 90:
        return 'good', "Good"
    elif 90 < angle <= 110:
        return 'warning', "Caution"
    else:
        return 'bad', "Fix"


def analyze_ski_pose(
    landmarks,
    width,
    height,
    visibility_threshold: float = 0.5,
    visibility_threshold_legs: float = None,
):
    """
    Analyze ski posture and return angle measurements

    Parameters:
        landmarks: List of pose landmarks from MediaPipe
        width: ROI width
        height: ROI height
        visibility_threshold: Minimum visibility for upper-body landmarks
        visibility_threshold_legs: Minimum visibility for leg landmarks.
            Falls back to ``visibility_threshold`` when None.

    Returns:
        dict with analysis results
    """
    if len(landmarks) < 33:
        return None

    if visibility_threshold_legs is None:
        visibility_threshold_legs = visibility_threshold

    results = {
        'angles': {},
        'evaluations': {},
        'score': 0
    }

    try:
        # Get landmark points
        left_shoulder = get_landmark_point(landmarks, LEFT_SHOULDER, width, height)
        right_shoulder = get_landmark_point(landmarks, RIGHT_SHOULDER, width, height)
        left_hip = get_landmark_point(landmarks, LEFT_HIP, width, height)
        right_hip = get_landmark_point(landmarks, RIGHT_HIP, width, height)
        left_knee = get_landmark_point(landmarks, LEFT_KNEE, width, height)
        right_knee = get_landmark_point(landmarks, RIGHT_KNEE, width, height)
        left_ankle = get_landmark_point(landmarks, LEFT_ANKLE, width, height)
        right_ankle = get_landmark_point(landmarks, RIGHT_ANKLE, width, height)
        left_foot = get_landmark_point(landmarks, LEFT_FOOT_INDEX, width, height)
        right_foot = get_landmark_point(landmarks, RIGHT_FOOT_INDEX, width, height)

        # Helper to check visibility using a looser threshold on leg landmarks.
        def _visible(*indices: int) -> bool:
            for i in indices:
                threshold = (
                    visibility_threshold_legs if i in LEG_LANDMARK_INDICES
                    else visibility_threshold
                )
                if not is_landmark_visible(landmarks, i, threshold):
                    return False
            return True

        # Calculate angles (only when all required landmarks are visible)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle) if _visible(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE) else None
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle) if _visible(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE) else None
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee) if _visible(LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE) else None
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee) if _visible(RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE) else None
        shoulder_tilt = calculate_horizontal_angle(left_shoulder, right_shoulder) if _visible(LEFT_SHOULDER, RIGHT_SHOULDER) else None
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot) if _visible(LEFT_KNEE, LEFT_ANKLE, LEFT_FOOT_INDEX) else None
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot) if _visible(RIGHT_KNEE, RIGHT_ANKLE, RIGHT_FOOT_INDEX) else None

        # Store angles (use 0.0 for display when not available)
        results['angles'] = {
            'left_knee': left_knee_angle if left_knee_angle is not None else 0.0,
            'right_knee': right_knee_angle if right_knee_angle is not None else 0.0,
            'left_hip': left_hip_angle if left_hip_angle is not None else 0.0,
            'right_hip': right_hip_angle if right_hip_angle is not None else 0.0,
            'shoulder_tilt': shoulder_tilt if shoulder_tilt is not None else 0.0,
            'left_ankle': left_ankle_angle if left_ankle_angle is not None else 0.0,
            'right_ankle': right_ankle_angle if right_ankle_angle is not None else 0.0
        }

        # Evaluate each angle (skip from scoring if landmarks not visible)
        evaluations = {}
        score_total = 0
        score_count = 0

        # Knee evaluations
        if left_knee_angle is not None:
            status, label = evaluate_knee_angle(left_knee_angle)
            evaluations['left_knee'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['left_knee'] = {'status': 'info', 'label': 'N/A'}

        if right_knee_angle is not None:
            status, label = evaluate_knee_angle(right_knee_angle)
            evaluations['right_knee'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['right_knee'] = {'status': 'info', 'label': 'N/A'}

        # Hip evaluations
        if left_hip_angle is not None:
            status, label = evaluate_hip_angle(left_hip_angle)
            evaluations['left_hip'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['left_hip'] = {'status': 'info', 'label': 'N/A'}

        if right_hip_angle is not None:
            status, label = evaluate_hip_angle(right_hip_angle)
            evaluations['right_hip'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['right_hip'] = {'status': 'info', 'label': 'N/A'}

        # Shoulder tilt evaluation
        if shoulder_tilt is not None:
            status, label = evaluate_shoulder_tilt(shoulder_tilt)
            evaluations['shoulder_tilt'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['shoulder_tilt'] = {'status': 'info', 'label': 'N/A'}

        # Ankle evaluations
        if left_ankle_angle is not None:
            status, label = evaluate_ankle_angle(left_ankle_angle)
            evaluations['left_ankle'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['left_ankle'] = {'status': 'info', 'label': 'N/A'}

        if right_ankle_angle is not None:
            status, label = evaluate_ankle_angle(right_ankle_angle)
            evaluations['right_ankle'] = {'status': status, 'label': label}
            score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
            score_count += 1
        else:
            evaluations['right_ankle'] = {'status': 'info', 'label': 'N/A'}

        results['evaluations'] = evaluations
        results['score'] = int(score_total / score_count) if score_count > 0 else 0

        # Calculate shoulder center (for zoom centering)
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2
        )
        results['shoulder_center'] = shoulder_center

    except Exception as e:
        return None

    return results
