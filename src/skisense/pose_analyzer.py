import math

import numpy as np

from .pose_topology import MEDIAPIPE_33, PoseTopology

# MediaPipe Pose Landmark indices (kept for backward compatibility; derived
# from MEDIAPIPE_33 so the authoritative source remains pose_topology.py)
LEFT_SHOULDER = MEDIAPIPE_33.indices["left_shoulder"]
RIGHT_SHOULDER = MEDIAPIPE_33.indices["right_shoulder"]
LEFT_HIP = MEDIAPIPE_33.indices["left_hip"]
RIGHT_HIP = MEDIAPIPE_33.indices["right_hip"]
LEFT_KNEE = MEDIAPIPE_33.indices["left_knee"]
RIGHT_KNEE = MEDIAPIPE_33.indices["right_knee"]
LEFT_ANKLE = MEDIAPIPE_33.indices["left_ankle"]
RIGHT_ANKLE = MEDIAPIPE_33.indices["right_ankle"]
LEFT_FOOT_INDEX = MEDIAPIPE_33.indices["left_foot"]
RIGHT_FOOT_INDEX = MEDIAPIPE_33.indices["right_foot"]

# Landmarks commonly occluded in ski poses; re-exported for callers that
# import this constant directly.
LEG_LANDMARK_INDICES = MEDIAPIPE_33.leg_indices

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
    topology: PoseTopology = MEDIAPIPE_33,
):
    """
    Analyze ski posture and return angle measurements.

    Parameters:
        landmarks: List of pose landmarks from a backend
        width: ROI width
        height: ROI height
        visibility_threshold: Minimum visibility for upper-body landmarks
        visibility_threshold_legs: Minimum visibility for leg landmarks.
            Falls back to ``visibility_threshold`` when None.
        topology: Pose topology describing the landmark layout. Determines
            which indices to sample and whether ankle angle evaluation is
            possible (requires ``topology.has_foot``).

    Returns:
        dict with angles / evaluations / score / shoulder_center
    """
    if len(landmarks) < topology.num_landmarks:
        return None

    if visibility_threshold_legs is None:
        visibility_threshold_legs = visibility_threshold

    idx = topology.indices
    leg_indices = topology.leg_indices

    results = {
        'angles': {},
        'evaluations': {},
        'score': 0,
    }

    try:
        left_shoulder = get_landmark_point(landmarks, idx["left_shoulder"], width, height)
        right_shoulder = get_landmark_point(landmarks, idx["right_shoulder"], width, height)
        left_hip = get_landmark_point(landmarks, idx["left_hip"], width, height)
        right_hip = get_landmark_point(landmarks, idx["right_hip"], width, height)
        left_knee = get_landmark_point(landmarks, idx["left_knee"], width, height)
        right_knee = get_landmark_point(landmarks, idx["right_knee"], width, height)
        left_ankle = get_landmark_point(landmarks, idx["left_ankle"], width, height)
        right_ankle = get_landmark_point(landmarks, idx["right_ankle"], width, height)

        if topology.has_foot:
            left_foot = get_landmark_point(landmarks, idx["left_foot"], width, height)
            right_foot = get_landmark_point(landmarks, idx["right_foot"], width, height)
        else:
            left_foot = None
            right_foot = None

        def _visible(*indices: int) -> bool:
            for i in indices:
                threshold = (
                    visibility_threshold_legs if i in leg_indices
                    else visibility_threshold
                )
                if not is_landmark_visible(landmarks, i, threshold):
                    return False
            return True

        left_knee_angle = (
            calculate_angle(left_hip, left_knee, left_ankle)
            if _visible(idx["left_hip"], idx["left_knee"], idx["left_ankle"]) else None
        )
        right_knee_angle = (
            calculate_angle(right_hip, right_knee, right_ankle)
            if _visible(idx["right_hip"], idx["right_knee"], idx["right_ankle"]) else None
        )
        left_hip_angle = (
            calculate_angle(left_shoulder, left_hip, left_knee)
            if _visible(idx["left_shoulder"], idx["left_hip"], idx["left_knee"]) else None
        )
        right_hip_angle = (
            calculate_angle(right_shoulder, right_hip, right_knee)
            if _visible(idx["right_shoulder"], idx["right_hip"], idx["right_knee"]) else None
        )
        shoulder_tilt = (
            calculate_horizontal_angle(left_shoulder, right_shoulder)
            if _visible(idx["left_shoulder"], idx["right_shoulder"]) else None
        )

        if topology.has_foot:
            left_ankle_angle = (
                calculate_angle(left_knee, left_ankle, left_foot)
                if _visible(idx["left_knee"], idx["left_ankle"], idx["left_foot"]) else None
            )
            right_ankle_angle = (
                calculate_angle(right_knee, right_ankle, right_foot)
                if _visible(idx["right_knee"], idx["right_ankle"], idx["right_foot"]) else None
            )
        else:
            # Topology has no foot landmark; ankle-angle scoring is unavailable.
            left_ankle_angle = None
            right_ankle_angle = None

        results['angles'] = {
            'left_knee': left_knee_angle if left_knee_angle is not None else 0.0,
            'right_knee': right_knee_angle if right_knee_angle is not None else 0.0,
            'left_hip': left_hip_angle if left_hip_angle is not None else 0.0,
            'right_hip': right_hip_angle if right_hip_angle is not None else 0.0,
            'shoulder_tilt': shoulder_tilt if shoulder_tilt is not None else 0.0,
            'left_ankle': left_ankle_angle if left_ankle_angle is not None else 0.0,
            'right_ankle': right_ankle_angle if right_ankle_angle is not None else 0.0,
        }

        evaluations = {}
        score_total = 0
        score_count = 0

        def _grade(angle_value, evaluator_fn, key):
            nonlocal score_total, score_count
            if angle_value is not None:
                status, label = evaluator_fn(angle_value)
                evaluations[key] = {'status': status, 'label': label}
                score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
                score_count += 1
            else:
                evaluations[key] = {'status': 'info', 'label': 'N/A'}

        _grade(left_knee_angle, evaluate_knee_angle, 'left_knee')
        _grade(right_knee_angle, evaluate_knee_angle, 'right_knee')
        _grade(left_hip_angle, evaluate_hip_angle, 'left_hip')
        _grade(right_hip_angle, evaluate_hip_angle, 'right_hip')
        _grade(shoulder_tilt, evaluate_shoulder_tilt, 'shoulder_tilt')
        _grade(left_ankle_angle, evaluate_ankle_angle, 'left_ankle')
        _grade(right_ankle_angle, evaluate_ankle_angle, 'right_ankle')

        results['evaluations'] = evaluations
        results['score'] = int(score_total / score_count) if score_count > 0 else 0

        # Shoulder center is used for zoom centering
        shoulder_center = (
            (left_shoulder[0] + right_shoulder[0]) / 2,
            (left_shoulder[1] + right_shoulder[1]) / 2,
        )
        results['shoulder_center'] = shoulder_center

    except Exception:
        return None

    return results
