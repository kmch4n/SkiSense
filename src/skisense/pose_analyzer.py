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


def analyze_ski_pose(landmarks, width, height):
    """
    Analyze ski posture and return angle measurements

    Parameters:
        landmarks: List of pose landmarks from MediaPipe
        width: ROI width
        height: ROI height

    Returns:
        dict with analysis results
    """
    if len(landmarks) < 33:
        return None

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

        # Calculate knee angles (hip -> knee -> ankle)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Calculate hip angles (shoulder -> hip -> knee)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

        # Calculate shoulder tilt
        shoulder_tilt = calculate_horizontal_angle(left_shoulder, right_shoulder)

        # Calculate ankle angles (knee -> ankle -> foot)
        left_ankle_angle = calculate_angle(left_knee, left_ankle, left_foot)
        right_ankle_angle = calculate_angle(right_knee, right_ankle, right_foot)

        # Store angles
        results['angles'] = {
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle,
            'left_hip': left_hip_angle,
            'right_hip': right_hip_angle,
            'shoulder_tilt': shoulder_tilt,
            'left_ankle': left_ankle_angle,
            'right_ankle': right_ankle_angle
        }

        # Evaluate each angle
        evaluations = {}
        score_total = 0
        score_count = 0

        # Knee evaluations
        status, label = evaluate_knee_angle(left_knee_angle)
        evaluations['left_knee'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        status, label = evaluate_knee_angle(right_knee_angle)
        evaluations['right_knee'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        # Hip evaluations
        status, label = evaluate_hip_angle(left_hip_angle)
        evaluations['left_hip'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        status, label = evaluate_hip_angle(right_hip_angle)
        evaluations['right_hip'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        # Shoulder tilt evaluation
        status, label = evaluate_shoulder_tilt(shoulder_tilt)
        evaluations['shoulder_tilt'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        # Ankle evaluations
        status, label = evaluate_ankle_angle(left_ankle_angle)
        evaluations['left_ankle'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        status, label = evaluate_ankle_angle(right_ankle_angle)
        evaluations['right_ankle'] = {'status': status, 'label': label}
        score_total += 100 if status == 'good' else (50 if status == 'warning' else 0)
        score_count += 1

        results['evaluations'] = evaluations
        results['score'] = int(score_total / score_count) if score_count > 0 else 0

    except Exception as e:
        return None

    return results
