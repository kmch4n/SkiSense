"""Image preprocessing helpers shared across pose estimation backends.

Kept outside the backends so both can apply the same preprocessing pipeline
for apples-to-apples comparison.
"""
import cv2


def apply_clahe(bgr_image, clip_limit: float = 2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE on the L channel of a BGR image and return a BGR image.

    Boosts local contrast without distorting colour balance. Opt-in because
    validation showed it often amplifies snow noise and hurts MediaPipe
    accuracy on ski footage.
    """
    lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)
    lab = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
