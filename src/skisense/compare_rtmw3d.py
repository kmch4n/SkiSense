"""Optional RTMW3D adapter for visual comparison of projected keypoints.

The RTMW3D wrapper's 3D coordinate convention has not been validated for
SkiSense's angle measurements. This adapter exposes only its 2D projection
to the shared drawing contract and intentionally returns no score.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .pose_topology import PoseTopology


WHOLEBODY_133 = PoseTopology(
    name="coco_wholebody_133_2d",
    num_landmarks=133,
    indices={
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
        "left_foot": 17,
        "right_foot": 20,
    },
    connections=[
        (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (15, 17), (15, 18), (15, 19),
        (16, 20), (16, 21), (16, 22),
    ],
    leg_indices=frozenset(range(11, 23)),
    lr_swap_pairs=[
        (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16),
        (17, 20), (18, 21), (19, 22),
    ],
    has_foot=True,
    is_3d=False,
)


@dataclass(frozen=True)
class _Landmark:
    x: float
    y: float
    visibility: float


class Rtmw3dComparisonBackend:
    """Return RTMW3D's projected whole-body pose without angle scoring."""

    display_name = "RTMW3D (projected 2D)"
    topology = WHOLEBODY_133

    def __init__(self, device: str = "cpu", model=None):
        self._device = device
        self._model = model

    def _get_model(self):
        if self._model is None:
            try:
                from rtmlib import Wholebody3d
            except ImportError as exc:
                raise RuntimeError(
                    "RTMW3D comparison requires rtmlib and onnxruntime-gpu "
                    "(or onnxruntime for CPU)."
                ) from exc
            self._model = Wholebody3d(
                backend="onnxruntime",
                device=self._device,
            )
        return self._model

    @staticmethod
    def _person_center(points: np.ndarray, scores: np.ndarray) -> Optional[np.ndarray]:
        torso = np.array([5, 6, 11, 12])
        valid = (scores[torso] > 0.25) & np.isfinite(points[torso]).all(axis=1)
        if np.any(valid):
            return points[torso][valid].mean(axis=0)
        body = np.arange(23)
        valid = (scores[body] > 0.25) & np.isfinite(points[body]).all(axis=1)
        if np.any(valid):
            return points[body][valid].mean(axis=0)
        return None

    def estimate(self, frame, bbox) -> Tuple[Optional[dict], None]:
        """Select the closest person and expose its full-frame 2D landmarks."""
        x, y, box_w, box_h = bbox
        if box_w <= 0 or box_h <= 0:
            return None, None

        height, width = frame.shape[:2]
        keypoints_3d, scores, _, keypoints_2d = self._get_model()(frame)
        points = np.asarray(keypoints_2d)
        confidence = np.asarray(scores)
        if (
            points.ndim != 3 or points.shape[1:] != (133, 2)
            or confidence.shape != points.shape[:2] or len(points) == 0
        ):
            return None, None

        target = np.array([x + box_w / 2, y + box_h / 2])
        candidates = [self._person_center(person, person_scores)
                      for person, person_scores in zip(points, confidence)]
        distances = [
            np.linalg.norm(center - target) if center is not None else np.inf
            for center in candidates
        ]
        selected = int(np.argmin(distances))
        if not np.isfinite(distances[selected]):
            return None, None
        center = candidates[selected]
        if not (
            x - box_w / 2 <= center[0] <= x + box_w * 1.5
            and y - box_h / 2 <= center[1] <= y + box_h * 1.5
        ):
            return None, None

        landmarks = [
            _Landmark(
                x=float(point[0] / width),
                y=float(point[1] / height),
                visibility=float(score) if np.isfinite(point).all() else 0.0,
            )
            for point, score in zip(points[selected], confidence[selected])
        ]
        entry = {
            "landmarks": landmarks,
            "bbox": (0, 0, width, height),
            "detection_bbox": tuple(bbox),
            "topology": WHOLEBODY_133,
            "coordinate_system": "rtmw3d_model_coordinates_unverified",
        }
        coordinates_3d = np.asarray(keypoints_3d)
        if coordinates_3d.shape == (len(points), 133, 3):
            entry["joints_3d"] = coordinates_3d[selected]
        return entry, None

    def close(self) -> None:
        """The rtmlib wrapper does not expose an explicit close operation."""
        self._model = None
