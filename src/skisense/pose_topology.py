"""Pose topology definitions.

SkiSense now uses YOLO11-Pose as its pose estimator. YOLO11-Pose emits
COCO-17 keypoints, which do not include a separate foot landmark. The analyzer
therefore skips ankle-angle scoring and evaluates the joints that COCO-17 can
represent reliably.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


@dataclass(frozen=True)
class PoseTopology:
    """Describes the landmark layout produced by a pose estimation backend."""

    name: str
    num_landmarks: int
    indices: Dict[str, int]
    connections: List[Tuple[int, int]]
    leg_indices: FrozenSet[int]
    lr_swap_pairs: List[Tuple[int, int]]
    has_foot: bool


# COCO-17 skeleton emitted by YOLO11-Pose.
#   0:nose   1:left_eye        2:right_eye
#   3:left_ear  4:right_ear    5:left_shoulder  6:right_shoulder
#   7:left_elbow  8:right_elbow  9:left_wrist   10:right_wrist
#   11:left_hip  12:right_hip   13:left_knee    14:right_knee
#   15:left_ankle  16:right_ankle
_COCO_17_CONNECTIONS: List[Tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

COCO_17 = PoseTopology(
    name="coco17",
    num_landmarks=17,
    indices={
        "left_shoulder": 5, "right_shoulder": 6,
        "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14,
        "left_ankle": 15, "right_ankle": 16,
    },
    connections=_COCO_17_CONNECTIONS,
    leg_indices=frozenset({11, 12, 13, 14, 15, 16}),
    lr_swap_pairs=[
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16),
    ],
    has_foot=False,
)


def visibility_threshold_for(
    index: int,
    topology: PoseTopology,
    upper_threshold: float,
    leg_threshold: float,
) -> float:
    """Return the visibility threshold that applies to a landmark index."""
    return leg_threshold if index in topology.leg_indices else upper_threshold


def build_flip_swap_table(topology: PoseTopology) -> List[int]:
    """Build an index map for restoring horizontally flipped keypoints."""
    table = list(range(topology.num_landmarks))
    for a, b in topology.lr_swap_pairs:
        table[a], table[b] = b, a
    return table
