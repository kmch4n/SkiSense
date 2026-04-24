"""Pose topology definitions for different pose estimation backends.

Each backend emits keypoints in a specific topology (MediaPipe 33 vs COCO 17).
This module provides a uniform representation so the rest of the pipeline can
operate topology-agnostically: evaluation, drawing, and flip TTA each consult
the topology rather than hard-coding landmark indices.
"""
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple


@dataclass(frozen=True)
class PoseTopology:
    """Describes the landmark layout produced by a pose estimation backend."""

    name: str                                   # "mediapipe" | "coco17"
    num_landmarks: int
    indices: Dict[str, int]                     # joint name -> landmark index
    connections: List[Tuple[int, int]]          # skeleton edges for drawing
    leg_indices: FrozenSet[int]                 # indices that use the leg visibility threshold
    lr_swap_pairs: List[Tuple[int, int]]        # left/right landmark pairs for flip TTA
    has_foot: bool                              # whether foot landmarks are available


# MediaPipe Pose Landmarker skeleton edges (33 landmarks).
# Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_MEDIAPIPE_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]

MEDIAPIPE_33 = PoseTopology(
    name="mediapipe",
    num_landmarks=33,
    indices={
        "left_shoulder": 11, "right_shoulder": 12,
        "left_hip": 23, "right_hip": 24,
        "left_knee": 25, "right_knee": 26,
        "left_ankle": 27, "right_ankle": 28,
        "left_foot": 31, "right_foot": 32,
    },
    connections=_MEDIAPIPE_CONNECTIONS,
    leg_indices=frozenset({23, 24, 25, 26, 27, 28, 29, 30, 31, 32}),
    lr_swap_pairs=[
        (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
        (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
    ],
    has_foot=True,
)


# COCO-17 skeleton, emitted by YOLO11-Pose, ViTPose, Sapiens, and most modern
# pose estimators. No distinct foot_index landmark exists; ankle-angle scoring
# is disabled automatically when ``has_foot`` is False.
#   0:nose   1:left_eye        2:right_eye
#   3:left_ear  4:right_ear    5:left_shoulder  6:right_shoulder
#   7:left_elbow  8:right_elbow  9:left_wrist   10:right_wrist
#   11:left_hip  12:right_hip   13:left_knee    14:right_knee
#   15:left_ankle  16:right_ankle
_COCO_17_CONNECTIONS: List[Tuple[int, int]] = [
    (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (5, 6), (5, 11), (6, 12), (11, 12),         # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
    (0, 1), (0, 2), (1, 3), (2, 4),             # head
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
    """Return the visibility threshold that applies to a landmark index.

    Leg landmarks (hip/knee/ankle/foot) use ``leg_threshold``; all others
    use ``upper_threshold``. Matches the two-tier threshold model used by
    ``analyze_ski_pose`` and the skeleton drawing helpers.
    """
    return leg_threshold if index in topology.leg_indices else upper_threshold


def build_flip_swap_table(topology: PoseTopology) -> List[int]:
    """Build an index map so ``table[i]`` returns the index whose flipped
    counterpart ends up at position ``i`` in the original orientation.

    Used when merging pose landmarks from a horizontally flipped inference
    pass back into the original orientation.
    """
    table = list(range(topology.num_landmarks))
    for a, b in topology.lr_swap_pairs:
        table[a], table[b] = b, a
    return table
