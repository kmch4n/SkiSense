"""Pose topology definitions.

SkiSense uses SAM 3D Body as its pose estimator. SAM 3D Body emits MHR
(Momentum Human Rig) keypoints; SkiSense consumes the first 63 entries
covering body, feet, and wrists, providing the joints required for
ski-posture scoring (shoulders, hips, knees, ankles, toes) plus the
elbow→wrist arm segment used by the rendered overlay.

The legacy COCO-17 layout remains exported because pose_analyzer's
visibility-threshold helper and historical tests refer to it.
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
    is_3d: bool = False


# ---------------------------------------------------------------------------
# COCO-17 (legacy reference; YOLO11-Pose was the previous backend)
# ---------------------------------------------------------------------------
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
    is_3d=False,
)


# ---------------------------------------------------------------------------
# MHR_BODY (current runtime topology emitted by SAM 3D Body)
# ---------------------------------------------------------------------------
# SAM 3D Body returns the first 70 MHR keypoints. SkiSense keeps the first
# 63 because wrists are at indices 41/62; entries 21–40 and 42–61 are
# hand-finger joints, allocated as unused shims so they do not affect any
# computation. Index layout (matches mhr70 metadata):
#
#   0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
#   5:left_shoulder 6:right_shoulder
#   7:left_elbow    8:right_elbow
#   9:left_hip      10:right_hip
#   11:left_knee    12:right_knee
#   13:left_ankle   14:right_ankle
#   15:left_big_toe  16:left_small_toe   17:left_heel
#   18:right_big_toe 19:right_small_toe  20:right_heel
#   41:right_wrist  62:left_wrist
_MHR_BODY_CONNECTIONS: List[Tuple[int, int]] = [
    # Legs
    (9, 11), (11, 13),
    (10, 12), (12, 14),
    # Feet (ankle -> toes/heel)
    (13, 15), (13, 16), (13, 17),
    (14, 18), (14, 19), (14, 20),
    # Torso
    (5, 9), (6, 10), (5, 6), (9, 10),
    # Arms (shoulder -> elbow -> wrist)
    (5, 7), (7, 62),
    (6, 8), (8, 41),
    # Head
    (0, 1), (0, 2), (1, 3), (2, 4),
]

MHR_BODY = PoseTopology(
    name="mhr_body",
    num_landmarks=63,
    indices={
        "left_shoulder": 5, "right_shoulder": 6,
        "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 62, "right_wrist": 41,
        "left_hip": 9, "right_hip": 10,
        "left_knee": 11, "right_knee": 12,
        "left_ankle": 13, "right_ankle": 14,
        # ankle-angle requires a foot landmark; big-toe tip mirrors the
        # role of "foot" used by analyze_ski_pose.
        "left_foot": 15, "right_foot": 18,
    },
    connections=_MHR_BODY_CONNECTIONS,
    leg_indices=frozenset({9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}),
    lr_swap_pairs=[
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14),
        (15, 18), (16, 19), (17, 20),
        (41, 62),
    ],
    has_foot=True,
    is_3d=True,
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
