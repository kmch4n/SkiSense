"""Behavioral checks for the pose model comparison workflow."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.skisense.pose_comparison import (
    draw_pose,
    frame_indices,
    select_target_bbox,
    serialize_pose,
)
from src.skisense.pose_topology import COCO_17
from src.skisense.backends.yolo11_backend import Yolo11Backend


class Landmark:
    def __init__(self, x, y, visibility):
        self.x = x
        self.y = y
        self.visibility = visibility


class PoseComparisonTests(unittest.TestCase):
    def test_yolo_backend_accepts_an_alternate_pose_checkpoint(self):
        with patch.object(Yolo11Backend, "_build_model", return_value=object()):
            backend = Yolo11Backend(model_filename="yolo26x-pose.pt")
        self.assertEqual(backend.model_filename, "yolo26x-pose.pt")

    def test_yolo_roi_pose_matches_requested_person_in_a_crowd(self):
        class Model:
            def __call__(self, roi, **kwargs):
                class Keypoints:
                    xyn = torch.tensor([[[0.1, 0.1]] * 17,
                                        [[0.8, 0.8]] * 17])
                    conf = torch.ones((2, 17))

                class Boxes:
                    xyxy = torch.tensor([[0, 0, 30, 30], [70, 70, 100, 100]])
                    conf = torch.tensor([0.95, 0.7])

                class Result:
                    keypoints = Keypoints()
                    boxes = Boxes()

                return [Result()]

        backend = Yolo11Backend.__new__(Yolo11Backend)
        backend._model = Model()
        backend._yolo_device = "cpu"
        backend._yolo_half = False
        landmarks = backend._infer_landmarks(
            np.zeros((100, 100, 3), dtype=np.uint8),
            target_box=(65, 65, 35, 35),
        )
        self.assertAlmostEqual(landmarks[0].x, 0.8)

    def test_frame_indices_include_evenly_spaced_interior_frames(self):
        self.assertEqual(frame_indices(100, 3), [25, 50, 75])
        self.assertEqual(frame_indices(1, 3), [0])

    def test_select_target_bbox_uses_largest_person(self):
        self.assertEqual(
            select_target_bbox([(0, 0, 10, 10), (5, 5, 20, 10)]),
            (5, 5, 20, 10),
        )
        self.assertIsNone(select_target_bbox([]))

    def test_serialize_pose_preserves_image_coordinates_and_visibility(self):
        entry = {
            "bbox": (10, 20, 100, 80),
            "topology": COCO_17,
            "landmarks": [Landmark(0.25, 0.5, 0.9)],
        }
        result = serialize_pose(entry)
        self.assertEqual(result["topology"], "coco17")
        self.assertEqual(result["landmarks"][0]["x"], 35.0)
        self.assertEqual(result["landmarks"][0]["y"], 60.0)
        self.assertEqual(result["landmarks"][0]["visibility"], 0.9)

    def test_nonfinite_landmark_is_safe_to_save_and_draw(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        entry = {
            "bbox": (0, 0, 100, 100),
            "topology": COCO_17,
            "landmarks": [Landmark(float("nan"), 0.5, 1.0)],
        }
        result = serialize_pose(entry)
        self.assertIsNone(result["landmarks"][0]["x"])
        self.assertEqual(int(draw_pose(frame, entry).sum()), 0)

    def test_serialize_pose_keeps_3d_points_separate_from_2d_projection(self):
        entry = {
            "bbox": (0, 0, 100, 100),
            "topology": COCO_17,
            "landmarks": [Landmark(0.1, 0.2, 1.0)],
            "joints_3d": np.array([[1.0, 2.0, 3.0]]),
            "coordinate_system": "model_coordinates_unverified",
        }
        result = serialize_pose(entry)
        self.assertEqual(result["joints_3d"], [[1.0, 2.0, 3.0]])
        self.assertEqual(result["coordinate_system"], "model_coordinates_unverified")

    def test_draw_pose_changes_frame_for_visible_skeleton(self):
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        landmarks = [Landmark(0.5, 0.5, 0.0) for _ in range(17)]
        landmarks[5] = Landmark(0.2, 0.2, 0.9)
        landmarks[7] = Landmark(0.5, 0.5, 0.9)
        entry = {"bbox": (0, 0, 200, 200), "topology": COCO_17, "landmarks": landmarks}
        result = draw_pose(frame, entry)
        self.assertGreater(int(result.sum()), 0)
        self.assertEqual(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
