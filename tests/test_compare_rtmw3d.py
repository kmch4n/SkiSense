"""Focused checks for the optional RTMW3D comparison adapter."""

import unittest

import numpy as np

from src.skisense.compare_rtmw3d import Rtmw3dComparisonBackend


class FakeWholebody3d:
    def __call__(self, frame):
        points_2d = np.zeros((2, 133, 2), dtype=float)
        scores = np.ones((2, 133), dtype=float)
        points_2d[0, :, :] = (10.0, 10.0)
        points_2d[1, :, :] = (80.0, 60.0)
        points_2d[1, 17, :] = (88.0, 80.0)
        return np.zeros((2, 133, 3)), scores, None, points_2d


class Rtmw3dComparisonBackendTests(unittest.TestCase):
    def test_selects_person_near_target_and_exposes_2d_feet(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        backend = Rtmw3dComparisonBackend(model=FakeWholebody3d())

        entry, analysis = backend.estimate(frame, (70, 50, 20, 20))

        self.assertIsNone(analysis)
        self.assertEqual(entry["bbox"], (0, 0, 100, 100))
        self.assertEqual(entry["detection_bbox"], (70, 50, 20, 20))
        self.assertEqual(len(entry["landmarks"]), 133)
        self.assertAlmostEqual(entry["landmarks"][17].x, 0.88)
        self.assertAlmostEqual(entry["landmarks"][17].y, 0.80)
        self.assertTrue(entry["topology"].has_foot)
        self.assertFalse(entry["topology"].is_3d)
        self.assertEqual(entry["topology"].indices["left_foot"], 17)
        self.assertEqual(entry["joints_3d"].shape, (133, 3))

    def test_rejects_invalid_bbox_without_running_model(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        backend = Rtmw3dComparisonBackend(model=FakeWholebody3d())

        self.assertEqual(backend.estimate(frame, (0, 0, 0, 20)), (None, None))

    def test_rejects_a_person_far_from_the_requested_box(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        backend = Rtmw3dComparisonBackend(model=FakeWholebody3d())
        self.assertEqual(backend.estimate(frame, (40, 40, 10, 10)), (None, None))


if __name__ == "__main__":
    unittest.main()
