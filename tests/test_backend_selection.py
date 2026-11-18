"""The regular pipeline must select a usable pose model for each run."""

import unittest
from unittest.mock import patch

from src.skisense.backends import AVAILABLE_BACKENDS, get_backend, resolve_backend_name


class BackendSelectionTests(unittest.TestCase):
    def test_yolo26_is_available_and_resolves_when_configured(self):
        self.assertIn("yolo26", AVAILABLE_BACKENDS)
        with patch("src.skisense.config.POSE_BACKEND", "yolo26"):
            self.assertEqual(resolve_backend_name(), "yolo26")

    def test_yolo26_builds_with_its_own_checkpoint(self):
        with patch("src.skisense.backends.yolo11_backend.Yolo11Backend._build_model"):
            backend = get_backend(backend="yolo26")
        self.assertEqual(backend.model_filename, "yolo26x-pose.pt")
        self.assertIn("YOLO26", backend.display_name)

    def test_explicit_choice_overrides_default(self):
        with patch("src.skisense.backends.yolo11_backend.Yolo11Backend._build_model"):
            backend = get_backend(backend="yolo11")
        self.assertEqual(backend.model_filename, "yolo11x-pose.pt")


if __name__ == "__main__":
    unittest.main()
