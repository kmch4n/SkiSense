"""MediaPipe Pose Landmarker backend.

Wraps the Google MediaPipe Tasks API so it conforms to the shared
``PoseBackend`` interface. Owns model download, tempdir copy (needed to
avoid MediaPipe's non-ASCII path issue on Windows), per-frame inference,
and the optional horizontal-flip TTA merge used for heavily occluded
limbs.
"""
import atexit
import os
import shutil
import tempfile
from typing import Optional, Tuple

from .._logging import SuppressStderr

# SuppressStderr also runs the absl / TF env-var setup on first import.
# Wrap MediaPipe import to silence its C++ startup chatter in non-debug runs.
with SuppressStderr():
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks_python
    from mediapipe.tasks.python import vision

from ..config import (
    CLAHE_ENABLED,
    FLIP_TTA_ENABLED,
    MODEL_DIR,
    POSE_DETECTION_CONFIDENCE,
    POSE_MODEL,
    POSE_MODEL_URL,
    POSE_PRESENCE_CONF,
    POSE_TRACKING_CONF,
    POSE_VISIBILITY_THRESHOLD,
    POSE_VISIBILITY_THRESHOLD_LEGS,
    ROI_PADDING_RATIO,
)
from ..pose_analyzer import analyze_ski_pose
from ..pose_topology import MEDIAPIPE_33, build_flip_swap_table
from ..preprocessing import apply_clahe
from .base import PoseBackend


def _running_mode(name: str):
    """Translate a string running-mode into the MediaPipe enum."""
    if name == "video":
        return vision.RunningMode.VIDEO
    if name == "image":
        return vision.RunningMode.IMAGE
    raise ValueError(f"Unknown running_mode: {name!r}")


class _MergedLandmark:
    """Lightweight shim mirroring ``NormalizedLandmark`` attributes."""
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x, y, z, visibility, presence):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class MediaPipeBackend(PoseBackend):
    """MediaPipe Pose Landmarker backend (33 landmarks)."""

    topology = MEDIAPIPE_33

    def __init__(self, running_mode: str):
        """Build the underlying ``PoseLandmarker``.

        Args:
            running_mode: ``"video"`` or ``"image"``. Video mode is stateful
                and requires monotonic ``timestamp_ms`` values on each call.
        """
        self._running_mode_name = running_mode
        self._mp_running_mode = _running_mode(running_mode)
        self._flip_swap_table = build_flip_swap_table(MEDIAPIPE_33)
        self._landmarker = self._build_landmarker()

    # ------------------------------------------------------------------
    # Model setup

    def _build_landmarker(self):
        model_path = os.path.join(MODEL_DIR, POSE_MODEL)

        if not os.path.exists(model_path):
            import urllib.request
            os.makedirs(MODEL_DIR, exist_ok=True)
            urllib.request.urlretrieve(POSE_MODEL_URL, model_path)

        # Copy to a temp dir to sidestep MediaPipe's non-ASCII path bug on Windows.
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, POSE_MODEL)
        shutil.copy2(model_path, temp_model_path)
        atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

        base_options = mp_tasks_python.BaseOptions(model_asset_path=temp_model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=self._mp_running_mode,
            min_pose_detection_confidence=POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=POSE_PRESENCE_CONF,
            min_tracking_confidence=POSE_TRACKING_CONF,
            num_poses=1,
        )

        with SuppressStderr():
            return vision.PoseLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API

    def estimate(
        self,
        frame,
        bbox,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        x, y, w, h = bbox
        pad_w = int(w * ROI_PADDING_RATIO)
        pad_h = int(h * ROI_PADDING_RATIO)
        frame_h, frame_w = frame.shape[:2]
        px = max(0, x - pad_w)
        py = max(0, y - pad_h)
        pw = min(frame_w, x + w + pad_w) - px
        ph = min(frame_h, y + h + pad_h) - py

        roi = frame[py:py + ph, px:px + pw]
        if CLAHE_ENABLED:
            roi = apply_clahe(roi)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        primary_result = self._run_inference(roi_rgb, timestamp_ms)
        if not primary_result.pose_landmarks:
            return None, None

        landmarks = primary_result.pose_landmarks[0]

        # Flip TTA — only for still-image mode because video mode expects a
        # single input per timestamp and would confuse the internal tracker.
        if FLIP_TTA_ENABLED and self._running_mode_name == "image":
            flipped_roi_rgb = cv2.flip(roi_rgb, 1)
            flipped_result = self._run_inference(flipped_roi_rgb, timestamp_ms)
            if flipped_result.pose_landmarks:
                landmarks = self._merge_flipped(landmarks, flipped_result.pose_landmarks[0])

        roi_h, roi_w = roi_rgb.shape[:2]
        analysis = analyze_ski_pose(
            landmarks,
            roi_w,
            roi_h,
            visibility_threshold=POSE_VISIBILITY_THRESHOLD,
            visibility_threshold_legs=POSE_VISIBILITY_THRESHOLD_LEGS,
            topology=MEDIAPIPE_33,
        )
        if not analysis:
            return None, None

        entry = {
            "landmarks": landmarks,
            "bbox": (px, py, pw, ph),
            "shoulder_center": analysis.get("shoulder_center"),
            "topology": MEDIAPIPE_33,
        }
        return entry, analysis

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    # ------------------------------------------------------------------
    # Internals

    def _run_inference(self, roi_rgb, timestamp_ms):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        with SuppressStderr():
            if self._mp_running_mode == vision.RunningMode.VIDEO:
                return self._landmarker.detect_for_video(mp_image, timestamp_ms)
            return self._landmarker.detect(mp_image)

    def _merge_flipped(self, landmarks_orig, landmarks_flipped):
        """Combine original and flipped passes, keeping the higher-visibility
        landmark per joint after unflipping the second pass."""
        merged = []
        for i in range(len(landmarks_orig)):
            orig = landmarks_orig[i]
            flipped_src = landmarks_flipped[self._flip_swap_table[i]]
            unflipped = _MergedLandmark(
                x=1.0 - flipped_src.x,
                y=flipped_src.y,
                z=getattr(flipped_src, "z", 0.0),
                visibility=flipped_src.visibility,
                presence=getattr(flipped_src, "presence", 0.0),
            )
            if unflipped.visibility > orig.visibility:
                merged.append(unflipped)
            else:
                merged.append(_MergedLandmark(
                    x=orig.x, y=orig.y, z=getattr(orig, "z", 0.0),
                    visibility=orig.visibility,
                    presence=getattr(orig, "presence", 0.0),
                ))
        return merged
