"""Ultralytics YOLO11-Pose backend.

Drives Ultralytics' YOLO11-Pose model through the shared ``PoseBackend``
interface. Emits COCO-17 landmarks (no foot keypoints), so
``analyze_ski_pose`` automatically skips ankle-angle scoring when this
backend is selected.
"""
import os
import shutil
from typing import List, Optional, Tuple

from .._logging import SuppressStderr

with SuppressStderr():
    import cv2
    from ultralytics import YOLO

from ..config import (
    CLAHE_ENABLED,
    FLIP_TTA_ENABLED,
    MODEL_DIR,
    POSE_VISIBILITY_THRESHOLD,
    POSE_VISIBILITY_THRESHOLD_LEGS,
    ROI_PADDING_RATIO,
    YOLO_POSE_CONFIDENCE,
    YOLO_POSE_MODEL,
)
from ..pose_analyzer import analyze_ski_pose
from ..pose_topology import COCO_17, build_flip_swap_table
from ..preprocessing import apply_clahe
from .base import PoseBackend


class _YoloLandmark:
    """Minimal landmark shim matching the attribute shape consumed by
    ``analyze_ski_pose`` and the drawing helpers.
    """
    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x: float, y: float, visibility: float):
        self.x = x
        self.y = y
        self.z = 0.0
        self.visibility = visibility
        self.presence = visibility


def _resolve_pose_device(device, use_gpu: bool, device_str: str):
    """Translate the shared device triple into Ultralytics device/half args."""
    if device_str == "cuda":
        return 0, True
    if device_str == "mps":
        return "mps", False
    return "cpu", False


class Yolo11Backend(PoseBackend):
    """YOLO11-Pose backend emitting COCO-17 landmarks."""

    topology = COCO_17

    def __init__(self, device=None, use_gpu: bool = False, device_str: str = "cpu"):
        self._device = device
        self._use_gpu = use_gpu
        self._device_str = device_str
        self._yolo_device, self._yolo_half = _resolve_pose_device(device, use_gpu, device_str)
        self._flip_swap_table = build_flip_swap_table(COCO_17)
        self._model = self._build_model()

    # ------------------------------------------------------------------
    # Model setup

    def _build_model(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        preferred_path = os.path.join(MODEL_DIR, YOLO_POSE_MODEL)

        if not os.path.exists(preferred_path):
            self._ensure_weights_under_models(preferred_path)

        with SuppressStderr():
            model = YOLO(preferred_path)
            if self._use_gpu and self._device is not None:
                model.to(self._device)
        return model

    def _ensure_weights_under_models(self, preferred_path: str) -> None:
        """Place YOLO-Pose weights at ``preferred_path``.

        If a copy already exists in the current working directory from an
        earlier run, move it instead of re-downloading. Otherwise trigger
        Ultralytics' auto-download while ``MODEL_DIR`` is the working
        directory so the file lands in the right place from the start.
        """
        cwd_path = os.path.abspath(YOLO_POSE_MODEL)
        if (os.path.exists(cwd_path)
                and os.path.dirname(cwd_path) != os.path.abspath(MODEL_DIR)):
            shutil.move(cwd_path, preferred_path)
            return

        original_cwd = os.getcwd()
        try:
            os.chdir(MODEL_DIR)
            with SuppressStderr():
                # Instantiating YOLO with a bare filename triggers an
                # auto-download into the current working directory.
                _ = YOLO(YOLO_POSE_MODEL)
        finally:
            os.chdir(original_cwd)

    # ------------------------------------------------------------------
    # Public API

    def estimate(
        self,
        frame,
        bbox,
        timestamp_ms: Optional[int] = None,  # unused; backend is stateless
    ) -> Tuple[Optional[dict], Optional[dict]]:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None, None

        pad_w = int(w * ROI_PADDING_RATIO)
        pad_h = int(h * ROI_PADDING_RATIO)
        frame_h, frame_w = frame.shape[:2]
        px = max(0, x - pad_w)
        py = max(0, y - pad_h)
        pw = min(frame_w, x + w + pad_w) - px
        ph = min(frame_h, y + h + pad_h) - py

        if pw <= 0 or ph <= 0:
            return None, None

        roi = frame[py:py + ph, px:px + pw]
        if roi.size == 0:
            return None, None

        if CLAHE_ENABLED:
            roi = apply_clahe(roi)

        landmarks = self._infer_landmarks(roi)
        if landmarks is None:
            return None, None

        # YOLO is stateless — Flip TTA can be enabled in both image and
        # video mode without confusing any internal tracker state.
        if FLIP_TTA_ENABLED:
            flipped_roi = cv2.flip(roi, 1)
            flipped_landmarks = self._infer_landmarks(flipped_roi)
            if flipped_landmarks is not None:
                landmarks = self._merge_flipped(landmarks, flipped_landmarks)

        analysis = analyze_ski_pose(
            landmarks,
            pw,
            ph,
            visibility_threshold=POSE_VISIBILITY_THRESHOLD,
            visibility_threshold_legs=POSE_VISIBILITY_THRESHOLD_LEGS,
            topology=COCO_17,
        )
        if not analysis:
            return None, None

        entry = {
            "landmarks": landmarks,
            "bbox": (px, py, pw, ph),
            "shoulder_center": analysis.get("shoulder_center"),
            "topology": COCO_17,
        }
        return entry, analysis

    def estimate_full_frame(self, frame) -> List[Tuple[dict, dict]]:
        """Run one full-frame YOLO11-Pose pass and return all valid people.

        The returned landmark entries use the full frame as their coordinate
        base so drawing can reuse the existing ROI-aware helpers.
        """
        frame_h, frame_w = frame.shape[:2]
        with SuppressStderr():
            results = self._model(
                frame,
                conf=YOLO_POSE_CONFIDENCE,
                verbose=False,
                device=self._yolo_device,
                half=self._yolo_half,
            )

        if not results:
            return []

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xyn is None or len(keypoints.xyn) == 0:
            return []

        boxes = getattr(result, "boxes", None)
        boxes_xyxy = None
        boxes_conf = None
        if boxes is not None and boxes.xyxy is not None:
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            if boxes.conf is not None:
                boxes_conf = boxes.conf.cpu().numpy()

        xyn_all = keypoints.xyn.cpu().numpy()
        conf_all = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
        entries = []

        for person_idx, xyn in enumerate(xyn_all):
            conf = conf_all[person_idx] if conf_all is not None else [1.0] * len(xyn)
            landmarks = self._build_landmarks(xyn, conf)
            analysis = analyze_ski_pose(
                landmarks,
                frame_w,
                frame_h,
                visibility_threshold=POSE_VISIBILITY_THRESHOLD,
                visibility_threshold_legs=POSE_VISIBILITY_THRESHOLD_LEGS,
                topology=COCO_17,
            )
            if not analysis:
                continue

            detection_bbox = self._bbox_for_person(
                boxes_xyxy,
                person_idx,
                frame_w,
                frame_h,
            )
            entry = {
                "landmarks": landmarks,
                "bbox": (0, 0, frame_w, frame_h),
                "detection_bbox": detection_bbox,
                "shoulder_center": analysis.get("shoulder_center"),
                "topology": COCO_17,
            }
            if boxes_conf is not None:
                entry["confidence"] = float(boxes_conf[person_idx])
            entries.append((entry, analysis))

        return entries

    def close(self) -> None:
        # Ultralytics YOLO does not expose an explicit close; drop the
        # reference so CUDA memory can be reclaimed by GC.
        self._model = None

    # ------------------------------------------------------------------
    # Internals

    def _infer_landmarks(self, roi):
        """Run YOLO-Pose on a ROI and return a COCO-17 landmark list, or None.

        When multiple persons are detected inside the (padded) ROI, we pick
        the one with the highest bbox confidence.
        """
        with SuppressStderr():
            results = self._model(
                roi,
                conf=YOLO_POSE_CONFIDENCE,
                verbose=False,
                device=self._yolo_device,
                half=self._yolo_half,
            )

        if not results:
            return None
        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xyn is None or len(keypoints.xyn) == 0:
            return None

        boxes = getattr(result, "boxes", None)
        person_idx = 0
        if boxes is not None and boxes.conf is not None and len(boxes.conf) > 1:
            conf_tensor = boxes.conf.cpu().numpy()
            person_idx = int(conf_tensor.argmax())

        xyn = keypoints.xyn[person_idx].cpu().numpy()  # (17, 2)
        # Per-keypoint confidence is the most reliable proxy for visibility.
        if keypoints.conf is not None:
            conf = keypoints.conf[person_idx].cpu().numpy()  # (17,)
        else:
            conf = [1.0] * len(xyn)

        return self._build_landmarks(xyn, conf)

    def _build_landmarks(self, xyn, conf) -> List[_YoloLandmark]:
        """Convert Ultralytics normalized keypoints into landmark shims."""
        return [
            _YoloLandmark(
                x=float(xyn[i, 0]),
                y=float(xyn[i, 1]),
                visibility=float(conf[i]),
            )
            for i in range(len(xyn))
        ]

    def _bbox_for_person(
        self,
        boxes_xyxy,
        person_idx: int,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[int, int, int, int]:
        """Return a clamped bbox for a full-frame YOLO11-Pose detection."""
        if boxes_xyxy is None or person_idx >= len(boxes_xyxy):
            return (0, 0, frame_w, frame_h)

        x1, y1, x2, y2 = boxes_xyxy[person_idx]
        x1 = int(max(0, min(frame_w, x1)))
        y1 = int(max(0, min(frame_h, y1)))
        x2 = int(max(0, min(frame_w, x2)))
        y2 = int(max(0, min(frame_h, y2)))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        if w <= 0 or h <= 0:
            return (0, 0, frame_w, frame_h)
        return (x1, y1, w, h)

    def _merge_flipped(self, landmarks_orig, landmarks_flipped):
        """Pick the higher-visibility landmark per joint between the original
        and (unflipped) flipped passes.
        """
        merged = []
        for i in range(len(landmarks_orig)):
            orig = landmarks_orig[i]
            flipped_src = landmarks_flipped[self._flip_swap_table[i]]
            unflipped = _YoloLandmark(
                x=1.0 - flipped_src.x,
                y=flipped_src.y,
                visibility=flipped_src.visibility,
            )
            if unflipped.visibility > orig.visibility:
                merged.append(unflipped)
            else:
                merged.append(orig)
        return merged
