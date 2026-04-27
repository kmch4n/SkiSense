"""Zoom tracking module for SkiSense.

Handles center tracking and zoom functionality to keep the skier
centered in the output video.
"""
import numpy as np
import cv2
from typing import Optional, Tuple, List


class ZoomTracker:
    """Tracks a target and applies smooth zoom to keep them centered."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        zoom_scale: float = 2.0,
        smoothing: float = 0.15,
        padding: float = 1.5,
        target_area_ratio: float = 0.5,
        max_scale: float = 5.0,
    ):
        """Initialize the ZoomTracker.

        Args:
            frame_width: Width of the input frame
            frame_height: Height of the input frame
            zoom_scale: Fallback zoom magnification
            smoothing: EMA smoothing factor (0.0-1.0, lower = smoother)
            padding: Padding multiplier around bounding box
            target_area_ratio: Target bbox area ratio in the output frame
            max_scale: Maximum dynamic zoom magnification
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zoom_scale = zoom_scale
        self.smoothing = smoothing
        self.padding = padding
        self.target_area_ratio = target_area_ratio
        self.max_scale = max_scale

        # State variables for smoothing
        self.smooth_center_x: Optional[float] = None
        self.smooth_center_y: Optional[float] = None
        self.smooth_scale: Optional[float] = None

        # Track selection state
        self.main_track_id: Optional[int] = None

        # Detection timeout settings
        self.detection_timeout = 30  # Frames (~1 second at 30fps)
        self.frames_since_detection = 0
        self.last_valid_target: Optional[Tuple[float, float, float]] = None  # (cx, cy, scale)

    def select_main_target(
        self,
        tracks: List,
        rects: List[Tuple[int, int, int, int]]
    ) -> Optional[Tuple[int, int, int, int]]:
        """Select the main tracking target from detected tracks.

        Selection priority:
        1. Previously tracked target (if still valid)
        2. Largest bounding box (closest to camera)

        Args:
            tracks: List of Deep SORT Track objects
            rects: List of bounding boxes (x, y, w, h)

        Returns:
            Selected bounding box (x, y, w, h) or None
        """
        confirmed_tracks = [t for t in tracks if t.is_confirmed()]

        if not confirmed_tracks:
            return None

        # Try to keep tracking the same target
        if self.main_track_id is not None:
            for track in confirmed_tracks:
                if track.track_id == self.main_track_id:
                    ltrb = track.to_ltrb()
                    return self._ltrb_to_xywh(ltrb)

        # Select new target: largest bounding box
        best_track = None
        best_area = 0

        for track in confirmed_tracks:
            ltrb = track.to_ltrb()
            area = (ltrb[2] - ltrb[0]) * (ltrb[3] - ltrb[1])
            if area > best_area:
                best_area = area
                best_track = track

        if best_track:
            self.main_track_id = best_track.track_id
            return self._ltrb_to_xywh(best_track.to_ltrb())

        return None

    def _ltrb_to_xywh(self, ltrb: Tuple[float, ...]) -> Tuple[int, int, int, int]:
        """Convert (left, top, right, bottom) to (x, y, w, h)."""
        l, t, r, b = ltrb
        return (int(l), int(t), int(r - l), int(b - t))

    def _scale_for_bbox(self, bbox: Tuple[int, int, int, int]) -> float:
        """Calculate zoom so the target bbox fills the configured area ratio."""
        _x, _y, w, h = bbox
        bbox_area = max(1.0, float(w * h))
        target_area = self.frame_width * self.frame_height * self.target_area_ratio
        target_scale = np.sqrt(target_area / bbox_area)
        return float(np.clip(target_scale, 1.0, self.max_scale))

    def _state_for_detection(
        self,
        bbox: Optional[Tuple[int, int, int, int]],
        target_center: Optional[Tuple[float, float]],
    ) -> Tuple[float, float, float]:
        """Return immediate zoom target state for a detected or missing bbox."""
        if bbox is not None:
            self.frames_since_detection = 0

            x, y, w, h = bbox
            if target_center is not None:
                target_cx, target_cy = target_center
            else:
                target_cx = x + w / 2
                target_cy = y + h / 2
            target_scale = self._scale_for_bbox(bbox)
            self.last_valid_target = (target_cx, target_cy, target_scale)
            return target_cx, target_cy, target_scale

        self.frames_since_detection += 1
        if self.frames_since_detection < self.detection_timeout and self.last_valid_target:
            return self.last_valid_target
        return self.frame_width / 2, self.frame_height / 2, 1.0

    def _set_zoom_state(self, center_x: float, center_y: float, scale: float) -> None:
        """Apply target zoom state without smoothing."""
        self.smooth_center_x = center_x
        self.smooth_center_y = center_y
        self.smooth_scale = scale

    def apply_zoom(
        self,
        frame: np.ndarray,
        center_x: float,
        center_y: float,
        scale: float
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Apply zoom transformation to frame.

        Args:
            frame: Input frame (BGR)
            center_x: Zoom center X coordinate
            center_y: Zoom center Y coordinate
            scale: Zoom scale factor

        Returns:
            Tuple of (zoomed_frame, crop_region)
            crop_region: (x1, y1, x2, y2) in original frame coordinates
        """
        h, w = frame.shape[:2]

        # Calculate crop region
        crop_w = w / scale
        crop_h = h / scale

        # Calculate crop boundaries (centered on target)
        x1 = center_x - crop_w / 2
        y1 = center_y - crop_h / 2
        x2 = x1 + crop_w
        y2 = y1 + crop_h

        # Pad instead of clamping so the target center can stay fixed even near
        # the original frame edges.
        x1_i = int(np.floor(x1))
        y1_i = int(np.floor(y1))
        x2_i = int(np.ceil(x2))
        y2_i = int(np.ceil(y2))

        pad_left = max(0, -x1_i)
        pad_top = max(0, -y1_i)
        pad_right = max(0, x2_i - w)
        pad_bottom = max(0, y2_i - h)

        if pad_left or pad_top or pad_right or pad_bottom:
            source = cv2.copyMakeBorder(
                frame,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_REPLICATE,
            )
        else:
            source = frame

        crop_x1 = x1_i + pad_left
        crop_y1 = y1_i + pad_top
        crop_x2 = x2_i + pad_left
        crop_y2 = y2_i + pad_top
        cropped = source[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped.size == 0:
            return frame.copy(), (0, 0, w, h)

        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed, (x1, y1, x2, y2)

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: List,
        rects: List[Tuple[int, int, int, int]],
        target_center: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """Process a single frame with zoom tracking.

        Args:
            frame: Input frame (without overlays)
            tracks: Deep SORT track objects
            rects: Bounding boxes from YOLO detection
            target_center: Target center in ROI coordinates (optional)

        Returns:
            Tuple of (zoomed_frame, zoom_info)
            zoom_info: dict with 'center', 'scale', 'crop' keys, or None if no zoom
        """
        # Select main target
        bbox = self.select_main_target(tracks, rects)

        if bbox is not None and target_center is not None:
            x, y, _w, _h = bbox
            target_center = (x + target_center[0], y + target_center[1])

        target_cx, target_cy, target_scale = self._state_for_detection(
            bbox,
            target_center,
        )
        self._set_zoom_state(target_cx, target_cy, target_scale)

        # Apply zoom
        zoomed_frame, crop_region = self.apply_zoom(
            frame, self.smooth_center_x, self.smooth_center_y, self.smooth_scale
        )

        zoom_info = {
            'center': (self.smooth_center_x, self.smooth_center_y),
            'scale': self.smooth_scale,
            'crop': crop_region
        }

        return zoomed_frame, zoom_info

    def process_bbox(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        target_center: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """Process a frame using a direct bbox instead of Deep SORT tracks.

        Args:
            frame: Input frame without overlays.
            bbox: Target bbox in full-frame coordinates, or None.
            target_center: Optional target center in full-frame coordinates.

        Returns:
            Tuple of (zoomed_frame, zoom_info).
        """
        target_cx, target_cy, target_scale = self._state_for_detection(
            bbox,
            target_center,
        )
        self._set_zoom_state(target_cx, target_cy, target_scale)

        zoomed_frame, crop_region = self.apply_zoom(
            frame, self.smooth_center_x, self.smooth_center_y, self.smooth_scale
        )

        zoom_info = {
            'center': (self.smooth_center_x, self.smooth_center_y),
            'scale': self.smooth_scale,
            'crop': crop_region
        }

        return zoomed_frame, zoom_info

    def reset(self):
        """Reset tracker state."""
        self.smooth_center_x = None
        self.smooth_center_y = None
        self.smooth_scale = None
        self.main_track_id = None
