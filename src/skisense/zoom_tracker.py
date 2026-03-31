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
        padding: float = 1.5
    ):
        """Initialize the ZoomTracker.

        Args:
            frame_width: Width of the input frame
            frame_height: Height of the input frame
            zoom_scale: Fixed zoom magnification
            smoothing: EMA smoothing factor (0.0-1.0, lower = smoother)
            padding: Padding multiplier around bounding box
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.zoom_scale = zoom_scale
        self.smoothing = smoothing
        self.padding = padding

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

    def calculate_zoom_params(
        self,
        bbox: Optional[Tuple[int, int, int, int]]
    ) -> Tuple[float, float, float]:
        """Calculate smooth zoom parameters.

        Args:
            bbox: Target bounding box (x, y, w, h) or None

        Returns:
            Tuple of (center_x, center_y, scale)
        """
        if bbox is None:
            # No detection: gradually return to center with no zoom
            target_cx = self.frame_width / 2
            target_cy = self.frame_height / 2
            target_scale = 1.0
        else:
            x, y, w, h = bbox
            target_cx = x + w / 2
            target_cy = y + h / 2

            if self.auto_scale:
                # Calculate scale based on bounding box size
                # Larger bbox = lower zoom (person is closer)
                bbox_ratio = max(w / self.frame_width, h / self.frame_height)
                # Target: bbox should occupy ~40% of zoomed frame
                target_ratio = 0.4
                target_scale = target_ratio / (bbox_ratio * self.padding)
                target_scale = np.clip(target_scale, self.min_scale, self.max_scale)
            else:
                target_scale = self.zoom_scale

        # Apply EMA smoothing
        if self.smooth_center_x is None:
            self.smooth_center_x = target_cx
            self.smooth_center_y = target_cy
            self.smooth_scale = target_scale
        else:
            self.smooth_center_x += self.smoothing * (target_cx - self.smooth_center_x)
            self.smooth_center_y += self.smoothing * (target_cy - self.smooth_center_y)
            self.smooth_scale += self.smoothing * (target_scale - self.smooth_scale)

        return self.smooth_center_x, self.smooth_center_y, self.smooth_scale

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

        # Clamp to frame boundaries
        if x1 < 0:
            x2 -= x1
            x1 = 0
        if y1 < 0:
            y2 -= y1
            y1 = 0
        if x2 > w:
            x1 -= (x2 - w)
            x2 = w
        if y2 > h:
            y1 -= (y2 - h)
            y2 = h

        # Ensure within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Crop and resize
        cropped = frame[int(y1):int(y2), int(x1):int(x2)]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return zoomed, (x1, y1, x2, y2)

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: List,
        rects: List[Tuple[int, int, int, int]],
        shoulder_center: Optional[Tuple[float, float]] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """Process a single frame with zoom tracking.

        Args:
            frame: Input frame (without overlays)
            tracks: Deep SORT track objects
            rects: Bounding boxes from YOLO detection
            shoulder_center: Shoulder center in ROI coordinates (optional)

        Returns:
            Tuple of (zoomed_frame, zoom_info)
            zoom_info: dict with 'center', 'scale', 'crop' keys, or None if no zoom
        """
        # Select main target
        bbox = self.select_main_target(tracks, rects)

        # Detection success
        if bbox:
            self.frames_since_detection = 0

            # Determine zoom center
            if shoulder_center:
                # Use shoulder center (convert ROI to frame coordinates)
                x, y, w, h = bbox
                target_cx = x + shoulder_center[0]
                target_cy = y + shoulder_center[1]
            else:
                # Use bounding box center as fallback
                x, y, w, h = bbox
                target_cx = x + w / 2
                target_cy = y + h / 2

            # Use fixed zoom scale
            target_scale = self.zoom_scale

            # Save last valid target
            self.last_valid_target = (target_cx, target_cy, target_scale)

        # Detection failure
        else:
            self.frames_since_detection += 1

            # Before timeout: maintain last valid position
            if self.frames_since_detection < self.detection_timeout and self.last_valid_target:
                target_cx, target_cy, target_scale = self.last_valid_target
            # After timeout: return to center
            else:
                target_cx = self.frame_width / 2
                target_cy = self.frame_height / 2
                target_scale = 1.0

        # Apply EMA smoothing
        if self.smooth_center_x is None:
            self.smooth_center_x = target_cx
            self.smooth_center_y = target_cy
            self.smooth_scale = target_scale
        else:
            self.smooth_center_x += self.smoothing * (target_cx - self.smooth_center_x)
            self.smooth_center_y += self.smoothing * (target_cy - self.smooth_center_y)
            self.smooth_scale += self.smoothing * (target_scale - self.smooth_scale)

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

    def reset(self):
        """Reset tracker state."""
        self.smooth_center_x = None
        self.smooth_center_y = None
        self.smooth_scale = None
        self.main_track_id = None
