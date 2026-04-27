"""Abstract base class for pose estimation backends."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from ..pose_topology import PoseTopology


class PoseBackend(ABC):
    """Uniform interface for pose estimation backends.

    Each concrete backend owns its underlying model and exposes a single
    ``estimate`` method that takes a frame + bounding box and returns
    ``(landmark_entry, analysis)``. The landmark entry shape is:

    ```
    {
        "landmarks":       list,               # objects with .x, .y, .visibility
        "bbox":            (px, py, pw, ph),   # padded ROI in frame coordinates
        "torso_center":    (x, y),             # preferred zoom target
        "shoulder_center": (x, y),             # fallback zoom target
        "topology":        PoseTopology,       # describes landmark indices
    }
    ```
    """

    topology: PoseTopology

    @abstractmethod
    def estimate(
        self,
        frame,
        bbox,
        timestamp_ms: Optional[int] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Run pose estimation on a single person's ROI.

        Args:
            frame: Full BGR frame (ndarray).
            bbox: ``(x, y, w, h)`` in frame coordinates.
            timestamp_ms: Monotonic frame timestamp in milliseconds. Only
                consumed by stateful backends running in video mode.

        Returns:
            ``(landmark_entry, analysis)``. Either element may be ``None``
            when estimation fails; callers should check both.
        """

    def close(self) -> None:
        """Release any resources held by the backend.

        Default is a no-op; subclasses override when they own model handles.
        """
        return None
