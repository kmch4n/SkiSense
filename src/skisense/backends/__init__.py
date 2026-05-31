"""Pose estimation backend dispatcher.

Exposes ``get_backend(...)`` which selects a concrete ``PoseBackend``
based on ``SKISENSE_POSE_BACKEND``:

- ``"sam3d"`` (default): SAM 3D Body, 3D MHR-21, CUDA required.
- ``"yolo11"``: YOLO11-Pose, 2D COCO-17, CPU/MPS/CUDA capable.
"""
from .base import PoseBackend


def get_backend(
    running_mode: str = "video",
    device=None,
    use_gpu: bool = False,
    device_str: str = "cpu",
) -> PoseBackend:
    """Build and return the configured pose backend.

    The backend is chosen by ``SKISENSE_POSE_BACKEND``. Both backends are
    stateless across image and video flows and share the same
    ``(entry, analysis)`` contract plus ``estimate_full_frame`` for fast
    mode, so callers do not need to know which one is active.

    Args:
        running_mode: Accepted for interface compatibility.
        device: PyTorch device for GPU-capable backends.
        use_gpu: True when CUDA or MPS is active.
        device_str: ``"cuda"``, ``"mps"``, or ``"cpu"``. SAM 3D Body
            requires ``"cuda"`` and raises otherwise.

    Returns:
        An initialised ``PoseBackend`` subclass instance.
    """
    from ..config import POSE_BACKEND

    if POSE_BACKEND == "yolo11":
        from .yolo11_backend import Yolo11Backend
        return Yolo11Backend(device=device, use_gpu=use_gpu, device_str=device_str)

    from .sam3d_backend import Sam3dBackend
    return Sam3dBackend(device=device, use_gpu=use_gpu, device_str=device_str)


__all__ = ["PoseBackend", "get_backend"]
