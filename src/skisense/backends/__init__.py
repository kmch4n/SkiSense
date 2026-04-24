"""Pose estimation backend dispatcher.

Exposes ``get_backend(...)`` so the pipeline remains ready for future pose
engines while YOLO11-Pose is the sole runtime backend today.
"""
from .base import PoseBackend


def get_backend(
    running_mode: str = "video",
    device=None,
    use_gpu: bool = False,
    device_str: str = "cpu",
) -> PoseBackend:
    """Build and return the configured pose backend.

    Args:
        running_mode: Accepted for interface compatibility; YOLO11-Pose is
            stateless in both video and image flows.
        device: PyTorch device for GPU-capable backends (e.g. YOLO11-Pose).
        use_gpu: True when CUDA or MPS is active.
        device_str: ``"cuda"``, ``"mps"``, or ``"cpu"`` used to select
            backend-specific device flags such as FP16 on CUDA.

    Returns:
        An initialised ``PoseBackend`` subclass instance.
    """
    from .yolo11_backend import Yolo11Backend
    return Yolo11Backend(device=device, use_gpu=use_gpu, device_str=device_str)


__all__ = ["PoseBackend", "get_backend"]
