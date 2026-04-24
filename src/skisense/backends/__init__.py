"""Pose estimation backend dispatcher.

Exposes ``get_backend(name, ...)`` that returns a ``PoseBackend`` instance
configured for the requested engine. Backends are resolved lazily to avoid
importing their heavy dependencies when unused.
"""
from .base import PoseBackend


def get_backend(
    name: str,
    *,
    running_mode: str = "video",
    device=None,
    use_gpu: bool = False,
    device_str: str = "cpu",
) -> PoseBackend:
    """Build and return a pose backend by name.

    Args:
        name: Backend identifier. Currently supports ``"mediapipe"``.
        running_mode: ``"video"`` (stateful, needs monotonic timestamps) or
            ``"image"`` (stateless). Only meaningful for MediaPipe.
        device: PyTorch device for GPU-capable backends (e.g. YOLO11-Pose).
        use_gpu: True when CUDA or MPS is active.
        device_str: ``"cuda"``, ``"mps"``, or ``"cpu"`` used to select
            backend-specific device flags such as FP16 on CUDA.

    Returns:
        An initialised ``PoseBackend`` subclass instance.

    Raises:
        ValueError: Unknown backend name.
    """
    if name == "mediapipe":
        from .mediapipe_backend import MediaPipeBackend
        return MediaPipeBackend(running_mode=running_mode)

    if name == "yolo11":
        from .yolo11_backend import Yolo11Backend
        return Yolo11Backend(device=device, use_gpu=use_gpu, device_str=device_str)

    raise ValueError(f"Unknown pose backend: {name!r}")


__all__ = ["PoseBackend", "get_backend"]
