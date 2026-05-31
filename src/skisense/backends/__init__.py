"""Pose estimation backend dispatcher.

Exposes ``get_backend(...)`` so future engines can be swapped in via the
``PoseBackend`` ABC while SAM 3D Body is the sole runtime backend today.
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
        running_mode: Accepted for interface compatibility; SAM 3D Body
            is stateless in both video and image flows.
        device: PyTorch device for GPU-capable backends.
        use_gpu: True when CUDA is active. MPS/CPU are not supported by
            SAM 3D Body's reference implementation.
        device_str: ``"cuda"``, ``"mps"``, or ``"cpu"``. SAM 3D Body
            requires ``"cuda"``.

    Returns:
        An initialised ``PoseBackend`` subclass instance.
    """
    from .sam3d_backend import Sam3dBackend
    return Sam3dBackend(device=device, use_gpu=use_gpu, device_str=device_str)


__all__ = ["PoseBackend", "get_backend"]
