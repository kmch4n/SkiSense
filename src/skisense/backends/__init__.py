"""Pose estimation backend dispatcher.

Exposes ``get_backend(...)`` which selects a concrete ``PoseBackend``:

- ``"sam3d"`` (default): SAM 3D Body, 3D MHR-21, CUDA required.
- ``"yolo11"``: YOLO11-Pose, 2D COCO-17, CPU/MPS/CUDA capable. This is
  the pre-migration engine, kept as a supported fallback.

Resolution order is ``get_backend(backend=...)`` (i.e. the ``run.py``
``--pose-backend`` flag) first, then ``SKISENSE_POSE_BACKEND`` in
``.env``, then the built-in default.
"""
from typing import Optional

from ..config import AVAILABLE_POSE_BACKENDS
from .base import PoseBackend

#: Backend names accepted by ``get_backend`` and ``run.py --pose-backend``.
#: Defined in ``config.py`` so the env-var validation and the CLI choices
#: cannot drift apart.
AVAILABLE_BACKENDS = AVAILABLE_POSE_BACKENDS


def resolve_backend_name(backend: Optional[str] = None) -> str:
    """Normalise and validate a backend name.

    Args:
        backend: Explicit backend name, typically from the CLI. ``None``
            or an empty string falls back to ``SKISENSE_POSE_BACKEND``.

    Returns:
        One of ``AVAILABLE_BACKENDS``.

    Raises:
        ValueError: If ``backend`` is not a known backend name.
    """
    from ..config import POSE_BACKEND

    if backend is None or not backend.strip():
        return POSE_BACKEND

    name = backend.strip().lower()
    if name not in AVAILABLE_BACKENDS:
        raise ValueError(
            f"Unsupported pose backend: {backend!r}. "
            f"Expected one of {', '.join(AVAILABLE_BACKENDS)}."
        )
    return name


def get_backend(
    running_mode: str = "video",
    device=None,
    use_gpu: bool = False,
    device_str: str = "cpu",
    backend: Optional[str] = None,
) -> PoseBackend:
    """Build and return the requested pose backend.

    Both backends are stateless across image and video flows and share the
    same ``(entry, analysis)`` contract plus ``estimate_full_frame`` for
    fast mode, so callers do not need to know which one is active.

    Args:
        running_mode: Accepted for interface compatibility.
        device: PyTorch device for GPU-capable backends.
        use_gpu: True when CUDA or MPS is active.
        device_str: ``"cuda"``, ``"mps"``, or ``"cpu"``. SAM 3D Body
            requires ``"cuda"`` and raises otherwise.
        backend: Explicit backend name overriding ``SKISENSE_POSE_BACKEND``.

    Returns:
        An initialised ``PoseBackend`` subclass instance.
    """
    name = resolve_backend_name(backend)

    if name == "yolo11":
        from .yolo11_backend import Yolo11Backend
        return Yolo11Backend(device=device, use_gpu=use_gpu, device_str=device_str)

    from .sam3d_backend import Sam3dBackend
    return Sam3dBackend(device=device, use_gpu=use_gpu, device_str=device_str)


__all__ = [
    "AVAILABLE_BACKENDS",
    "PoseBackend",
    "get_backend",
    "resolve_backend_name",
]
