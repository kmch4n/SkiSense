"""Stderr and warning suppression for noisy native libraries.

Importing this module applies environment variables and Python-side logging
overrides before any C++-backed library (MediaPipe, TensorFlow, absl, etc.)
is imported in the process. The ``SuppressStderr`` context manager can
additionally redirect OS-level stderr during specific heavy calls.

Both ``main.py`` and the ``backends`` package import this module so they
share a single definition.
"""
import io
import os
import sys

from .config import DEBUG

__all__ = ["DEBUG", "SuppressStderr"]


if not DEBUG:
    # Env vars must be set before TensorFlow/MediaPipe import.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    os.environ.setdefault("TF_CPP_VMODULE", "tflite=0")

    import warnings
    warnings.filterwarnings("ignore")

    import absl.logging  # noqa: E402
    absl.logging.set_verbosity(absl.logging.ERROR)
    absl.logging.set_stderrthreshold(absl.logging.ERROR)

    import logging as _stdlogging  # noqa: E402
    _stdlogging.getLogger("mediapipe").setLevel(_stdlogging.ERROR)
    _stdlogging.getLogger("ultralytics").setLevel(_stdlogging.ERROR)
    _stdlogging.getLogger("tensorflow").setLevel(_stdlogging.ERROR)


class SuppressStderr:
    """OS-level stderr suppression context manager.

    Becomes a no-op when ``DEBUG`` is True so callers can wrap heavy ops
    unconditionally without branching in the call site.
    """

    def __enter__(self):
        if DEBUG:
            return self
        self._original_stderr = sys.stderr
        sys.stderr = io.StringIO()
        self._original_stderr_fd = os.dup(2)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DEBUG:
            return
        os.dup2(self._original_stderr_fd, 2)
        os.close(self._original_stderr_fd)
        os.close(self._devnull)
        sys.stderr = self._original_stderr
