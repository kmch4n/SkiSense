"""Meta SAM 3D Body backend.

Drives Meta's SAM 3D Body model through the shared ``PoseBackend``
interface. Emits MHR-21 landmarks (body + feet) so ``analyze_ski_pose``
scores knee, hip and ankle joints with true 3D angles.

The Meta repository ships as a regular Python package (``sam_3d_body``)
plus a HuggingFace-hosted checkpoint. ``setup_sam_3d_body`` returns a
``SAM3DBodyEstimator`` whose ``process_one_image`` accepts a full BGR/RGB
frame plus optional pre-computed bboxes. SkiSense relies on YOLOv8x +
Deep SORT for detection/tracking and feeds bboxes here per-frame.
"""
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from .._logging import SuppressStderr

with SuppressStderr():
    import cv2
    import torch

from ..config import (
    POSE_VISIBILITY_THRESHOLD,
    POSE_VISIBILITY_THRESHOLD_LEGS,
    ROI_PADDING_RATIO,
    SAM3D_HF_REPO,
    SAM3D_USE_HAND_REFINE,
)
from ..pose_analyzer import analyze_ski_pose
from ..pose_topology import MHR_BODY
from .base import PoseBackend


class _MhrLandmark:
    """Landmark shim consumed by ``analyze_ski_pose`` and drawing helpers.

    ``x`` and ``y`` are normalised to the padded ROI (0-1), so existing
    drawing code (``landmark.x * bw``) keeps working without changes.
    ``x3d``, ``y3d``, ``z3d`` carry the MHR camera-frame metres, which
    are used for visibility-invariant 3D joint angles.
    """
    __slots__ = (
        "x", "y", "z", "visibility", "presence",
        "x3d", "y3d", "z3d",
    )

    def __init__(
        self,
        x: float,
        y: float,
        x3d: float,
        y3d: float,
        z3d: float,
        visibility: float,
    ):
        self.x = x
        self.y = y
        self.z = z3d  # kept as a depth alias for callers that read ``.z``
        self.visibility = visibility
        self.presence = visibility
        self.x3d = x3d
        self.y3d = y3d
        self.z3d = z3d


def _ensure_external_on_path() -> None:
    """Add the vendored SAM 3D Body checkout to ``sys.path``.

    SkiSense vendors the repository under ``external/sam-3d-body`` so
    Phase 0 only needs a ``git clone``. The package directory imports
    ``notebook.utils`` and ``tools.vis_utils`` as top-level packages,
    matching Meta's documented usage.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    vendor = os.path.join(project_root, "external", "sam-3d-body")
    if vendor not in sys.path and os.path.isdir(vendor):
        sys.path.insert(0, vendor)


class Sam3dBackend(PoseBackend):
    """SAM 3D Body backend emitting MHR-21 landmarks with 3D coords."""

    topology = MHR_BODY
    display_name = "SAM 3D Body (MHR-21)"

    def __init__(self, device=None, use_gpu: bool = False, device_str: str = "cpu"):
        self._device = device
        self._use_gpu = use_gpu
        self._device_str = device_str
        if device_str != "cuda":
            # SAM 3D Body's reference implementation moves batches to CUDA
            # unconditionally; CPU/MPS paths are unsupported in v1.
            raise RuntimeError(
                "SAM 3D Body backend requires CUDA. "
                f"Current device_str={device_str!r}."
            )

        _ensure_external_on_path()
        # Build the estimator directly via the sam_3d_body package, not
        # via notebook.utils.setup_sam_3d_body. The notebook helper
        # imports pyrender's Renderer, which pulls in EGL — unavailable
        # on stock Windows installs. The direct path also lets us skip
        # the optional detector, segmentor and FOV estimator without
        # touching vendored code.
        with SuppressStderr():
            from sam_3d_body import (  # type: ignore
                load_sam_3d_body_hf,
                SAM3DBodyEstimator,
            )

        model, model_cfg = load_sam_3d_body_hf(SAM3D_HF_REPO, device="cuda")
        self._estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self._inference_type = "full" if SAM3D_USE_HAND_REFINE else "body"

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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bboxes_xyxy = np.array([[px, py, px + pw, py + ph]], dtype=np.float32)

        with SuppressStderr():
            outputs = self._estimator.process_one_image(
                frame_rgb,
                bboxes=bboxes_xyxy,
                inference_type=self._inference_type,
            )

        if not outputs:
            return None, None

        landmarks = self._build_landmarks(outputs[0], px, py, pw, ph)
        if landmarks is None:
            return None, None

        analysis = analyze_ski_pose(
            landmarks,
            pw,
            ph,
            visibility_threshold=POSE_VISIBILITY_THRESHOLD,
            visibility_threshold_legs=POSE_VISIBILITY_THRESHOLD_LEGS,
            topology=MHR_BODY,
        )
        if not analysis:
            return None, None

        entry = {
            "landmarks": landmarks,
            "bbox": (px, py, pw, ph),
            "detection_bbox": (x, y, w, h),
            "torso_center": analysis.get("torso_center"),
            "shoulder_center": analysis.get("shoulder_center"),
            "hip_center": analysis.get("hip_center"),
            "topology": MHR_BODY,
            # Carry 3D artifacts forward for callers that want to render
            # the mesh or save MHR parameters in future enhancements.
            "joints_3d": outputs[0].get("pred_keypoints_3d"),
            "mesh_vertices": outputs[0].get("pred_vertices"),
            "mhr_params": {
                "body_pose": outputs[0].get("body_pose_params"),
                "shape": outputs[0].get("shape_params"),
                "scale": outputs[0].get("scale_params"),
            },
            "camera_params": {
                "focal_length": outputs[0].get("focal_length"),
                "translation": outputs[0].get("pred_cam_t"),
            },
        }
        return entry, analysis

    def estimate_full_frame(self, frame) -> List[Tuple[dict, dict]]:
        """Run SAM 3D Body on the full frame using its bundled detector.

        Used by ``--fast`` mode. Each detected person is mapped onto the
        same entry schema as ``estimate`` so existing drawing helpers
        remain reusable.
        """
        frame_h, frame_w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with SuppressStderr():
            outputs = self._estimator.process_one_image(
                frame_rgb,
                inference_type=self._inference_type,
            )

        if not outputs:
            return []

        results: List[Tuple[dict, dict]] = []
        for person in outputs:
            sam3d_bbox = person.get("bbox")
            if sam3d_bbox is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in sam3d_bbox[:4])
            px = max(0, int(x1))
            py = max(0, int(y1))
            pw = min(frame_w, int(x2)) - px
            ph = min(frame_h, int(y2)) - py
            if pw <= 0 or ph <= 0:
                continue

            landmarks = self._build_landmarks(person, px, py, pw, ph)
            if landmarks is None:
                continue

            analysis = analyze_ski_pose(
                landmarks,
                pw,
                ph,
                visibility_threshold=POSE_VISIBILITY_THRESHOLD,
                visibility_threshold_legs=POSE_VISIBILITY_THRESHOLD_LEGS,
                topology=MHR_BODY,
            )
            if not analysis:
                continue

            entry = {
                "landmarks": landmarks,
                "bbox": (px, py, pw, ph),
                "detection_bbox": (px, py, pw, ph),
                "torso_center": analysis.get("torso_center"),
                "shoulder_center": analysis.get("shoulder_center"),
                "hip_center": analysis.get("hip_center"),
                "topology": MHR_BODY,
                "joints_3d": person.get("pred_keypoints_3d"),
                "mesh_vertices": person.get("pred_vertices"),
            }
            results.append((entry, analysis))

        return results

    def close(self) -> None:
        # SAM 3D Body holds Torch modules on CUDA; drop the reference so
        # GC can release VRAM. Subsequent calls will fail (caller's job).
        self._estimator = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internals

    def _build_landmarks(
        self,
        person: dict,
        px: int,
        py: int,
        pw: int,
        ph: int,
    ) -> Optional[List[_MhrLandmark]]:
        """Convert one SAM 3D Body output dict into ``_MhrLandmark`` shims.

        The MHR keypoint arrays carry at least 70 joints; SkiSense uses
        the first ``MHR_BODY.num_landmarks`` entries (body + feet).
        """
        kp2d = person.get("pred_keypoints_2d")
        kp3d = person.get("pred_keypoints_3d")
        if kp2d is None or kp3d is None:
            return None

        # SAM 3D Body does not expose per-joint confidence in the public
        # output, so visibility is taken as 1.0 when the joint index is
        # present. Future work could derive visibility from MHR's
        # per-bone confidence if the model exposes it.
        n_required = MHR_BODY.num_landmarks
        if kp2d.shape[0] < n_required or kp3d.shape[0] < n_required:
            return None

        # Build the set of indices SkiSense actually consumes. Hand-finger
        # joints (everything between 20 and 63 that is not the wrist
        # indices 41/62) are emitted by SAM 3D Body but are not used by
        # SkiSense, and the hand decoder is off; marking them invisible
        # keeps the drawing helpers from sprinkling stray red dots over
        # the skier's hands.
        used_indices = set(MHR_BODY.indices.values())
        for a, b in MHR_BODY.connections:
            used_indices.add(a)
            used_indices.add(b)

        landmarks: List[_MhrLandmark] = []
        for i in range(n_required):
            x_pix, y_pix = float(kp2d[i, 0]), float(kp2d[i, 1])
            x_norm = (x_pix - px) / pw if pw > 0 else 0.0
            y_norm = (y_pix - py) / ph if ph > 0 else 0.0
            x3d, y3d, z3d = (float(v) for v in kp3d[i, :3])
            visibility = 1.0 if i in used_indices else 0.0
            landmarks.append(
                _MhrLandmark(
                    x=x_norm,
                    y=y_norm,
                    x3d=x3d,
                    y3d=y3d,
                    z3d=z3d,
                    visibility=visibility,
                )
            )
        return landmarks


