"""Compare pose backends on identical skier frames and detection boxes."""

from __future__ import annotations

import json
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import INPUT_DIR, OUTPUT_DIR, SAM3D_HF_REPO, YOLO_POSE_MODEL, YOLO26_POSE_MODEL


def frame_indices(total_frames: int, count: int) -> list[int]:
    """Choose evenly spaced interior frames without duplicates."""
    if total_frames <= 0 or count <= 0:
        return []
    return sorted({min(total_frames - 1, int(total_frames * i / (count + 1)))
                   for i in range(1, count + 1)})


def select_target_bbox(bboxes: list[tuple[int, int, int, int]]):
    """Pick the largest detected person in a sampled frame."""
    return max(bboxes, key=lambda box: box[2] * box[3], default=None)


def serialize_pose(entry: dict[str, Any]) -> dict[str, Any]:
    """Store image-space points and visibility without model-specific objects."""
    x, y, w, h = entry["bbox"]

    def landmark_data(lm):
        valid = np.isfinite(lm.x) and np.isfinite(lm.y)
        return {
            "x": float(x + lm.x * w) if valid else None,
            "y": float(y + lm.y * h) if valid else None,
            "visibility": float(lm.visibility) if np.isfinite(lm.visibility) else 0.0,
        }

    result = {
        "topology": entry["topology"].name,
        "bbox": [int(x), int(y), int(w), int(h)],
        "landmarks": [landmark_data(lm) for lm in entry["landmarks"]],
    }
    joints_3d = entry.get("joints_3d")
    if joints_3d is not None:
        if hasattr(joints_3d, "detach"):
            joints_3d = joints_3d.detach().cpu().numpy()
        values = np.asarray(joints_3d)
        if values.ndim == 2 and values.shape[1] == 3:
            result["joints_3d"] = [
                [float(value) if np.isfinite(value) else None for value in point]
                for point in values
            ]
            result["coordinate_system"] = entry.get(
                "coordinate_system", "mhr_camera_space"
            )
    return result


def draw_pose(frame: np.ndarray, entry: dict[str, Any]) -> np.ndarray:
    """Draw visible joints on an unzoomed copy of the input frame."""
    result = frame.copy()
    topology = entry["topology"]
    x, y, w, h = entry["bbox"]
    landmarks = entry["landmarks"]

    def point(index: int):
        if index >= len(landmarks) or landmarks[index].visibility < 0.3:
            return None
        lm = landmarks[index]
        if not (np.isfinite(lm.x) and np.isfinite(lm.y)):
            return None
        return int(x + lm.x * w), int(y + lm.y * h)

    visible_indices = set()
    for start, end in topology.connections:
        visible_indices.update((start, end))
        first, second = point(start), point(end)
        if first is not None and second is not None:
            cv2.line(result, first, second, (0, 255, 0), 3, cv2.LINE_AA)
    for index in sorted(visible_indices):
        location = point(index)
        if location is not None:
            cv2.circle(result, location, 4, (0, 120, 255), -1, cv2.LINE_AA)
    return result


def _load_samples(input_path: Path, indices: list[int]):
    """Read requested image or video frames before loading inference models."""
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise ValueError(f"Could not read image: {input_path}")
        return [(0, frame)]

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not indices:
        indices = frame_indices(total, 3)
    samples = []
    for index in indices:
        if index < 0 or index >= total:
            cap.release()
            raise ValueError(f"Frame {index} is outside 0..{total - 1}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Could not read frame {index}")
        samples.append((index, frame))
    cap.release()
    return samples


def _make_backend(name: str, device, use_gpu: bool, device_str: str):
    if name == "rtmw3d":
        from .compare_rtmw3d import Rtmw3dComparisonBackend
        return Rtmw3dComparisonBackend(device=device_str)
    from .backends import get_backend
    return get_backend(device=device, use_gpu=use_gpu, device_str=device_str, backend=name)


def _crop_for_comparison(frame: np.ndarray, bbox, size: int = 560) -> np.ndarray:
    """Use the same fixed square around the skier in every model column."""
    x, y, w, h = bbox
    height, width = frame.shape[:2]
    side = min(max(w, h) * 3, max(width, height))
    cx, cy = x + w / 2, y + h / 2
    left = max(0, int(cx - side / 2))
    top = max(0, int(cy - side / 2))
    right = min(width, int(cx + side / 2))
    bottom = min(height, int(cy + side / 2))
    crop = frame[top:bottom, left:right]
    canvas = np.full((size, size, 3), 32, dtype=np.uint8)
    if crop.size:
        scale = min(size / crop.shape[1], size / crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)),
                                    max(1, int(crop.shape[0] * scale))))
        rh, rw = resized.shape[:2]
        canvas[(size - rh) // 2:(size - rh) // 2 + rh,
               (size - rw) // 2:(size - rw) // 2 + rw] = resized
    return canvas


def _contact_sheet(columns: list[tuple[str, np.ndarray]], bbox) -> np.ndarray:
    tiles = []
    for label, image in columns:
        tile = _crop_for_comparison(image, bbox)
        tile = cv2.copyMakeBorder(tile, 46, 0, 0, 0, cv2.BORDER_CONSTANT,
                                  value=(30, 30, 30))
        cv2.putText(tile, label, (14, 31), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(tile)
    return np.hstack(tiles)


def compare(
    input_file: str,
    models: list[str],
    indices: list[int] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Save per-model overlays, side-by-side frames, and machine-readable points."""
    allowed = {"sam3d", "yolo11", "yolo26", "rtmw3d"}
    if not models or len(models) != len(set(models)) or set(models) - allowed:
        raise ValueError(f"Choose unique models from: {', '.join(sorted(allowed))}")
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = Path(INPUT_DIR) / input_path
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    samples = _load_samples(input_path, indices or [])
    if not samples:
        raise ValueError("No frames selected")
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / f"comparison_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    from .main import load_yolo_model, resolve_device, run_yolo_detection
    from .config import DEVICE_PREFERENCE
    device, use_gpu, device_str = resolve_device(DEVICE_PREFERENCE)
    detector = load_yolo_model(device, use_gpu)
    detector_device = 0 if device_str == "cuda" else device_str
    selected = []
    for index, frame in samples:
        boxes = run_yolo_detection(detector, frame, detector_device,
                                   device_str == "cuda")
        bbox = select_target_bbox(boxes)
        if bbox is None:
            print(f"Frame {index}: no person detected; skipping")
            continue
        selected.append((index, frame, bbox))
    del detector
    if not selected:
        raise ValueError("No people detected in the selected frames")

    record: dict[str, Any] = {
        "input": input_path.name,
        "device": device_str,
        "package_versions": {},
        "models": models,
        "model_checkpoints": {
            "sam3d": SAM3D_HF_REPO,
            "yolo11": YOLO_POSE_MODEL,
            "yolo26": YOLO26_POSE_MODEL,
            "rtmw3d": "rtmlib Wholebody3d default checkpoint",
        },
        "selection": "largest YOLOv8x person bbox per frame",
        "interpretation": "2D overlays are comparable visually; 3D coordinates have model-specific frames and must not be compared numerically without alignment.",
        "frames": {},
    }
    for package in ("torch", "ultralytics", "rtmlib", "onnxruntime-gpu"):
        try:
            record["package_versions"][package] = version(package)
        except PackageNotFoundError:
            pass
    rendered = {}
    for index, frame, bbox in selected:
        key = str(index)
        record["frames"][key] = {"target_bbox": list(bbox), "models": {}}
        original = frame.copy()
        x, y, w, h = bbox
        cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 0), 3)
        rendered[key] = [("Original / target", original)]

    import torch
    for name in models:
        print(f"Loading {name}...")
        backend = None
        try:
            backend = _make_backend(name, device, use_gpu, device_str)
            for index, frame, bbox in selected:
                key = str(index)
                start = time.perf_counter()
                try:
                    entry, _ = backend.estimate(frame, bbox)
                    elapsed = round(time.perf_counter() - start, 3)
                    if entry is None:
                        result = {"status": "no_pose", "seconds": elapsed}
                        overlay = frame.copy()
                    else:
                        result = {"status": "ok", "seconds": elapsed,
                                  **serialize_pose(entry)}
                        result["visibility_semantics"] = (
                            "placeholder_one_for_used_joints" if name == "sam3d"
                            else "model_keypoint_score"
                        )
                        overlay = draw_pose(frame, entry)
                    filename = f"frame_{index:06d}_{name}.jpg"
                    cv2.imwrite(str(output_dir / filename), overlay)
                    result["overlay"] = filename
                    record["frames"][key]["models"][name] = result
                    rendered[key].append((f"{name} ({elapsed:.2f}s)", overlay))
                    print(f"  frame {index}: {result['status']} in {elapsed:.2f}s")
                except Exception as exc:
                    record["frames"][key]["models"][name] = {
                        "status": "error", "error": str(exc),
                    }
                    rendered[key].append((f"{name}: error", frame))
                    print(f"  frame {index}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            for index, frame, _ in selected:
                key = str(index)
                record["frames"][key]["models"][name] = {
                    "status": "error", "error": str(exc),
                }
                rendered[key].append((f"{name}: unavailable", frame))
            print(f"{name} unavailable: {type(exc).__name__}: {exc}")
        finally:
            if backend is not None:
                backend.close()
            if device_str == "cuda":
                torch.cuda.empty_cache()

    sheets = []
    for index, _, bbox in selected:
        filename = f"frame_{index:06d}_comparison.jpg"
        cv2.imwrite(str(output_dir / filename), _contact_sheet(rendered[str(index)], bbox))
        record["frames"][str(index)]["comparison"] = filename
        sheets.append(filename)
    (output_dir / "results.json").write_text(
        json.dumps(record, indent=4, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    import html
    items = "\n".join(
        f'<h2>Frame {index}</h2><img src="{html.escape(filename)}" alt="Pose comparison for frame {index}">'
        for (index, _, _), filename in zip(selected, sheets)
    )
    (output_dir / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>SkiSense pose comparison</title>'
        '<style>body{font:16px system-ui;background:#14191f;color:#fff;margin:24px}'
        'img{max-width:100%;height:auto}h2{margin-top:40px}</style>'
        f'<h1>SkiSense pose comparison</h1><p>{html.escape(input_path.name)} · '
        'The same target box is used for model association. Scores are omitted because '
        'joint definitions differ.</p>' + items + '</html>\n', encoding="utf-8"
    )
    return output_dir
