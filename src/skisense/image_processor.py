"""SkiSense - Single-image pose analysis flow.

Analyzes a single ski image (still frame) and saves the annotated result.
Shares YOLO detection and drawing helpers with ``main.py`` and delegates
pose estimation to the configured pose backend.
"""
import os
import shutil
import time
from datetime import datetime

import cv2

from .backends import get_backend
from .config import DEVICE_PREFERENCE, INPUT_DIR, OUTPUT_DIR
from .main import (
    draw_bbox_on_zoomed_frame,
    draw_info_panel,
    draw_landmarks_on_zoomed_frame,
    load_yolo_model,
    log_message,
    resolve_device,
    run_yolo_detection,
)
from .pose_topology import MEDIAPIPE_33


def _pick_largest_bbox(rects):
    """Return the bbox with the largest area, or None if empty."""
    if not rects:
        return None
    return max(rects, key=lambda r: r[2] * r[3])


def process_image(image_file: str = None):
    """Analyze a single ski image and save the annotated output.

    Args:
        image_file: Image filename in input/ directory. Defaults to "image.jpg".
    """
    if image_file is None:
        image_file = "image.jpg"

    device, use_gpu, device_str = resolve_device(DEVICE_PREFERENCE)

    if device_str == "cuda":
        import torch
        torch.backends.cudnn.benchmark = True

    log_message(f"Using device: {device_str.upper()}")
    if use_gpu:
        log_message(f"GPU acceleration: enabled ({device})")

    log_message("=" * 40)
    log_message("Component configuration:")
    if device_str == "cuda":
        log_message("  - YOLO: CUDA GPU (half=True)")
    elif device_str == "mps":
        log_message("  - YOLO: MPS GPU (half=False)")
    else:
        log_message("  - YOLO: CPU")
    log_message("  - Pose backend: MediaPipe (IMAGE mode)")
    log_message("=" * 40)

    image_path = os.path.join(INPUT_DIR, image_file)
    if not os.path.exists(image_path):
        log_message(f"Error: Input image not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        log_message(f"Error: Could not decode image: {image_path}")
        return

    height, width = frame.shape[:2]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    output_image_path = os.path.join(output_dir, "image_pose.jpg")
    input_copy_path = os.path.join(output_dir, "image.jpg")
    shutil.copy2(image_path, input_copy_path)

    pose_backend = get_backend("mediapipe", running_mode="image")
    yolo_model = load_yolo_model(device, use_gpu)

    if device_str == "cuda":
        yolo_device, yolo_half = 0, True
    elif device_str == "mps":
        yolo_device, yolo_half = "mps", False
    else:
        yolo_device, yolo_half = "cpu", False

    start_time = time.time()
    log_message("処理を開始します")

    rects = run_yolo_detection(yolo_model, frame, yolo_device, yolo_half)

    output_frame = frame.copy()
    # Identity zoom info so drawing helpers emit absolute coordinates.
    zoom_info = {
        "center": (width / 2, height / 2),
        "scale": 1.0,
        "crop": (0, 0, width, height),
    }

    analysis = None
    landmarks_entry = None

    target_bbox = _pick_largest_bbox(rects)
    if target_bbox is None:
        log_message("人物が検出されませんでした。元画像のみ保存します。")
    else:
        draw_bbox_on_zoomed_frame(output_frame, target_bbox, zoom_info)
        landmarks_entry, analysis = pose_backend.estimate(frame, target_bbox)
        if landmarks_entry is not None:
            draw_landmarks_on_zoomed_frame(
                output_frame,
                landmarks_entry["landmarks"],
                landmarks_entry["bbox"],
                zoom_info,
                topology=landmarks_entry.get("topology", MEDIAPIPE_33),
            )
            draw_info_panel(output_frame, analysis)
            log_message(f"姿勢スコア: {analysis['score']}/100")
        else:
            log_message("骨格推定に失敗しました。bbox のみ描画します。")

    cv2.imwrite(output_image_path, output_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

    elapsed_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_seconds, 60)
    elapsed_str = f"{minutes}分{seconds}秒" if minutes > 0 else f"{seconds}秒"

    log_message(f"出力先: {output_dir}")
    log_message(f"処理が完了しました ({elapsed_str})")

    pose_backend.close()


if __name__ == "__main__":
    process_image()
