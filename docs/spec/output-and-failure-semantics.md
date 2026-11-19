# Input/output and failure semantics

**Status:** Observed as-is; several error-handling defects are tracked separately. **Code:** `main.py`, `image_processor.py`, `pose_comparison.py`, backend constructors and `config.py`.

## 1. File-system contract

`PROJECT_ROOT` is derived from the `config.py` source location, independent of the current shell working directory. Standard directories are `models/`, `input/`, and `output/`; empty tracked `.gitkeep` files ensure the folders exist in a clean checkout. Generated model files, input data and output media are git-ignored.

| Mode | Output directory | Files written or copied |
| --- | --- | --- |
| Normal video | `output/YYYYMMDD_HHMMSS/` | `video_pose.mp4`, source copied as `video.mp4`, conditional `best_shot.jpg` |
| Fast video | Same pattern | `video_pose_fast.mp4`, source copied as `video.mp4`, conditional `best_shot.jpg` |
| Still image | Same pattern | `image_pose.jpg`, source copied as `image.jpg` |
| Model comparison | `output/comparison_YYYYMMDD_HHMMSS/` | `index.html`, `results.json`, `frame_NNNNNN_comparison.jpg`, `frame_NNNNNN_<model>.jpg` for completed model calls |

The timestamp is wall-clock time at **one-second resolution** and does not guarantee uniqueness; repeated or concurrent executions during the same second can reuse the same directory and overwrite files. Inputs can be relative to `input/` or absolute paths; a non-MP4 source video is still copied with the name `video.mp4`, and a non-JPEG source image is still copied as `image.jpg` **without transcoding**. These names can misrepresent actual original file formats.

Video output uses OpenCV `mp4v` with source FPS and dimensions. Best-shot JPEGs and annotated still-image JPEGs request quality 95. No audio stream is copied into the generated MP4 by the frame-based writer. The source-video copy is independent of annotated output.

## 2. Normal successful output behavior

Video captures frames until EOF or optional GUI interruption (`q`); it writes each processed frame to the writer. The best shot retains the **first** frame with a strictly maximal score observed so far, and is exported only when the final best score is **positive**. If no positive score occurs, the source copy and MP4 may still exist but no `best_shot.jpg` is produced. There is no per-frame score JSON, CSV, sidecar metadata or score-version identifier in the normal run.

Still-image mode writes an annotation image even when no person was detected, in which case it is a copy of the decoded frame. When a person is found but pose inference fails, the annotation contains its detector bbox only. Comparison output is described in [model-comparison.md](model-comparison.md); unlike regular runs, it records per-model errors as JSON status entries.

## 3. Failure-handling matrix

| Failure or condition | Current behavior | Contract risk |
| --- | --- | --- |
| Unsupported `run.py` flag combination | `argparse.parser.error` | Explicit CLI error, exit 2 |
| Invalid programmatic target mode | Raises `ValueError` | No structured error/result type |
| Unavailable SAM3D CUDA, weights or source | Raises during backend construction/import | No fallback to YOLO backend |
| Missing/unopenable video | Timestamped error and early return, backend closed | Program can exit successfully at process level |
| FPS <= 0, width <= 0, height <= 0 | Logs invalid metadata and returns, closes capture/backend | Does not verify FPS is finite; no nonzero exit |
| Nonpositive `CAP_PROP_FRAME_COUNT` | Not rejected solely for frame count | Progress/selection behavior may be incomplete |
| Failed `cv2.VideoWriter` initialization | **No `isOpened()` check**; continues processing ([#17](https://github.com/kmch4n/SkiSense/issues/17)) | A missing/invalid MP4 can be reported as complete |
| Failed `VideoWriter.write` or truncated output | No explicit success validation | Output completeness not guaranteed |
| Invalid image file / decode failure | Logs then returns after backend initialization | Backend is **not closed** on this early path |
| Failed `cv2.imwrite` (still or best shot) | Return boolean not checked | Silent missing JPEG possible |
| `shutil.copy2` fails | Exception can propagate | Partial output, cleanup not guaranteed |
| No detected person in image | Writes unannotated image | Expected fallback, no score |
| Pose estimate unavailable in video | No current info panel for that frame; target bbox may still zoom | Best-shot uses only successful analyses |
| Target lost in zoom | Last valid zoom held for 1–29 missing frames; reset at 30 | No explicit lost-target warning |
| Comparison model unavailable | Per-model `error` entries, others continue | Partial result is expected |
| Unexpected inference/rendering exception | Usually propagates; no general video/image `try/finally` around frame loop | Possible unreleased handles and partial files |

An early return inside `process_video` or `process_image` does **not** use `sys.exit(nonzero)`. Do not rely on shell exit status to confirm output success. The video pipeline does not perform atomic output moves, integrity validation, retry, checksum, or rollback. It may log completion even when an encoder cannot write.

## 4. Resource and warning behavior

`_logging.SuppressStderr` hides Python and OS-level stderr on selected heavy calls when `SKISENSE_DEBUG=false`, including native library output; warnings are globally filtered and Ultralytics logging lowered. This is a process-wide file-descriptor operation and not a thread-isolated log boundary. Debug mode disables this suppression. Errors are not converted to a common exception taxonomy.

The standard successful video path releases `cv2.VideoCapture`, `cv2.VideoWriter` and the pose backend; image closes the backend on its successful final path. YOLO backend drops its model reference; SAM3D clears CUDA cache on close. Comparison explicitly closes each backend in a `finally`. There is no comprehensive `try/finally` ensuring video and image cleanup for unexpected exceptions.

## 5. Required future contracts

Before implementing robust I/O, specify: validation before costly inference, explicit nonzero exit/raised exceptions, writer-open and post-write checks, atomic/unique output directories, canonical source file naming, source metadata preservation, partial-output cleanup and deterministic no-person behavior. Add mock tests for unsupported codecs, unwritable output, invalid FPS, corrupt images, copy failures and interrupt handling. [Issue #17](https://github.com/kmch4n/SkiSense/issues/17) explicitly covers MP4 writer validation.
