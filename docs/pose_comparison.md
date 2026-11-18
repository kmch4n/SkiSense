# Pose model comparison

`compare.py` runs multiple pose models against the same sampled frames and
uses one skier bounding box to select each model's target person. It saves a side-by-side image for each frame,
individual full-resolution overlays, and image-space keypoints in
`results.json`. The regular pipeline selects `yolo26` by default and accepts
`--pose-backend yolo26`, `yolo11`, or `sam3d`; `compare.py` does not change
that choice. RTMW3D remains comparison-only because its 3D coordinate frame
and score semantics have not been validated for the regular pipeline.

## Setup

Install the base and SAM 3D Body dependencies described in the README.
The optional RTMW3D column additionally requires `rtmlib` and ONNX Runtime:

```bash
pip install --upgrade "ultralytics>=8.4.215"
pip install "rtmlib==0.0.16" "onnxruntime-gpu==1.32.0"
# rtmlib also requires the CPU distribution; install the GPU files last:
pip install --force-reinstall --no-deps onnxruntime-gpu
```

`rtmlib` downloads its detector and RTMW3D weights on first use. The
Ultralytics weights also download on first use. SAM 3D Body needs its gated
Hugging Face weights and CUDA, as in the regular pipeline. For a CPU-only
comparison, omit `sam3d` and use `onnxruntime` instead of `onnxruntime-gpu`.

## Run

Put an image or video in `input/`, then run:

```bash
python compare.py video.mp4 --frames 30,90
python compare.py skier.jpg --models yolo11,yolo26,rtmw3d
```

Available models: `sam3d`, `yolo11`, `yolo26`, and `rtmw3d`. The default
includes all four. Without `--frames`, three evenly spaced video frames are
chosen. Frame numbers are zero-based. The largest YOLOv8x person detection
in each frame is the shared target; inspect the cyan box in the first column
to verify it is the intended skier. The program keeps each model loaded while
processing the selected frames and releases it before loading the next one.

Open `output/comparison_YYYYMMDD_HHMMSS/index.html` or the individual
`frame_*_comparison.jpg` files. `results.json` records the selected bbox,
package versions, per-model status, inference time, projected 2D landmarks,
and available raw 3D joints. The `visibility` field is a model keypoint score
except for SAM 3D Body, which uses a placeholder value of 1.0 for the joints
it exposes; see `visibility_semantics` in the JSON. The first frame's time
can include model initialization, checkpoint downloads, and GPU warmup;
compare later frames for a rough steady-state runtime. The generated output
stays ignored by Git.

If optional RTMW3D dependencies are absent or SAM 3D Body weights are not
accessible, the affected column records `status: error` in `results.json`.
Other selected models continue. Check each column's status before
interpreting the comparison.

## Interpretation

The columns show the same source pixels. YOLO11, YOLO26, and SAM 3D Body
receive the shared target box as a region of interest. RTMW3D runs on the
full frame and matches its detected person to the box by torso position;
it reports `no_pose` if the match is too far away. A model can still fail
to detect or correctly associate the target person. YOLO11 and
YOLO26 use COCO-17 body points; RTMW3D adds projected foot points; SAM 3D
Body provides MHR body and foot points. None of these pretrained models
tracks ski tips, ski tails, or poles. See the
[ski-specific model research](pose_comparison_research_ja.md) for datasets
that annotate those elements.

The comparator deliberately omits SkiSense's score. Different topologies and
2D/3D coordinate systems make a shared score misleading. RTMW3D's raw 3D
coordinates are saved for inspection, but their frame and scale are not
validated for SkiSense joint-angle scoring. A visual match to a few frames
is not an accuracy benchmark; evaluate more turns, views, occlusions, and
skier sizes before choosing a model.
