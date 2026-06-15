# RT-HBTNet: Real-Time Hybrid Blur-Texture Network

RT-HBTNet is a research prototype for conveyor belt speed estimation using
monocular video only. It is designed as a compact experimental baseline for
low-light and motion-blur conditions, not as a production accuracy guarantee.

The system does not use sensors, encoders, lasers, or heavy optical-flow models
as the main method.

## Core Idea

RT-HBTNet combines two lightweight visual branches:

- Temporal Texture Branch learns speed from temporal texture dynamics across a
  sequence of conveyor ROI frames. It uses lightweight Temporal Shift Modules,
  decomposed `(2+1)D` convolutions, and multi-scale ROI pooling so local texture
  motion remains visible across changes in apparent belt scale.
- Blur Physics Branch learns speed from motion-blur-induced visual signatures in
  a key ROI frame. It combines learned CNN features with fixed physics-inspired
  descriptors: Sobel edge attenuation, radial FFT frequency-band statistics, and
  horizontal, vertical, and diagonal directional blur-kernel responses. This is
  still a learned latent cue, not an explicit blur-length measurement.
- Context Encoder estimates observation quality and branch reliability context.
  It does not estimate speed; it helps fusion decide whether the Temporal
  Texture Branch or Blur Physics Branch should be trusted more.
- Cross-attention fusion lets texture features query blur features and blur
  features query texture features. The resulting attention bias is combined with
  branch confidence and context bias before producing the final texture-vs-blur
  fusion weights.
- EMA or Kalman stabilization smooths the final speed estimate for display.

Input tensors use shape `B,T,C,H,W`.

## Repository Layout

```text
configs/      YAML configuration files
datasets/     Synthetic and labeled-video datasets
models/       RT-HBTNet branches, fusion blocks, and model factory
scripts/      Training, inference, ONNX export, benchmark, and demo utilities
tests/        Pytest coverage for ROI, filters, factories, model shapes, and export
utils/        Preprocessing, ROI detection, metrics, calibration, visualization
```

## Installation

```bash
pip install -r requirements.txt
```

For Windows virtual environments:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

Generate a small synthetic conveyor video and matching `labels.csv`:

```bash
python scripts/demo_synthetic_video.py --output data/synthetic_conveyor.mp4 --speed 2.0 --blur --low-light
```

Run a short synthetic training smoke test:

```bash
python scripts/train.py --config configs/default.yaml --synthetic --synthetic-samples 64 --epochs 1 --batch-size 2 --num-workers 0 --save-dir runs/smoke_train
```

Run inference on the generated video without opening a display window:

```bash
python scripts/infer_video.py --config configs/default.yaml --weights runs/smoke_train/best.pt --video data/synthetic_conveyor.mp4 --no-display --save-output runs/smoke_train/demo_output.mp4
```

## Train With Synthetic Data

Synthetic training lets the full pipeline run without real conveyor data:

```bash
python scripts/train.py --config configs/default.yaml --synthetic
```

For a short smoke run on Windows or CPU-only machines:

```bash
python scripts/train.py --config configs/default.yaml --synthetic --synthetic-samples 64 --epochs 1 --batch-size 2 --num-workers 0 --save-dir runs/smoke_train
```

Checkpoints are saved to `runs/train` by default:

- `best.onnx` and `last.onnx` for deployment/inference runtimes that consume ONNX.
- `best.pt` and `last.pt` PyTorch checkpoints for training state, debugging, or
  re-exporting.
- `config.yaml`
- `history.csv` with per-epoch loss and validation metrics.
- `training_curves.png` with loss, MAE, RMSE, and MAPE curves.

Disable ONNX export during training with:

```bash
python scripts/train.py --config configs/default.yaml --synthetic --no-export-onnx
```

Disable training curve plots with:

```bash
python scripts/train.py --config configs/default.yaml --synthetic --no-plots
```

Export an existing PyTorch checkpoint manually with:

```bash
python scripts/export_onnx.py --config configs/default.yaml --weights runs/train/best.pt --output runs/train/best.onnx
```

The ONNX graph keeps batch size dynamic, while sequence length and ROI input
size are fixed from the config used during export. During PyTorch
training/inference the blur physics descriptor uses `torch.fft.rfft2`; during
ONNX export it switches to an equivalent fixed-size DFT path built from
matrix multiplications so export remains compatible with opset 17.

## Run Inference

```bash
python scripts/infer_video.py --config configs/default.yaml --weights runs/train/best.pt --video data/synthetic_conveyor.mp4
```

Optional fixed ROI override:

```bash
python scripts/infer_video.py --config configs/default.yaml --weights runs/train/best.pt --video data/synthetic_conveyor.mp4 --roi "100,100,400,160"
```

Use webcam input:

```bash
python scripts/infer_video.py --config configs/default.yaml --weights runs/train/best.pt --camera 0
```

Run headless and save the annotated dashboard video:

```bash
python scripts/infer_video.py --config configs/default.yaml --weights runs/train/best.pt --video data/synthetic_conveyor.mp4 --no-display --save-output runs/infer/output.mp4
```

Calibrate the model output against a known belt speed. During the calibration
window, raw model predictions are collected and a scale factor is saved to
`calibration.json` by default:

```bash
python scripts/infer_video.py --config configs/default.yaml --weights runs/train/best.pt --video data/synthetic_conveyor.mp4 --known-speed 2.0 --calibrate-seconds 10 --calibration calibration.json
```

Later inference runs load the saved calibration file automatically when
`--known-speed` is omitted.

## Benchmark

```bash
python scripts/benchmark.py --config configs/default.yaml
```

The benchmark reports device, input shape, parameter count, average latency,
and FPS.

Benchmark with a trained checkpoint:

```bash
python scripts/benchmark.py --config configs/default.yaml --weights runs/train/best.pt --batch-size 1 --sequence-length 64
```

## Dataset Format

For real videos, create `labels.csv` with:

```csv
video_path,start_frame,end_frame,speed_mps
videos/belt_001.mp4,0,300,1.25
videos/belt_002.mp4,50,350,2.10
```

Then train with:

```bash
python scripts/train.py --config configs/default.yaml --labels data/labels.csv --video-root data/videos
```

Relative `video_path` values are resolved through `--video-root`.

The labeled-video dataset samples `data.sequence_length` frames evenly from
`start_frame` to `end_frame`. Each sample is preprocessed into ROI tensors with
shape `T,C,H,W`; the DataLoader adds the batch dimension used by the model.

## ROI Config

ROI settings live in `configs/default.yaml`:

```yaml
roi:
  mode: "auto_motion"
  rois: []
  auto_motion:
    warmup_frames: 45
    max_rois: 1
    fallback: "full"
    motion_threshold: 18.0
    score_threshold: 20.0
    min_area_ratio: 0.02
    max_area_ratio: 0.85
  resize_width: 128
  resize_height: 64
```

With `auto_motion`, inference reads a short warm-up clip, detects moving
texture regions with OpenCV frame-difference and Sobel texture energy, then
locks the detected boxes for the rest of the run. If detection fails, it falls
back to the full frame by default. For manual ROIs, set `mode: "fixed"` and use
`rois: [[x, y, w, h]]` in source-frame pixels, or pass `--roi "x,y,w,h"` on the
CLI. Multiple ROIs are supported at inference time; RT-HBTNet runs each ROI and
uses median speed voting with average confidence for display.

## Configuration Notes

- `project.device: "auto"` chooses CUDA when available, otherwise CPU.
- `data.sequence_length` controls the temporal window length.
- `data.grayscale: true` and `model.in_channels: 1` are the default path.
- `training.export_onnx: true` exports `best.onnx` during training and
  `last.onnx` at the end.
- `stabilization.type` supports `ema` and `kalman` through the stabilizer
  factory.

## Testing

Run the test suite with:

```bash
pytest
```

For a narrower smoke check:

```bash
pytest tests/test_model_shapes.py tests/test_factories.py
```

## Limitations

This is a prototype. Do not treat its output as production-grade measurement
without validation against calibrated ground truth.

CV-only monocular video cannot reliably estimate absolute speed if the belt has
no useful visual texture, repeated ambiguous patterns, or no motion-induced
signal. For metric speed, the system needs known reference speed data or
pixel-to-meter calibration. Camera position, lens, zoom, and ROI should remain
fixed after calibration.

Lighting, exposure, rolling shutter, belt material, camera angle, and blur level
can all shift model behavior. Real deployment requires site-specific data,
calibration, and validation.
