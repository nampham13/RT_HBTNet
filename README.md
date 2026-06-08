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
  a key ROI frame. This is a learned latent cue, not an explicit blur-length
  measurement.
- Context Encoder estimates observation quality and branch reliability context.
  It does not estimate speed; it helps fusion decide whether the Temporal
  Texture Branch or Blur Physics Branch should be trusted more.
- Confidence-aware fusion combines the texture and blur predictions using their
  predicted confidences plus the context branch-bias signal.
- EMA or Kalman stabilization smooths the final speed estimate for display.

Input tensors use shape `B,T,C,H,W`.

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

- `best.pt`
- `last.pt`
- `config.yaml`

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

## Benchmark

```bash
python scripts/benchmark.py --config configs/default.yaml
```

The benchmark reports device, input shape, parameter count, average latency,
and FPS.

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

## ROI Config

ROI settings live in `configs/default.yaml`:

```yaml
roi:
  mode: "fixed"
  rois:
    - [100, 100, 400, 160]
  resize_width: 128
  resize_height: 64
```

Each ROI is `[x, y, w, h]` in source-frame pixels. Multiple ROIs are supported
at inference time; RT-HBTNet runs each ROI and uses median speed voting with
average confidence for display.

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
