# BT-ShutterNet

Physics-guided exposure-fraction estimation from ordinary blurred video.

The project no longer predicts metric speed. Its target is the exposure
fraction (shutter angle)

\[
\alpha=\frac{t_{\mathrm{exposure}}}{t_{\mathrm{frame}}}\in[0,1],
\qquad \theta=360^\circ\alpha.
\]

Under locally constant motion, inter-frame displacement \(d\) and the
within-exposure blur trajectory \(b\) satisfy

\[
b \simeq \alpha d.
\]

BT-ShutterNet preserves separate temporal-texture and blur-physics branches,
but does not fuse two unconstrained scalar regressions. The branches predict
dense vector fields and uncertainty maps. A non-learned weighted
least-squares layer then estimates \(\alpha\).

## Why both cues are needed

- A blurred frame alone cannot distinguish fast motion from long exposure.
- Temporal displacement alone does not reveal how much of the frame interval
  the sensor integrated light.
- Their ratio is physically meaningful and does not require metric scene
  scale, camera calibration, or speed labels.

## Architecture

```text
blurred clip B,T,C,H,W
          |
  shared MobileNetV3 encoder
          |
   +------+------+
   |             |
temporal       key-frame blur
(2+1)D/TSM     directional/Sobel cues
   |             |
motion field d  blur field b
uncertainty     uncertainty
   +------ context quality ------+
                    |
       alpha = weighted |b.d| / ||d||^2
```

The blur direction is ambiguous modulo sign, so training and inference use
sign-invariant vector consistency.

## Data

The implemented training adapter uses frame sequences with Middlebury
`.flo` ground truth, such as MPI Sintel:

```text
data/sintel/
  training/
    final/<scene>/frame_XXXX.png
    flow/<scene>/frame_XXXX.flo
```

The renderer creates physically controlled blur by integrating flow-warped
sharp frames in linear-light space. Every sample provides:

- blurred clip;
- exposure fraction `alpha`;
- center-frame inter-frame motion flow;
- blur flow `alpha * motion_flow`;
- valid-pixel mask.

The train/validation split is scene-disjoint to prevent neighboring frames
from leaking across partitions.

See [data/README.md](data/README.md) for setup details.

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Smoke data

Create a tiny deterministic dataset for checking the full pipeline:

```powershell
python scripts/make_toy_exposure_data.py
python scripts/train.py --config configs/smoke.yaml --epochs 1 --num-workers 0
```

Toy data is only for software verification and must never be reported as a
research result.

## Training

Before training, inspect a few generated samples:

```powershell
python scripts/visualize_data.py ^
  --config configs/default.yaml ^
  --dataset exposure_flow ^
  --count 8 ^
  --output-dir runs/visualize_data/sintel
```

If training feels slow, profile the input pipeline before changing the model:

```powershell
python scripts/profile_input_pipeline.py ^
  --config configs/default.yaml ^
  --data-root data/sintel ^
  --samples 64 ^
  --workers 0,2,4
```

```powershell
python scripts/train.py ^
  --config configs/default.yaml ^
  --dataset exposure_flow ^
  --data-root data/sintel ^
  --save-dir runs/exposure
```

Artifacts:

- `best.pt` and `last.pt`;
- resolved `config.yaml`;
- `history.csv` containing alpha MAE/RMSE, motion EPE and blur EPE.

Evaluate synthetic or manifest-based real video:

```powershell
python scripts/evaluate.py ^
  --config configs/default.yaml ^
  --weights runs/exposure/best.pt ^
  --dataset exposure_video ^
  --data-root data/bsd ^
  --manifest data/bsd/manifest.csv ^
  --output runs/exposure/bsd_report.json
```

## Inference

```powershell
python scripts/infer_video.py ^
  --config configs/default.yaml ^
  --weights runs/exposure/best.pt ^
  --video path/to/video.mp4
```

The output reports exposure fraction, shutter angle, inferred exposure time
from video FPS, and model confidence.

## Benchmark

```powershell
python scripts/benchmark.py --config configs/default.yaml --sequence-length 5
```

## Evaluation protocol

The primary real benchmark is the Beam-Splitter Dataset. The direct baseline
is Korčák and Matas, *Video Shutter Angle Estimation using Optical Flow and
Linear Blur*, which reports exposure-fraction MAE 0.039 over 600 clips.

Required comparisons and ablations are specified in
[docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md).
Measured implementation diagnostics are tracked separately in
[docs/DEVELOPMENT_STATUS.md](docs/DEVELOPMENT_STATUS.md).

## Current status

- Dataset renderer, model, loss, training, inference and tests are implemented.
- Unit tests verify blur synthesis, target consistency and the physics layer.
- The full model has 342,703 trainable parameters and measured 10.30 ms per
  five-frame 64x128 clip (97.1 FPS) on the local GTX 1650.
- No paper result is claimed yet: Sintel/BSD experiments still need to be run.

## Limitations

- The local model assumes approximately constant motion during exposure.
- Tiny blur, severe blur, saturation, low texture and rolling-shutter vertical
  motion remain difficult.
- Synthetic flow-warp blur does not fully reproduce real sensor noise,
  response functions, occlusion and spatially varying exposure.
- A confidence score is not a substitute for calibration; reliability must be
  evaluated on held-out real videos.
