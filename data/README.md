# Data layout

## MPI Sintel-style supervised training

```text
data/raw/sintel/
  training/
    final/
      alley_1/
        frame_0001.png
        ...
    flow/
      alley_1/
        frame_0001.flo
        ...
```

The adapter scans odd-length clips and synthesizes a random exposure fraction
for each sample. Configuration lives under `data.datasets.exposure_flow`.

Important controls:

- `alpha_min`, `alpha_max`: training exposure range.
- `integration_samples`: samples used for exposure integration.
- `samples_per_clip`: deterministic alpha variants per source clip.
- `stride`: source-window stride.
- `max_flow`: invalid-flow rejection threshold.

Do not use random frame-level train/validation splitting. Scenes are the
independent statistical units.

## Beam-Splitter Dataset

BSD should be kept separately, for example:

```text
data/raw/bsd/
```

Its real exposure settings should be used for final testing and optional
scene-disjoint adaptation. Do not tune hyperparameters on the reported test
clips.

Create a CSV manifest:

```csv
video_path,exposure_time_ms,fps,start_frame,end_frame,scene
clips/example.mp4,8,15,0,99,scene_001
```

Evaluate with:

```powershell
python scripts/evaluate.py --config configs/default.yaml --weights runs/exposure/best.pt --dataset exposure_video --data-root data/raw/bsd --manifest data/raw/bsd/manifest.csv
```

## Toy data

`scripts/make_toy_exposure_data.py` writes:

```text
data/toy_exposure/
```

This data validates code paths only. It is not an experimental benchmark.
