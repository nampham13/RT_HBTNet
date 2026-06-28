# Research and evaluation protocol

## Research question

Can a lightweight network estimate video exposure fraction by separately
learning inter-frame displacement and within-exposure blur, then enforcing
their physical relationship?

## Main hypothesis

For observable patches with approximately constant local velocity,

\[
\mathbf b_i \simeq \alpha\mathbf d_i.
\]

The global estimate is a robust uncertainty-weighted projection:

\[
\hat\alpha =
\operatorname{clip}_{[0,1]}
\frac{\sum_i w_i |\mathbf b_i^\top\mathbf d_i|}
     {\sum_i w_i \|\mathbf d_i\|_2^2+\epsilon}.
\]

The absolute dot product handles the sign ambiguity of a linear blur kernel.

## Datasets

1. MPI Sintel or Spring with flow GT for controlled synthetic training.
2. Adobe240/high-FPS video as a second synthetic-blur domain if time permits.
3. Beam-Splitter Dataset for real exposure-fraction evaluation.

Splits must be video/scene-disjoint. No frame or crop from one scene may
appear in both training and validation.

## Baselines

Required:

1. Constant training-set median.
2. Direct clip-to-alpha regression using the same encoder budget.
3. Blur-only alpha regression.
4. Temporal-only alpha regression.
5. Korčák–Matas: RAFT + Gong blur flow + robust ratio.
6. Full blur-temporal physics model.

Implemented training switches:

```powershell
# Direct clip-to-alpha baseline
python scripts/train.py --config configs/default.yaml --prediction-mode direct --alpha-only

# Single-cue baselines
python scripts/train.py --config configs/default.yaml --prediction-mode blur_only --alpha-only
python scripts/train.py --config configs/default.yaml --prediction-mode temporal_only --alpha-only

# Full model
python scripts/train.py --config configs/default.yaml --prediction-mode physics
```

Useful component substitutions:

- RAFT flow + learned blur branch + physics layer.
- Learned temporal flow + Gong blur flow + physics layer.
- Exposure Trajectory Recovery replacing linear blur flow.

## Ablations

- without context quality;
- without uncertainty weighting;
- without dense local-ratio supervision;
- learned scalar fusion instead of physics layer;
- without directional blur descriptors;
- 3, 5, 7 and 9 input frames;
- synthetic-only versus scene-disjoint real adaptation;
- alpha ranges: `<0.1`, `0.1–0.3`, `>0.3`;
- normal light versus low-light/low-texture subsets.

## Metrics

Primary:

- exposure-fraction MAE;
- exposure-fraction RMSE;
- median absolute error.

Diagnostic:

- temporal motion EPE;
- blur-flow sign-invariant EPE;
- expected calibration error or risk–coverage curve for confidence;
- parameters, latency, FPS and peak memory.

Report per-alpha-bin performance. A single aggregate MAE hides the known
failure modes at very short and very long exposures.

The constant baseline must be computed on the training partition and passed
unchanged to evaluation. Computing it from test labels is leakage.

## Data visualization checks

Run dataset visualization before every serious training/evaluation run:

```powershell
python scripts/visualize_data.py --config configs/default.yaml --dataset exposure_flow --count 16
```

For synthetic flow data, inspect:

- temporal frame order and ROI crop;
- motion-flow direction and magnitude;
- blur-flow direction and magnitude;
- local alpha map implied by `blur_flow / motion_flow`;
- valid-mask coverage;
- `summary.csv` rows with tiny motion, low valid ratio or suspicious alpha.

For real-video manifests, inspect:

- frame sampling range;
- FPS and exposure-time conversion into alpha;
- scene/video identifiers used for disjoint splitting;
- clips with saturation, low texture or rolling-shutter artifacts.

Do not proceed to benchmark reporting if visualization reveals broken frame
order, wrong flow orientation, invalid alpha metadata or split leakage.

## Go/no-go criteria

- Synthetic validation MAE above 0.020: inspect renderer, supervision and
  identifiability before real-data experiments.
- BSD zero-shot MAE above 0.060: domain gap is too large for the current claim.
- BSD zero-shot MAE 0.040–0.050: viable only with a strong efficiency or
  uncertainty result.
- BSD MAE at or below 0.039: competitive with the direct published baseline.
- Scene-disjoint MAE at or below 0.035 plus real-time inference: strong target.

These thresholds are planning criteria, not predicted or achieved results.

## Integrity checklist

- Never report toy-data or training-set metrics as benchmark results.
- Separate measured results from forecasts.
- Publish seeds, scene splits, renderer settings and rejected samples.
- Include failed settings, especially direct end-to-end alpha regression.
- Do not claim the first exposure-fraction estimator; prior work exists.
- Describe synthetic-to-real limitations and linear-motion assumptions.
- Do not select BSD test clips based on model performance.
