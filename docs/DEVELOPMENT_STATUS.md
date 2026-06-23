# Development status — June 23, 2026

This file separates measured software diagnostics from paper-ready results.

## Implemented

- pivot from metric speed to exposure-fraction estimation;
- linear-light flow-warp blur renderer with known alpha;
- scene-disjoint dataset splitting;
- dense temporal-motion and blur-motion heads;
- uncertainty/context-weighted fixed physics estimator;
- direct, blur-only and temporal-only scalar baselines;
- real-video CSV adapter for known exposure settings;
- alpha metrics, per-alpha bins and risk–coverage evaluation;
- inference, benchmark, smoke-data generation and unit tests.

## Measured engineering results

Default model on the local NVIDIA GTX 1650:

- trainable parameters: 342,703;
- input: five grayscale frames at 64x128;
- mean latency: 10.299 ms;
- throughput: 97.10 clips/s.

Measurement used five warm-up iterations and twenty timed iterations. This is
an engineering benchmark, not an accuracy result.

## Toy diagnostic

The deterministic 32-scene toy dataset was split by scene:

- training samples: 96;
- validation samples: 32;
- physics model best validation alpha MAE: 0.1196;
- direct-regression best validation alpha MAE: 0.1581;
- fixed alpha=0.5 baseline MAE for the balanced toy alpha set: 0.275.

The physics model improved approximately 24% relative to direct regression on
this diagnostic. Dense motion EPE remained high, so these numbers only show
that the end-to-end system can learn a transferable blur/temporal ratio in a
controlled setting.

The earlier four-scene toy experiment failed to beat the constant baseline.
That negative result is retained conceptually because it demonstrates the
pipeline's sensitivity to scene diversity.

None of these toy numbers may be used as a paper result.

## Required before submission

1. Train on real benchmark-scale flow data such as Sintel or Spring.
2. Evaluate zero-shot on Beam-Splitter Dataset.
3. Reproduce the Korčák–Matas baseline or use its official implementation.
4. Run direct, blur-only, temporal-only and physics variants with at least
   three seeds.
5. Report confidence calibration and per-alpha failure analysis.
6. Verify renderer assumptions against high-frame-rate real integration.
7. Freeze the scene split and record hashes/configuration before final tuning.
