from __future__ import annotations

import numpy as np
import torch

from utils.visualization import flow_to_color, local_alpha_ratio, make_sample_visualization, summarize_sample


def _sample() -> dict:
    frames = torch.zeros(3, 1, 8, 12)
    frames[0] += 0.2
    frames[1] += 0.5
    frames[2] += 0.8
    motion = torch.zeros(2, 8, 12)
    motion[0] = 4.0
    blur = motion * 0.5
    valid = torch.ones(1, 8, 12)
    return {
        "frames": frames,
        "alpha": torch.tensor([0.5]),
        "motion_flow": motion,
        "blur_flow": blur,
        "valid_mask": valid,
        "scene": "toy_scene",
    }


def test_flow_to_color_returns_rgb_image() -> None:
    flow = np.zeros((2, 8, 12), dtype=np.float32)
    flow[0] = 3.0
    rgb = flow_to_color(flow)

    assert rgb.shape == (8, 12, 3)
    assert rgb.dtype == np.uint8


def test_local_alpha_ratio_recovers_synthetic_alpha() -> None:
    sample = _sample()
    ratio, mask = local_alpha_ratio(
        sample["motion_flow"],
        sample["blur_flow"],
        valid_mask=sample["valid_mask"],
    )

    assert mask.all()
    assert np.allclose(ratio[mask], 0.5)


def test_make_sample_visualization_and_summary() -> None:
    sample = _sample()
    image = make_sample_visualization(sample, index=0, panel_width=64, panel_height=32)
    summary = summarize_sample(sample)

    assert image.ndim == 3
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    assert summary["scene"] == "toy_scene"
    assert np.isclose(float(summary["local_alpha_mean"]), 0.5)
