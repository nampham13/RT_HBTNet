from __future__ import annotations

import torch

from models import RTHBTNet


def test_model_shapes_and_confidence_ranges() -> None:
    model = RTHBTNet().eval()
    x = torch.rand(2, 64, 1, 64, 128)  # B,T,C,H,W

    with torch.no_grad():
        out = model(x)

    expected_keys = {
        "speed",
        "conf_final",
        "speed_tex",
        "conf_tex",
        "speed_blur",
        "conf_blur",
        "w_tex",
        "w_blur",
        "obs_quality",
        "context_bias_tex",
        "context_bias_blur",
    }
    assert expected_keys.issubset(out.keys())

    for key in expected_keys:
        assert out[key].shape == (2, 1)

    for key in ("conf_final", "conf_tex", "conf_blur", "w_tex", "w_blur", "obs_quality"):
        assert torch.all(out[key] >= 0.0)
        assert torch.all(out[key] <= 1.0)


def test_model_without_context_omits_context_outputs() -> None:
    model = RTHBTNet(use_context=False).eval()
    x = torch.rand(2, 8, 1, 64, 128)  # B,T,C,H,W

    with torch.no_grad():
        out = model(x)

    assert "obs_quality" not in out
    assert "context_bias_tex" not in out
    assert "context_bias_blur" not in out
