from __future__ import annotations

import torch

from rt_hbtnet.models import RTHBTNet


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
    }
    assert expected_keys.issubset(out.keys())

    for key in expected_keys:
        assert out[key].shape == (2, 1)

    for key in ("conf_final", "conf_tex", "conf_blur", "w_tex", "w_blur"):
        assert torch.all(out[key] >= 0.0)
        assert torch.all(out[key] <= 1.0)
