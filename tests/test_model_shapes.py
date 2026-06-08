from __future__ import annotations

import torch

from models.blocks import TemporalShift
from models.rt_hbtnet import TemporalTextureHead
from models import RTHBTNet
from models.temporal_texture_branch import TemporalTextureBranch


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


def test_temporal_texture_head_accepts_feature_maps() -> None:
    head = TemporalTextureHead(
        feature_dim=16,
        num_temporal_blocks=1,
        pool_scales=(1, 2),
    ).eval()
    frame_maps = torch.rand(2, 5, 16, 8, 12)  # B,T,D,H,W

    with torch.no_grad():
        out = head(frame_maps)

    assert out["speed_tex"].shape == (2, 1)
    assert out["conf_tex"].shape == (2, 1)
    assert out["texture_features"].shape == (2, 16)
    assert torch.all(out["conf_tex"] >= 0.0)
    assert torch.all(out["conf_tex"] <= 1.0)

    frame_vectors = torch.rand(2, 5, 16)  # B,T,D
    with torch.no_grad():
        vector_out = head(frame_vectors)
    assert vector_out["speed_tex"].shape == (2, 1)
    assert vector_out["texture_features"].shape == (2, 16)


def test_legacy_temporal_texture_branch_uses_clip_inputs() -> None:
    branch = TemporalTextureBranch(
        in_channels=1,
        base_channels=8,
        temporal_hidden=16,
        num_temporal_blocks=1,
        pool_scales=(1, 2),
    ).eval()
    x = torch.rand(2, 5, 1, 48, 96)  # B,T,C,H,W

    with torch.no_grad():
        out = branch(x)

    assert out["speed_tex"].shape == (2, 1)
    assert out["conf_tex"].shape == (2, 1)
    assert out["texture_features"].shape == (2, 16)


def test_temporal_shift_has_no_parameters_and_keeps_shape() -> None:
    shift = TemporalShift(fold_div=4)
    x = torch.arange(1 * 8 * 4 * 1 * 1, dtype=torch.float32).reshape(1, 8, 4, 1, 1)

    out = shift(x)

    assert sum(param.numel() for param in shift.parameters()) == 0
    assert out.shape == x.shape
