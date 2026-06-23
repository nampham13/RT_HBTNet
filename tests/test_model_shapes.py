from __future__ import annotations

import torch

from models import BTShutterNet
from models.blocks import TemporalShift
from models.rt_hbtnet import ExposurePhysicsLayer


def test_model_outputs_dense_cues_and_exposure_fraction() -> None:
    model = BTShutterNet(
        feature_dim=16,
        encoder_truncate_at=2,
        temporal_num_blocks=1,
    ).eval()
    x = torch.rand(2, 5, 1, 32, 48)

    with torch.no_grad():
        out = model(x)

    assert out["alpha"].shape == (2, 1)
    assert out["confidence"].shape == (2, 1)
    assert out["motion_flow"].shape[0:2] == (2, 2)
    assert out["blur_flow"].shape == out["motion_flow"].shape
    assert out["motion_logvar"].shape[0:2] == (2, 1)
    assert out["blur_logvar"].shape == out["motion_logvar"].shape
    assert out["alpha_map"].shape == out["motion_logvar"].shape
    assert out["physics_weight"].shape == out["motion_logvar"].shape
    assert out["context_quality"].shape == out["motion_logvar"].shape
    assert torch.all(out["alpha"] >= 0.0)
    assert torch.all(out["alpha"] <= 1.0)
    assert torch.all(out["confidence"] >= 0.0)
    assert torch.all(out["confidence"] <= 1.0)


def test_model_can_disable_context_quality() -> None:
    model = BTShutterNet(
        feature_dim=16,
        encoder_truncate_at=2,
        temporal_num_blocks=1,
        use_context_quality=False,
    ).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 3, 1, 32, 48))
    assert "context_quality" not in out


def test_direct_alpha_ablation_keeps_physics_diagnostic() -> None:
    model = BTShutterNet(
        feature_dim=16,
        encoder_truncate_at=2,
        temporal_num_blocks=1,
        prediction_mode="direct",
    ).eval()
    with torch.no_grad():
        out = model(torch.rand(2, 3, 1, 32, 48))
    assert out["alpha"].shape == (2, 1)
    assert out["alpha_physics"].shape == (2, 1)
    assert model.scalar_head is not None
    assert torch.all(out["alpha"] >= 0.0)
    assert torch.all(out["alpha"] <= 1.0)


def test_physics_layer_recovers_known_fraction_with_sign_ambiguity() -> None:
    physics = ExposurePhysicsLayer(min_motion_px=0.0)
    motion = torch.zeros(2, 2, 4, 6)
    motion[:, 0] = 3.0
    motion[:, 1] = 4.0
    blur = 0.4 * motion
    blur[1] *= -1.0
    zeros = torch.zeros(2, 1, 4, 6)

    out = physics(motion, blur, zeros, zeros)

    assert torch.allclose(out["alpha"], torch.full((2, 1), 0.4), atol=1.0e-5)
    assert torch.all(out["physics_residual"] < 1.0e-4)


def test_temporal_shift_has_no_parameters_and_keeps_shape() -> None:
    shift = TemporalShift(fold_div=4)
    x = torch.arange(1 * 8 * 4, dtype=torch.float32).reshape(1, 8, 4, 1, 1)
    out = shift(x)
    assert sum(param.numel() for param in shift.parameters()) == 0
    assert out.shape == x.shape
