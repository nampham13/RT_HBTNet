from __future__ import annotations

from models import BTShutterNet, BTShutterNetFactory
from models.rt_hbtnet import build_model_from_config


def _base_config() -> dict:
    return {
        "project": {"seed": 42},
        "model": {
            "encoder": "mobilenetv3_small",
            "encoder_truncate_at": 2,
            "encoder_include_edges": True,
            "in_channels": 1,
            "feature_dim": 16,
            "temporal_num_blocks": 1,
            "temporal_use_tsm": True,
            "use_context_quality": True,
            "dropout": 0.0,
        },
        "physics": {"eps": 1.0e-6, "min_motion_px": 0.1},
    }


def test_model_factory_creates_bt_shutternet() -> None:
    model = BTShutterNetFactory.create(_base_config())
    assert isinstance(model, BTShutterNet)
    assert model.frame_encoder is not None
    assert model.temporal_head is not None
    assert model.blur_head is not None
    assert model.context_head is not None


def test_build_model_from_config_uses_new_physics_pipeline() -> None:
    model = build_model_from_config(_base_config())
    assert isinstance(model, BTShutterNet)
    assert model.physics.min_motion_px == 0.1
