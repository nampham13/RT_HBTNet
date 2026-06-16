from __future__ import annotations

from datasets import DatasetFactory
from models import RTHBTNetFactory
from models.rt_hbtnet import RTHBTNet, build_model_from_config
from utils.filters import SpeedStabilizer, StabilizerFactory


def _base_config() -> dict:
    return {
        "project": {"seed": 42},
        "data": {
            "image_size": {"height": 32, "width": 48},
            "sequence_length": 4,
            "grayscale": True,
            "normalize": True,
            "clahe": {"enabled": False},
        },
        "roi": {
            "mode": "full",
            "rois": [],
            "resize_width": 48,
            "resize_height": 32,
        },
        "model": {
            "in_channels": 1,
            "base_channels": 8,
            "temporal_hidden": 16,
            "texture_num_blocks": 1,
            "texture_pool_scales": [1, 2],
            "use_context": False,
        },
        "fusion": {"eps": 1.0e-6, "min_confidence": 0.0},
        "augmentation": {"enabled": False},
        "inference": {"target_fps": 30},
        "stabilization": {"type": "ema", "ema_alpha": 0.5},
    }


def test_model_factory_creates_shared_encoder_model() -> None:
    config = _base_config()
    config["model"]["encoder"] = "mobilenetv3_small"
    config["model"]["encoder_truncate_at"] = 2

    model = RTHBTNetFactory.create(config)

    assert isinstance(model, RTHBTNet)
    assert model.frame_encoder is not None
    assert model.texture_head is not None
    assert model.blur_head is not None
    assert model.texture_branch is None


def test_build_model_from_config_delegates_to_factory_for_legacy_model() -> None:
    config = _base_config()
    config["model"]["encoder"] = "legacy"

    model = build_model_from_config(config)

    assert isinstance(model, RTHBTNet)
    assert model.texture_branch is not None
    assert model.blur_branch is not None
    assert model.frame_encoder is None


def test_stabilizer_factory_creates_speed_stabilizer() -> None:
    stabilizer = StabilizerFactory.create(_base_config())

    assert isinstance(stabilizer, SpeedStabilizer)
    assert stabilizer.update(2.0) == 2.0
