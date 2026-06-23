from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn


@dataclass(frozen=True)
class ModelBuildConfig:
    """Normalized construction options for BT-ShutterNet."""

    in_channels: int = 1
    feature_dim: int = 64
    dropout: float = 0.1
    encoder: str = "mobilenetv3_small"
    encoder_truncate_at: int = 4
    encoder_include_edges: bool = True
    temporal_num_blocks: int = 3
    temporal_use_tsm: bool = True
    use_context_quality: bool = True
    prediction_mode: str = "physics"
    physics_eps: float = 1.0e-6
    min_motion_px: float = 0.25

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "ModelBuildConfig":
        model_cfg = config.get("model", {})
        physics_cfg = config.get("physics", {})
        return cls(
            in_channels=int(model_cfg.get("in_channels", 1)),
            feature_dim=int(model_cfg.get("feature_dim", model_cfg.get("temporal_hidden", 64))),
            dropout=float(model_cfg.get("dropout", 0.1)),
            encoder=str(model_cfg.get("encoder", "mobilenetv3_small")),
            encoder_truncate_at=int(model_cfg.get("encoder_truncate_at", 4)),
            encoder_include_edges=bool(model_cfg.get("encoder_include_edges", True)),
            temporal_num_blocks=int(model_cfg.get("temporal_num_blocks", 3)),
            temporal_use_tsm=bool(model_cfg.get("temporal_use_tsm", True)),
            use_context_quality=bool(model_cfg.get("use_context_quality", True)),
            prediction_mode=str(model_cfg.get("prediction_mode", "physics")),
            physics_eps=float(physics_cfg.get("eps", 1.0e-6)),
            min_motion_px=float(physics_cfg.get("min_motion_px", 0.25)),
        )


class BTShutterNetFactory:
    """Factory facade for constructing the exposure-fraction model."""

    @staticmethod
    def create(config: dict[str, Any]) -> nn.Module:
        from .rt_hbtnet import BTShutterNet

        return BTShutterNet(ModelBuildConfig.from_project_config(config))


# Backward-compatible project API while old checkpoints/scripts are retired.
RTHBTNetFactory = BTShutterNetFactory
