from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from .blur_physics_branch import BlurPhysicsBranch
from .blocks import MobileNetV3SmallFrameEncoder
from .context_encoder import ObservationContextEncoder
from .fusion import ConfidenceAwareFusion
from .temporal_texture_branch import TemporalTextureBranch


@dataclass(frozen=True)
class ModelBuildConfig:
    """Normalized model construction options."""

    in_channels: int = 1
    base_channels: int = 24
    temporal_hidden: int = 64
    dropout: float = 0.1
    use_context: bool = True
    context_hidden: int | None = None
    encoder: str = "mobilenetv3_small"
    encoder_truncate_at: int = 4
    encoder_include_edges: bool = True
    texture_num_blocks: int = 3
    texture_pool_scales: tuple[int, ...] = (1, 2, 4)
    texture_use_tsm: bool = True
    fusion_eps: float = 1.0e-6
    min_confidence: float = 0.0

    @property
    def encoder_type(self) -> str:
        return self.encoder.lower()

    @classmethod
    def from_project_config(cls, config: dict[str, Any]) -> "ModelBuildConfig":
        model_cfg = config.get("model", {})
        fusion_cfg = config.get("fusion", {})
        texture_pool_scales = tuple(
            int(scale) for scale in model_cfg.get("texture_pool_scales", (1, 2, 4))
        )
        return cls(
            in_channels=int(model_cfg.get("in_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 24)),
            temporal_hidden=int(model_cfg.get("temporal_hidden", model_cfg.get("feature_dim", 64))),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_context=bool(model_cfg.get("use_context", True)),
            context_hidden=(
                None if model_cfg.get("context_hidden") is None else int(model_cfg.get("context_hidden"))
            ),
            encoder=str(model_cfg.get("encoder", "mobilenetv3_small")),
            encoder_truncate_at=int(model_cfg.get("encoder_truncate_at", 4)),
            encoder_include_edges=bool(model_cfg.get("encoder_include_edges", True)),
            texture_num_blocks=int(
                model_cfg.get("texture_num_blocks", model_cfg.get("num_temporal_blocks", 3))
            ),
            texture_pool_scales=texture_pool_scales,
            texture_use_tsm=bool(model_cfg.get("texture_use_tsm", True)),
            fusion_eps=float(fusion_cfg.get("eps", 1.0e-6)),
            min_confidence=float(fusion_cfg.get("min_confidence", 0.0)),
        )


@dataclass
class ModelComponents:
    """Components selected by the model factory."""

    frame_encoder: nn.Module | None
    texture_head: nn.Module | None
    blur_head: nn.Module | None
    texture_branch: nn.Module | None
    blur_branch: nn.Module | None


class EncoderComponentFactory:
    """Factory for mutually exclusive RT-HBTNet encoder/branch variants."""

    LEGACY_ENCODERS = {"legacy", "separate", "separate_encoders"}
    SHARED_ENCODERS = {"mobilenetv3_small", "mobilenetv3_small_truncated"}

    @classmethod
    def create(cls, cfg: ModelBuildConfig) -> ModelComponents:
        if cfg.encoder_type in cls.LEGACY_ENCODERS:
            return cls._create_legacy_components(cfg)
        if cfg.encoder_type in cls.SHARED_ENCODERS:
            return cls._create_shared_components(cfg)
        raise ValueError(f"Unsupported encoder type: {cfg.encoder}")

    @staticmethod
    def _create_legacy_components(cfg: ModelBuildConfig) -> ModelComponents:
        return ModelComponents(
            frame_encoder=None,
            texture_head=None,
            blur_head=None,
            texture_branch=TemporalTextureBranch(
                in_channels=cfg.in_channels,
                base_channels=cfg.base_channels,
                temporal_hidden=cfg.temporal_hidden,
                dropout=cfg.dropout,
                num_temporal_blocks=cfg.texture_num_blocks,
                pool_scales=cfg.texture_pool_scales,
                use_tsm=cfg.texture_use_tsm,
            ),
            blur_branch=BlurPhysicsBranch(
                in_channels=cfg.in_channels,
                base_channels=cfg.base_channels,
                feature_dim=cfg.temporal_hidden,
                dropout=cfg.dropout,
            ),
        )

    @staticmethod
    def _create_shared_components(cfg: ModelBuildConfig) -> ModelComponents:
        from .rt_hbtnet import BlurFeatureHead, TemporalTextureHead

        descriptor_channels = cfg.in_channels * (2 if cfg.encoder_include_edges else 1)
        return ModelComponents(
            frame_encoder=MobileNetV3SmallFrameEncoder(
                in_ch=descriptor_channels,
                feature_dim=cfg.temporal_hidden,
                truncate_at=cfg.encoder_truncate_at,
            ),
            texture_head=TemporalTextureHead(
                feature_dim=cfg.temporal_hidden,
                dropout=cfg.dropout,
                num_temporal_blocks=cfg.texture_num_blocks,
                pool_scales=cfg.texture_pool_scales,
                use_tsm=cfg.texture_use_tsm,
            ),
            blur_head=BlurFeatureHead(
                feature_dim=cfg.temporal_hidden,
                in_channels=cfg.in_channels,
                dropout=cfg.dropout,
            ),
            texture_branch=None,
            blur_branch=None,
        )


class ModelAuxFactory:
    """Factory for optional model helpers and fusion modules."""

    @staticmethod
    def create_context_encoder(cfg: ModelBuildConfig) -> ObservationContextEncoder | None:
        if not cfg.use_context:
            return None
        return ObservationContextEncoder(
            feature_dim=cfg.temporal_hidden,
            hidden_dim=cfg.context_hidden,
            dropout=cfg.dropout,
        )

    @staticmethod
    def create_fusion(cfg: ModelBuildConfig) -> ConfidenceAwareFusion:
        return ConfidenceAwareFusion(eps=cfg.fusion_eps, min_confidence=cfg.min_confidence)


class RTHBTNetFactory:
    """Factory facade for constructing RT-HBTNet from project config."""

    @staticmethod
    def create(config: dict[str, Any]) -> nn.Module:
        from .rt_hbtnet import RTHBTNet

        return RTHBTNet(ModelBuildConfig.from_project_config(config))
