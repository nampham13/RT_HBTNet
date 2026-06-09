from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .blur_physics_branch import BlurPhysicsDescriptor
from .blocks import (
    Conv2Plus1DBlock,
    MLPHead,
    MultiScaleTemporalPool,
    init_lightweight,
)
from .factory import EncoderComponentFactory, ModelAuxFactory, ModelBuildConfig, RTHBTNetFactory


class TemporalTextureHead(nn.Module):
    """Temporal speed head operating on shared per-frame feature maps."""

    def __init__(
        self,
        feature_dim: int = 64,
        dropout: float = 0.1,
        num_temporal_blocks: int = 3,
        pool_scales: tuple[int, ...] = (1, 2, 4),
        use_tsm: bool = True,
    ) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            *[
                Conv2Plus1DBlock(
                    channels=int(feature_dim),
                    dropout=float(dropout),
                    use_tsm=bool(use_tsm),
                )
                for _ in range(int(num_temporal_blocks))
            ]
        )
        self.pool = MultiScaleTemporalPool(
            channels=int(feature_dim),
            output_dim=int(feature_dim),
            scales=tuple(pool_scales),
            dropout=float(dropout),
        )
        self.speed_head = MLPHead(feature_dim, out_dim=1, hidden_dim=feature_dim, dropout=float(dropout))
        self.conf_head = MLPHead(feature_dim, out_dim=1, hidden_dim=feature_dim, dropout=float(dropout))
        init_lightweight(self)

    def forward(self, frame_features: torch.Tensor) -> dict[str, torch.Tensor]:
        if frame_features.ndim == 3:
            temporal_maps = frame_features.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # B,D,T,1,1
        elif frame_features.ndim == 5:
            temporal_maps = frame_features.permute(0, 2, 1, 3, 4).contiguous()  # B,D,T,H,W
        else:
            raise ValueError("frame_features must have shape B,T,D or B,T,D,H,W")

        temporal_maps = self.temporal(temporal_maps)  # B,D,T,H,W
        feat = self.pool(temporal_maps)  # B,D
        speed_tex = F.softplus(self.speed_head(feat))
        conf_logit = self.conf_head(feat)
        conf_tex = torch.sigmoid(conf_logit)
        return {
            "speed_tex": speed_tex,
            "conf_tex": conf_tex,
            "texture_features": feat,
            "confidence_logit": conf_logit.squeeze(-1),
        }


class BlurFeatureHead(nn.Module):
    """Blur speed head operating on shared features plus fixed blur physics."""

    def __init__(self, feature_dim: int = 64, in_channels: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.physics_descriptor = BlurPhysicsDescriptor(in_channels=int(in_channels))
        self.physics_proj = nn.Sequential(
            nn.Linear(self.physics_descriptor.summary_dim, int(feature_dim)),
            nn.LayerNorm(int(feature_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.feature_fuse = nn.Sequential(
            nn.Linear(int(feature_dim) * 2, int(feature_dim)),
            nn.LayerNorm(int(feature_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.speed_head = MLPHead(feature_dim, out_dim=1, hidden_dim=feature_dim, dropout=float(dropout))
        self.conf_head = MLPHead(feature_dim, out_dim=1, hidden_dim=feature_dim, dropout=float(dropout))
        init_lightweight(self)

    def forward(self, key_feature: torch.Tensor, key_frame: torch.Tensor) -> dict[str, torch.Tensor]:
        if key_feature.ndim != 2:
            raise ValueError("key_feature must have shape B,D")
        if key_frame.ndim != 4:
            raise ValueError("key_frame must have shape B,C,H,W")

        physics_summary = self.physics_descriptor.summary_features(key_frame)
        physics_feature = self.physics_proj(physics_summary)
        blur_feature = self.feature_fuse(torch.cat([key_feature, physics_feature], dim=1))

        speed_blur = F.softplus(self.speed_head(blur_feature))
        conf_blur = torch.sigmoid(self.conf_head(blur_feature))
        return {
            "speed_blur": speed_blur,
            "conf_blur": conf_blur,
            "blur_features": blur_feature,
        }


class RTHBTNet(nn.Module):
    """Full RT-HBTNet model.

    Input:
        ``x_seq`` with shape ``B,T,C,H,W``.

    Output:
        A dictionary containing fused speed/confidence and branch diagnostics.
        All speed, confidence, context-quality, and context-bias diagnostics use
        batch-aligned shapes. Scalar outputs use ``B,1``.
    """

    def __init__(
        self,
        in_channels: int | ModelBuildConfig = 1,
        base_channels: int = 24,
        temporal_hidden: int = 64,
        dropout: float = 0.1,
        use_context: bool = True,
        context_hidden: int | None = None,
        encoder: str = "mobilenetv3_small",
        encoder_truncate_at: int = 4,
        encoder_include_edges: bool = True,
        texture_num_blocks: int = 3,
        texture_pool_scales: tuple[int, ...] = (1, 2, 4),
        texture_use_tsm: bool = True,
        fusion_eps: float = 1.0e-6,
        min_confidence: float = 0.0,
    ) -> None:
        super().__init__()
        cfg = (
            in_channels
            if isinstance(in_channels, ModelBuildConfig)
            else ModelBuildConfig(
                in_channels=int(in_channels),
                base_channels=int(base_channels),
                temporal_hidden=int(temporal_hidden),
                dropout=float(dropout),
                use_context=bool(use_context),
                context_hidden=context_hidden,
                encoder=str(encoder),
                encoder_truncate_at=int(encoder_truncate_at),
                encoder_include_edges=bool(encoder_include_edges),
                texture_num_blocks=int(texture_num_blocks),
                texture_pool_scales=tuple(texture_pool_scales),
                texture_use_tsm=bool(texture_use_tsm),
                fusion_eps=float(fusion_eps),
                min_confidence=float(min_confidence),
            )
        )
        self.use_context = cfg.use_context
        self.in_channels = cfg.in_channels
        self.encoder_type = cfg.encoder_type
        self.encoder_include_edges = cfg.encoder_include_edges

        components = EncoderComponentFactory.create(cfg)
        self.frame_encoder = components.frame_encoder
        self.texture_head = components.texture_head
        self.blur_head = components.blur_head
        self.texture_branch = components.texture_branch
        self.blur_branch = components.blur_branch
        self.context_encoder = ModelAuxFactory.create_context_encoder(cfg)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x.repeat(self.in_channels, 1, 1, 1), persistent=False)
        self.register_buffer("sobel_y", sobel_y.repeat(self.in_channels, 1, 1, 1), persistent=False)
        self.fusion = ModelAuxFactory.create_fusion(cfg)

    def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_seq.ndim != 5:
            raise ValueError("x_seq must have shape B,T,C,H,W")
        if x_seq.shape[2] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x_seq.shape[2]}")

        if self.encoder_type in ("legacy", "separate", "separate_encoders"):
            if self.texture_branch is None or self.blur_branch is None:
                raise RuntimeError("legacy branches are not initialized")
            texture = self.texture_branch(x_seq)  # full sequence: B,T,C,H,W
            x_key = x_seq[:, -1]  # last frame: B,C,H,W
            blur = self.blur_branch(x_key)
            context = self._encode_context(
                texture_features=texture["texture_features"],
                blur_features=blur["blur_features"],
            )
        else:
            if self.frame_encoder is None or self.texture_head is None or self.blur_head is None:
                raise RuntimeError("shared encoder heads are not initialized")
            b, t, c, h, w = x_seq.shape
            x_flat = x_seq.reshape(b * t, c, h, w)
            descriptor = self._make_frame_descriptor(x_flat)
            feature_maps_flat = self.frame_encoder.forward_features(descriptor)  # B*T,D,H',W'
            _, d, feat_h, feat_w = feature_maps_flat.shape
            frame_maps = feature_maps_flat.reshape(b, t, d, feat_h, feat_w)  # B,T,D,H',W'
            frame_features = self.frame_encoder.pool_features(feature_maps_flat)
            frame_features = frame_features.reshape(b, t, -1)  # B,T,D
            texture = self.texture_head(frame_maps)
            blur = self.blur_head(frame_features[:, -1], x_seq[:, -1])
            context = self._encode_context(frame_features=frame_features)

        fused = self.fusion(
            speed_tex=texture["speed_tex"],  # B,1
            conf_tex=texture["conf_tex"],  # B,1
            speed_blur=blur["speed_blur"],  # B,1
            conf_blur=blur["conf_blur"],  # B,1
            context_bias=None if context is None else context["context_bias"],  # B,2
            obs_quality=None if context is None else context["obs_quality"],  # B,1
        )

        out = {
            "speed": fused["speed"],  # B,1
            "conf_final": fused["conf_final"],  # B,1
            "speed_tex": texture["speed_tex"],  # B,1
            "conf_tex": texture["conf_tex"],  # B,1
            "speed_blur": blur["speed_blur"],  # B,1
            "conf_blur": blur["conf_blur"],  # B,1
            "w_tex": fused["w_tex"],  # B,1
            "w_blur": fused["w_blur"],  # B,1
        }
        if context is not None:
            out.update(
                {
                    "obs_quality": context["obs_quality"],  # B,1
                    "context_bias_tex": context["context_bias_tex"],  # B,1
                    "context_bias_blur": context["context_bias_blur"],  # B,1
                }
            )
        return out

    def _encode_context(
        self,
        frame_features: torch.Tensor | None = None,
        *,
        texture_features: torch.Tensor | None = None,
        blur_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        if self.context_encoder is None:
            return None
        return self.context_encoder(
            frame_features=frame_features,
            texture_features=texture_features,
            blur_features=blur_features,
        )

    def _make_frame_descriptor(self, x_frame: torch.Tensor) -> torch.Tensor:
        if not self.encoder_include_edges:
            return x_frame

        grad_x = F.conv2d(x_frame, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x_frame, self.sobel_y, padding=1, groups=self.in_channels)
        edge_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
        return torch.cat([x_frame, edge_mag], dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model_from_config(config: dict[str, Any]) -> RTHBTNet:
    """Build ``RTHBTNet`` from the project config dictionary."""

    return RTHBTNetFactory.create(config)


if __name__ == "__main__":
    model = RTHBTNet()
    x = torch.randn(2, 64, 1, 64, 128)  # B,T,C,H,W
    y = model(x)
    print(f"parameters: {count_parameters(model)}")
    for name, tensor in y.items():
        print(f"{name}: {tuple(tensor.shape)}")
