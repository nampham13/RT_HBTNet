from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .blur_physics_branch import BlurPhysicsBranch
from .blocks import (
    Conv2Plus1DBlock,
    MLPHead,
    MobileNetV3SmallFrameEncoder,
    MultiScaleTemporalPool,
    init_lightweight,
)
from .context_encoder import ObservationContextEncoder
from .fusion import ConfidenceAwareFusion
from .temporal_texture_branch import TemporalTextureBranch


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
    """Blur speed head operating on the shared key-frame feature."""

    def __init__(self, feature_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.speed_head = MLPHead(feature_dim, out_dim=1, hidden_dim=feature_dim, dropout=float(dropout))
        self.conf_head = MLPHead(feature_dim, out_dim=1, hidden_dim=feature_dim, dropout=float(dropout))
        init_lightweight(self)

    def forward(self, key_feature: torch.Tensor) -> dict[str, torch.Tensor]:
        if key_feature.ndim != 2:
            raise ValueError("key_feature must have shape B,D")

        speed_blur = F.softplus(self.speed_head(key_feature))
        conf_blur = torch.sigmoid(self.conf_head(key_feature))
        return {
            "speed_blur": speed_blur,
            "conf_blur": conf_blur,
            "blur_features": key_feature,
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
        in_channels: int = 1,
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
        self.use_context = bool(use_context)
        self.in_channels = int(in_channels)
        self.encoder_type = str(encoder).lower()
        self.encoder_include_edges = bool(encoder_include_edges)

        if self.encoder_type in ("legacy", "separate", "separate_encoders"):
            self.texture_branch = TemporalTextureBranch(
                in_channels=self.in_channels,
                base_channels=int(base_channels),
                temporal_hidden=int(temporal_hidden),
                dropout=float(dropout),
                num_temporal_blocks=int(texture_num_blocks),
                pool_scales=tuple(texture_pool_scales),
                use_tsm=bool(texture_use_tsm),
            )
            self.blur_branch = BlurPhysicsBranch(
                in_channels=self.in_channels,
                base_channels=int(base_channels),
                feature_dim=int(temporal_hidden),
                dropout=float(dropout),
            )
            self.frame_encoder = None
            self.texture_head = None
            self.blur_head = None
        elif self.encoder_type in ("mobilenetv3_small", "mobilenetv3_small_truncated"):
            descriptor_channels = self.in_channels * (2 if self.encoder_include_edges else 1)
            self.frame_encoder = MobileNetV3SmallFrameEncoder(
                in_ch=descriptor_channels,
                feature_dim=int(temporal_hidden),
                truncate_at=int(encoder_truncate_at),
            )
            self.texture_head = TemporalTextureHead(
                feature_dim=int(temporal_hidden),
                dropout=float(dropout),
                num_temporal_blocks=int(texture_num_blocks),
                pool_scales=tuple(texture_pool_scales),
                use_tsm=bool(texture_use_tsm),
            )
            self.blur_head = BlurFeatureHead(
                feature_dim=int(temporal_hidden),
                dropout=float(dropout),
            )
            self.texture_branch = None
            self.blur_branch = None
        else:
            raise ValueError(f"Unsupported encoder type: {encoder}")

        self.context_encoder = (
            ObservationContextEncoder(
                feature_dim=int(temporal_hidden),
                hidden_dim=context_hidden,
                dropout=float(dropout),
            )
            if self.use_context
            else None
        )

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
        self.fusion = ConfidenceAwareFusion(eps=float(fusion_eps), min_confidence=float(min_confidence))

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
            blur = self.blur_head(frame_features[:, -1])
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

    model_cfg = config.get("model", {})
    fusion_cfg = config.get("fusion", {})
    texture_pool_scales = tuple(
        int(scale) for scale in model_cfg.get("texture_pool_scales", (1, 2, 4))
    )
    return RTHBTNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 24)),
        temporal_hidden=int(model_cfg.get("temporal_hidden", model_cfg.get("feature_dim", 64))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_context=bool(model_cfg.get("use_context", True)),
        context_hidden=None if model_cfg.get("context_hidden") is None else int(model_cfg.get("context_hidden")),
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


if __name__ == "__main__":
    model = RTHBTNet()
    x = torch.randn(2, 64, 1, 64, 128)  # B,T,C,H,W
    y = model(x)
    print(f"parameters: {count_parameters(model)}")
    for name, tensor in y.items():
        print(f"{name}: {tuple(tensor.shape)}")
