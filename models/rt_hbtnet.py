from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Conv2Plus1DBlock, ConvBNAct, MobileNetV3SmallFrameEncoder, init_lightweight
from .blur_descriptor import BlurPhysicsDescriptor
from .factory import BTShutterNetFactory, ModelBuildConfig


class DenseTemporalMotionHead(nn.Module):
    """Predict inter-frame displacement and aleatoric uncertainty."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_blocks: int = 3,
        dropout: float = 0.1,
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
                for _ in range(int(num_blocks))
            ]
        )
        self.refine = nn.Sequential(
            ConvBNAct(feature_dim * 2, feature_dim, kernel_size=3),
            ConvBNAct(feature_dim, feature_dim, kernel_size=3),
        )
        self.motion_head = nn.Conv2d(feature_dim, 2, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)
        init_lightweight(self.refine)
        init_lightweight(self.motion_head)
        init_lightweight(self.logvar_head)

    def forward(self, frame_maps: torch.Tensor) -> dict[str, torch.Tensor]:
        if frame_maps.ndim != 5:
            raise ValueError("frame_maps must have shape B,T,D,H,W")
        maps = frame_maps.permute(0, 2, 1, 3, 4).contiguous()
        temporal = self.temporal(maps)
        center = temporal[:, :, temporal.shape[2] // 2]
        endpoint_delta = temporal[:, :, -1] - temporal[:, :, 0]
        feature = self.refine(torch.cat([center, endpoint_delta], dim=1))
        return {
            "motion_flow": self.motion_head(feature),
            "motion_logvar": torch.clamp(self.logvar_head(feature), -6.0, 6.0),
            "temporal_features": feature,
        }


class DenseBlurMotionHead(nn.Module):
    """Predict the exposure trajectory encoded by one blurred key frame."""

    def __init__(self, feature_dim: int = 64, in_channels: int = 1) -> None:
        super().__init__()
        self.physics_descriptor = BlurPhysicsDescriptor(in_channels=int(in_channels))
        in_dim = int(feature_dim) + self.physics_descriptor.spatial_descriptor_channels
        self.refine = nn.Sequential(
            ConvBNAct(in_dim, feature_dim, kernel_size=3),
            ConvBNAct(feature_dim, feature_dim, kernel_size=3),
        )
        self.blur_head = nn.Conv2d(feature_dim, 2, kernel_size=3, padding=1)
        self.logvar_head = nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1)
        init_lightweight(self.refine)
        init_lightweight(self.blur_head)
        init_lightweight(self.logvar_head)

    def forward(self, key_map: torch.Tensor, key_frame: torch.Tensor) -> dict[str, torch.Tensor]:
        descriptor, _ = self.physics_descriptor(key_frame)
        descriptor = F.interpolate(
            descriptor,
            size=key_map.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        feature = self.refine(torch.cat([key_map, descriptor], dim=1))
        return {
            "blur_flow": self.blur_head(feature),
            "blur_logvar": torch.clamp(self.logvar_head(feature), -6.0, 6.0),
            "blur_features": feature,
        }


class ObservationQualityHead(nn.Module):
    """Estimate locations where both blur and temporal cues are observable."""

    def __init__(self, feature_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(feature_dim * 3, feature_dim, kernel_size=3),
            nn.Conv2d(feature_dim, 1, kernel_size=3, padding=1),
        )
        init_lightweight(self.net)

    def forward(
        self,
        key_map: torch.Tensor,
        temporal_features: torch.Tensor,
        blur_features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([key_map, temporal_features, blur_features], dim=1)))


class ScalarAlphaHead(nn.Module):
    """Small scalar baseline head used only for controlled ablations."""

    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(int(in_channels), int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), 1),
        )
        init_lightweight(self.net)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feature))


class ExposurePhysicsLayer(nn.Module):
    """Robust weighted least-squares estimator of exposure fraction.

    Motion blur has an orientation ambiguity, so the projection uses the
    absolute dot product between blur and inter-frame displacement vectors.
    """

    def __init__(self, eps: float = 1.0e-6, min_motion_px: float = 0.25) -> None:
        super().__init__()
        self.eps = float(eps)
        self.min_motion_px = float(min_motion_px)

    def forward(
        self,
        motion_flow: torch.Tensor,
        blur_flow: torch.Tensor,
        motion_logvar: torch.Tensor,
        blur_logvar: torch.Tensor,
        context_quality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if motion_flow.shape != blur_flow.shape or motion_flow.shape[1] != 2:
            raise ValueError("motion_flow and blur_flow must have matching B,2,H,W shapes")

        motion_energy = motion_flow.square().sum(dim=1, keepdim=True)
        projected_blur = (motion_flow * blur_flow).sum(dim=1, keepdim=True).abs()
        alpha_map = projected_blur / (motion_energy + self.eps)

        uncertainty_weight = torch.exp(
            -0.5 * torch.clamp(motion_logvar + blur_logvar, min=-8.0, max=8.0)
        )
        observable = (motion_energy >= self.min_motion_px**2).to(motion_flow.dtype)
        weight = uncertainty_weight * observable
        if context_quality is not None:
            weight = weight * context_quality

        numerator = (weight * projected_blur).sum(dim=(2, 3))
        denominator = (weight * motion_energy).sum(dim=(2, 3)) + self.eps
        alpha_raw = numerator / denominator
        alpha = torch.clamp(alpha_raw, 0.0, 1.0)

        signed_residual = torch.minimum(
            (blur_flow - alpha[:, :, None, None] * motion_flow).square().sum(dim=1, keepdim=True),
            (blur_flow + alpha[:, :, None, None] * motion_flow).square().sum(dim=1, keepdim=True),
        ).sqrt()
        residual_score = torch.exp(
            -(weight * signed_residual).sum(dim=(2, 3))
            / (weight.sum(dim=(2, 3)) + self.eps)
        )
        coverage = observable.mean(dim=(2, 3))
        confidence = torch.clamp(residual_score * torch.sqrt(coverage + self.eps), 0.0, 1.0)
        return {
            "alpha": alpha,
            "alpha_raw": alpha_raw,
            "alpha_map": alpha_map,
            "physics_weight": weight,
            "confidence": confidence,
            "physics_residual": signed_residual,
        }


class BTShutterNet(nn.Module):
    """Lightweight blur-temporal exposure-fraction estimator."""

    def __init__(
        self,
        config: ModelBuildConfig | None = None,
        *,
        in_channels: int = 1,
        feature_dim: int = 64,
        dropout: float = 0.1,
        encoder_truncate_at: int = 4,
        encoder_include_edges: bool = True,
        temporal_num_blocks: int = 3,
        temporal_use_tsm: bool = True,
        use_context_quality: bool = True,
        prediction_mode: str = "physics",
        physics_eps: float = 1.0e-6,
        min_motion_px: float = 0.25,
    ) -> None:
        super().__init__()
        cfg = config or ModelBuildConfig(
            in_channels=int(in_channels),
            feature_dim=int(feature_dim),
            dropout=float(dropout),
            encoder_truncate_at=int(encoder_truncate_at),
            encoder_include_edges=bool(encoder_include_edges),
            temporal_num_blocks=int(temporal_num_blocks),
            temporal_use_tsm=bool(temporal_use_tsm),
            use_context_quality=bool(use_context_quality),
            prediction_mode=str(prediction_mode),
            physics_eps=float(physics_eps),
            min_motion_px=float(min_motion_px),
        )
        if cfg.encoder.lower() not in {"mobilenetv3_small", "mobilenetv3_small_truncated"}:
            raise ValueError("BT-ShutterNet currently supports the shared MobileNetV3-Small encoder")

        self.in_channels = cfg.in_channels
        self.encoder_include_edges = cfg.encoder_include_edges
        self.prediction_mode = cfg.prediction_mode.lower()
        supported_modes = {"physics", "direct", "blur_only", "temporal_only"}
        if self.prediction_mode not in supported_modes:
            raise ValueError(f"prediction_mode must be one of {sorted(supported_modes)}")
        descriptor_channels = self.in_channels * (2 if self.encoder_include_edges else 1)
        self.frame_encoder = MobileNetV3SmallFrameEncoder(
            in_ch=descriptor_channels,
            feature_dim=cfg.feature_dim,
            truncate_at=cfg.encoder_truncate_at,
        )
        self.temporal_head = DenseTemporalMotionHead(
            feature_dim=cfg.feature_dim,
            num_blocks=cfg.temporal_num_blocks,
            dropout=cfg.dropout,
            use_tsm=cfg.temporal_use_tsm,
        )
        self.blur_head = DenseBlurMotionHead(feature_dim=cfg.feature_dim, in_channels=self.in_channels)
        self.context_head = ObservationQualityHead(cfg.feature_dim) if cfg.use_context_quality else None
        self.physics = ExposurePhysicsLayer(eps=cfg.physics_eps, min_motion_px=cfg.min_motion_px)
        self.scalar_head: ScalarAlphaHead | None = None
        if self.prediction_mode == "direct":
            self.scalar_head = ScalarAlphaHead(cfg.feature_dim * 3, cfg.feature_dim)
        elif self.prediction_mode in {"blur_only", "temporal_only"}:
            self.scalar_head = ScalarAlphaHead(cfg.feature_dim, cfg.feature_dim)

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

    def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_seq.ndim != 5:
            raise ValueError("x_seq must have shape B,T,C,H,W")
        if x_seq.shape[2] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x_seq.shape[2]}")

        batch, timesteps, channels, height, width = x_seq.shape
        flat = x_seq.reshape(batch * timesteps, channels, height, width)
        descriptor = self._make_frame_descriptor(flat)
        maps_flat = self.frame_encoder.forward_features(descriptor)
        frame_maps = maps_flat.reshape(batch, timesteps, *maps_flat.shape[1:])
        center = timesteps // 2
        key_map = frame_maps[:, center]
        key_frame = x_seq[:, center]

        temporal = self.temporal_head(frame_maps)
        blur = self.blur_head(key_map, key_frame)
        context_quality = None
        if self.context_head is not None:
            context_quality = self.context_head(
                key_map,
                temporal["temporal_features"],
                blur["blur_features"],
            )
        physics = self.physics(
            motion_flow=temporal["motion_flow"],
            blur_flow=blur["blur_flow"],
            motion_logvar=temporal["motion_logvar"],
            blur_logvar=blur["blur_logvar"],
            context_quality=context_quality,
        )
        out = {
            **physics,
            "alpha_physics": physics["alpha"],
            "motion_flow": temporal["motion_flow"],
            "motion_logvar": temporal["motion_logvar"],
            "blur_flow": blur["blur_flow"],
            "blur_logvar": blur["blur_logvar"],
        }
        if self.scalar_head is not None:
            if self.prediction_mode == "direct":
                scalar_feature = torch.cat(
                    [key_map, temporal["temporal_features"], blur["blur_features"]],
                    dim=1,
                )
            elif self.prediction_mode == "blur_only":
                scalar_feature = blur["blur_features"]
            else:
                scalar_feature = temporal["temporal_features"]
            out["alpha"] = self.scalar_head(scalar_feature)
        if context_quality is not None:
            out["context_quality"] = context_quality
        return out

    def _make_frame_descriptor(self, frame: torch.Tensor) -> torch.Tensor:
        if not self.encoder_include_edges:
            return frame
        grad_x = F.conv2d(frame, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(frame, self.sobel_y, padding=1, groups=self.in_channels)
        edge = torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
        return torch.cat([frame, edge], dim=1)


# Keep the original import name while making the scientific target explicit.
RTHBTNet = BTShutterNet


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_model_from_config(config: dict[str, Any]) -> BTShutterNet:
    return BTShutterNetFactory.create(config)  # type: ignore[return-value]
