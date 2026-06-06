from __future__ import annotations

import math

import torch
from torch import nn

from .blocks import init_lightweight


class ObservationContextEncoder(nn.Module):
    """Estimate observation quality and branch trust bias.

    This module does not predict speed. It summarizes visual evidence from the
    shared frame features, or from already-computed branch features in the
    legacy model path, and produces:

    - branch bias logits for texture-vs-blur fusion
    - a scalar observation-quality score in ``[0, 1]``
    """

    def __init__(
        self,
        feature_dim: int = 64,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        hidden_dim = int(hidden_dim or feature_dim)
        summary_dim = self.feature_dim * 4

        self.encoder = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.bias_head = nn.Linear(hidden_dim, 2)
        self.quality_head = nn.Linear(hidden_dim, 1)

        init_lightweight(self)
        self._init_neutral_heads()

    def forward(
        self,
        frame_features: torch.Tensor | None = None,
        *,
        texture_features: torch.Tensor | None = None,
        blur_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return context branch-bias and observation-quality predictions."""

        if frame_features is not None:
            summary = self._summarize_frame_features(frame_features)
        else:
            summary = self._summarize_branch_features(texture_features, blur_features)

        context_features = self.encoder(summary)
        context_bias = self.bias_head(context_features)
        obs_quality = torch.sigmoid(self.quality_head(context_features))

        return {
            "context_features": context_features,
            "context_bias": context_bias,
            "context_bias_tex": context_bias[:, :1],
            "context_bias_blur": context_bias[:, 1:2],
            "obs_quality": obs_quality,
        }

    def _summarize_frame_features(self, frame_features: torch.Tensor) -> torch.Tensor:
        if frame_features.ndim != 3:
            raise ValueError("frame_features must have shape B,T,D")
        if frame_features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {frame_features.shape[-1]}")

        mean_feature = frame_features.mean(dim=1)
        key_feature = frame_features[:, -1]
        if frame_features.shape[1] > 1:
            std_feature = frame_features.std(dim=1, unbiased=False)
            delta_feature = (frame_features[:, 1:] - frame_features[:, :-1]).abs().mean(dim=1)
        else:
            std_feature = torch.zeros_like(mean_feature)
            delta_feature = torch.zeros_like(mean_feature)
        return torch.cat([mean_feature, key_feature, std_feature, delta_feature], dim=1)

    def _summarize_branch_features(
        self,
        texture_features: torch.Tensor | None,
        blur_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if texture_features is None or blur_features is None:
            raise ValueError("texture_features and blur_features are required when frame_features is not provided")
        if texture_features.ndim != 2 or blur_features.ndim != 2:
            raise ValueError("texture_features and blur_features must have shape B,D")
        if texture_features.shape != blur_features.shape:
            raise ValueError("texture_features and blur_features must have matching shapes")
        if texture_features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {texture_features.shape[-1]}")

        diff_feature = (texture_features - blur_features).abs()
        mean_feature = 0.5 * (texture_features + blur_features)
        return torch.cat([texture_features, blur_features, diff_feature, mean_feature], dim=1)

    def _init_neutral_heads(self) -> None:
        nn.init.zeros_(self.bias_head.weight)
        nn.init.zeros_(self.bias_head.bias)
        nn.init.zeros_(self.quality_head.weight)
        nn.init.constant_(self.quality_head.bias, math.log(0.75 / 0.25))
