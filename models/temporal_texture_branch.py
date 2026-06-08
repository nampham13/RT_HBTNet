from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import Conv2Plus1DBlock, MLPHead, MultiScaleTemporalPool, init_lightweight


class TemporalTextureBranch(nn.Module):
    """Temporal texture branch for ROI clip dynamics.

    The branch learns motion-sensitive texture changes over time without
    running explicit optical flow. It keeps spatial feature maps through the
    temporal stack so local texture motion can be modeled before pooling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        temporal_hidden: int = 64,
        dropout: float = 0.1,
        num_temporal_blocks: int = 3,
        feature_dim: int | None = None,
        pool_scales: tuple[int, ...] = (1, 2, 4),
        use_tsm: bool = True,
    ) -> None:
        super().__init__()
        feat_dim = int(feature_dim or temporal_hidden)
        stem_channels = int(base_channels)
        self.in_channels = int(in_channels)
        self.stem = nn.Sequential(
            nn.Conv3d(
                self.in_channels,
                stem_channels,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(
                stem_channels,
                feat_dim,
                kernel_size=(1, 3, 3),
                stride=(1, 2, 2),
                padding=(0, 1, 1),
                bias=False,
            ),
            nn.BatchNorm3d(feat_dim),
            nn.SiLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            *[
                Conv2Plus1DBlock(
                    channels=feat_dim,
                    dropout=float(dropout),
                    use_tsm=bool(use_tsm),
                )
                for _ in range(int(num_temporal_blocks))
            ]
        )
        self.pool = MultiScaleTemporalPool(
            channels=feat_dim,
            output_dim=feat_dim,
            scales=tuple(pool_scales),
            dropout=float(dropout),
        )
        self.speed_head = MLPHead(feat_dim, out_dim=1, hidden_dim=feat_dim, dropout=float(dropout))
        self.conf_head = MLPHead(feat_dim, out_dim=1, hidden_dim=feat_dim, dropout=float(dropout))
        init_lightweight(self)

    def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        """Estimate texture-branch speed and confidence.

        Args:
            x_seq: ROI sequence tensor with shape ``B,T,C,H,W``.

        Returns:
            Dictionary with ``speed_tex`` and ``conf_tex`` shaped ``B,1`` plus
            ``texture_features`` shaped ``B,D``.
        """

        if x_seq.ndim != 5:
            raise ValueError("x_seq must have shape B,T,C,H,W")
        if x_seq.shape[2] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x_seq.shape[2]}")

        x = x_seq.permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,H,W
        texture_maps = self.stem(x)  # B,D,T,H',W'
        texture_maps = self.temporal(texture_maps)  # B,D,T,H',W'
        feat = self.pool(texture_maps)  # B,D

        speed_tex = F.softplus(self.speed_head(feat))  # B,1 non-negative
        conf_logit = self.conf_head(feat)  # B,1
        conf_tex = torch.sigmoid(conf_logit)  # B,1 in [0,1]
        return {
            "speed_tex": speed_tex,
            "conf_tex": conf_tex,
            "texture_features": feat,
            # Compatibility aliases for older fusion code.
            "speed": speed_tex.squeeze(-1),
            "confidence": conf_tex.squeeze(-1),
            "confidence_logit": conf_logit.squeeze(-1),
        }
