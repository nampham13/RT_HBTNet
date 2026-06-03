from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MLPHead, SmallFrameEncoder, TemporalConvBlock, init_lightweight


class TemporalTextureBranch(nn.Module):
    """Temporal texture branch for ROI clip dynamics.

    The branch learns motion-sensitive texture changes over time without
    running explicit optical flow. Sequence length is flexible because the
    temporal dimension is processed by Conv1d blocks and averaged at the end.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        temporal_hidden: int = 64,
        dropout: float = 0.1,
        num_temporal_blocks: int = 3,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        feat_dim = int(feature_dim or temporal_hidden)
        self.frame_encoder = SmallFrameEncoder(
            in_ch=int(in_channels),
            feature_dim=feat_dim,
            base_channels=int(base_channels),
        )
        self.temporal = nn.Sequential(
            *[
                TemporalConvBlock(
                    channels=feat_dim,
                    kernel_size=3,
                    dropout=float(dropout),
                )
                for _ in range(int(num_temporal_blocks))
            ]
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

        b, t, c, h, w = x_seq.shape
        x = x_seq.reshape(b * t, c, h, w)  # B*T,C,H,W
        frame_feat = self.frame_encoder(x)  # B*T,D
        frame_feat = frame_feat.reshape(b, t, -1)  # B,T,D
        temporal_feat = frame_feat.transpose(1, 2)  # B,D,T
        temporal_feat = self.temporal(temporal_feat)  # B,D,T
        feat = temporal_feat.mean(dim=-1)  # B,D

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
