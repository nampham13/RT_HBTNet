from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvBNAct, DepthwiseSeparableConv, MLPHead, init_lightweight


class BlurPhysicsBranch(nn.Module):
    """Learned latent blur branch for single ROI frames.

    This branch does not estimate explicit metric blur length and does not
    calculate optical flow. It learns a compact latent representation from the
    frame and a simple Sobel edge-magnitude channel, which helps expose
    speed-related smear and edge attenuation patterns under motion blur.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        feature_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        descriptor_channels = self.in_channels * 2
        c1 = int(base_channels)
        c2 = int(feature_dim)

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

        self.encoder = nn.Sequential(
            ConvBNAct(descriptor_channels, c1, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(c1, c2, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(c2, c2, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.speed_head = MLPHead(c2, out_dim=1, hidden_dim=c2, dropout=float(dropout))
        self.conf_head = MLPHead(c2, out_dim=1, hidden_dim=c2, dropout=float(dropout))
        init_lightweight(self)

    def forward(self, x_frame: torch.Tensor) -> dict[str, torch.Tensor]:
        """Estimate blur-branch speed and confidence.

        Args:
            x_frame: ROI frame tensor with shape ``B,C,H,W``.

        Returns:
            Dictionary with ``speed_blur`` and ``conf_blur`` shaped ``B,1`` plus
            ``blur_features`` shaped ``B,D``.
        """

        if x_frame.ndim != 4:
            raise ValueError("x_frame must have shape B,C,H,W")
        if x_frame.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x_frame.shape[1]}")

        edge_mag = self._sobel_magnitude(x_frame)  # B,C,H,W
        descriptor = torch.cat([x_frame, edge_mag], dim=1)  # B,2C,H,W
        feat = self.encoder(descriptor)  # B,D

        speed_blur = F.softplus(self.speed_head(feat))  # B,1 non-negative
        conf_logit = self.conf_head(feat)  # B,1
        conf_blur = torch.sigmoid(conf_logit)  # B,1 in [0,1]
        return {
            "speed_blur": speed_blur,
            "conf_blur": conf_blur,
            "blur_features": feat,
            # Compatibility aliases for older fusion code.
            "speed": speed_blur.squeeze(-1),
            "confidence": conf_blur.squeeze(-1),
            "confidence_logit": conf_logit.squeeze(-1),
        }

    def _sobel_magnitude(self, x_frame: torch.Tensor) -> torch.Tensor:
        # x_frame: B,C,H,W -> gradients: B,C,H,W
        grad_x = F.conv2d(x_frame, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x_frame, self.sobel_y, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-6)
