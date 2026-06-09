from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvBNAct, DepthwiseSeparableConv, MLPHead, init_lightweight


class BlurPhysicsDescriptor(nn.Module):
    """Fixed blur descriptors from edges, directional kernels, and FFT bands."""

    def __init__(
        self,
        in_channels: int = 1,
        directional_kernel_size: int = 5,
        fft_bands: tuple[tuple[float, float], ...] = (
            (0.0, 0.18),
            (0.18, 0.36),
            (0.36, 0.68),
            (0.68, 1.01),
        ),
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.directional_kernel_size = int(directional_kernel_size)
        if self.directional_kernel_size < 3 or self.directional_kernel_size % 2 == 0:
            raise ValueError("directional_kernel_size must be an odd integer >= 3")

        self.fft_bands = tuple((float(lo), float(hi)) for lo, hi in fft_bands)
        if not self.fft_bands:
            raise ValueError("at least one FFT frequency band is required")
        self.eps = float(eps)
        self.num_directions = 4
        self.fft_feature_dim = len(self.fft_bands) + 2
        self.directional_feature_dim = self.num_directions + 1
        self.summary_dim = self.fft_feature_dim + self.directional_feature_dim
        self.spatial_descriptor_channels = self.in_channels * (2 + self.num_directions)

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

        kernels = self._make_directional_kernels(self.directional_kernel_size)
        self.register_buffer("directional_kernels", kernels.repeat(self.in_channels, 1, 1, 1), persistent=False)

    def forward(self, x_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return spatial descriptor maps and compact physics summary features."""

        self._validate_frame(x_frame)
        directional_residual = self._directional_residuals(x_frame)
        descriptor = torch.cat(
            [
                x_frame,
                self._sobel_magnitude(x_frame),
                directional_residual,
            ],
            dim=1,
        )
        summary = torch.cat(
            [
                self._fft_band_features(x_frame),
                self._directional_summary(directional_residual),
            ],
            dim=1,
        )
        return descriptor, summary

    def summary_features(self, x_frame: torch.Tensor) -> torch.Tensor:
        """Return the compact FFT and directional summary without Sobel maps."""

        self._validate_frame(x_frame)
        directional_residual = self._directional_residuals(x_frame)
        return torch.cat(
            [
                self._fft_band_features(x_frame),
                self._directional_summary(directional_residual),
            ],
            dim=1,
        )

    def _validate_frame(self, x_frame: torch.Tensor) -> None:
        if x_frame.ndim != 4:
            raise ValueError("x_frame must have shape B,C,H,W")
        if x_frame.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x_frame.shape[1]}")

    @staticmethod
    def _make_directional_kernels(kernel_size: int) -> torch.Tensor:
        center = kernel_size // 2
        idx = torch.arange(kernel_size)
        kernels = torch.zeros(4, 1, kernel_size, kernel_size, dtype=torch.float32)
        kernels[0, 0, center, :] = 1.0  # horizontal motion smear
        kernels[1, 0, :, center] = 1.0  # vertical motion smear
        kernels[2, 0, idx, idx] = 1.0  # down-right diagonal smear
        kernels[3, 0, idx, kernel_size - 1 - idx] = 1.0  # up-right diagonal smear
        return kernels / float(kernel_size)

    def _sobel_magnitude(self, x_frame: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(x_frame, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(x_frame, self.sobel_y, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _directional_residuals(self, x_frame: torch.Tensor) -> torch.Tensor:
        padding = self.directional_kernel_size // 2
        directional_blur = F.conv2d(
            x_frame,
            self.directional_kernels,
            padding=padding,
            groups=self.in_channels,
        )
        reference = x_frame.repeat_interleave(self.num_directions, dim=1)
        return (reference - directional_blur).abs()

    def _directional_summary(self, directional_residual: torch.Tensor) -> torch.Tensor:
        batch = directional_residual.shape[0]
        response = directional_residual.reshape(
            batch,
            self.in_channels,
            self.num_directions,
            directional_residual.shape[-2],
            directional_residual.shape[-1],
        )
        energy = response.mean(dim=(1, 3, 4))
        anisotropy = (energy.amax(dim=1, keepdim=True) - energy.amin(dim=1, keepdim=True)) / (
            energy.mean(dim=1, keepdim=True) + self.eps
        )
        return torch.cat([energy, anisotropy], dim=1)

    def _fft_band_features(self, x_frame: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return self._dft_band_features_for_export(x_frame)

        fft_input = x_frame.float()
        spectrum = torch.fft.rfft2(fft_input, norm="ortho")
        power = spectrum.real.square() + spectrum.imag.square()
        return self._power_band_features(power, x_frame)

    def _dft_band_features_for_export(self, x_frame: torch.Tensor) -> torch.Tensor:
        dft_input = x_frame.float()
        height = int(x_frame.shape[-2])
        width = int(x_frame.shape[-1])
        freq_width = width // 2 + 1
        dtype = dft_input.dtype
        device = x_frame.device

        row_idx = torch.arange(height, device=device, dtype=dtype)
        col_idx = torch.arange(width, device=device, dtype=dtype)
        freq_y = torch.arange(height, device=device, dtype=dtype)
        freq_x = torch.arange(freq_width, device=device, dtype=dtype)
        angle_x = -6.283185307179586 * col_idx[:, None] * freq_x[None, :] / float(width)
        angle_y = -6.283185307179586 * row_idx[:, None] * freq_y[None, :] / float(height)
        cos_x = torch.cos(angle_x)
        sin_x = torch.sin(angle_x)
        cos_y = torch.cos(angle_y)
        sin_y = torch.sin(angle_y)

        real_x = torch.matmul(dft_input, cos_x)
        imag_x = torch.matmul(dft_input, sin_x)
        real_x = real_x.transpose(-2, -1)
        imag_x = imag_x.transpose(-2, -1)

        real = torch.matmul(real_x, cos_y) - torch.matmul(imag_x, sin_y)
        imag = torch.matmul(real_x, sin_y) + torch.matmul(imag_x, cos_y)
        real = real.transpose(-2, -1)
        imag = imag.transpose(-2, -1)
        power = (real.square() + imag.square()) / float(height * width)
        return self._power_band_features(power, x_frame)

    def _power_band_features(self, power: torch.Tensor, x_frame: torch.Tensor) -> torch.Tensor:
        power = torch.log1p(power)

        height = int(x_frame.shape[-2])
        width = int(x_frame.shape[-1])
        freq_y = torch.fft.fftfreq(height, d=1.0, device=x_frame.device)
        freq_x = torch.fft.rfftfreq(width, d=1.0, device=x_frame.device)
        radius = torch.sqrt(freq_y[:, None].square() + freq_x[None, :].square())
        radius = radius / radius.max().clamp_min(self.eps)

        flat_power = power.flatten(2)
        flat_radius = radius.flatten()
        bands = []
        for low, high in self.fft_bands:
            mask = ((flat_radius >= low) & (flat_radius < high)).to(dtype=flat_power.dtype)
            weighted_power = flat_power * mask.view(1, 1, -1)
            bands.append(weighted_power.sum(dim=(1, 2)).unsqueeze(1) / mask.sum().clamp_min(1.0))
        band_features = torch.cat(bands, dim=1)

        low_mid = band_features[:, :2].sum(dim=1, keepdim=True)
        high_tail = band_features[:, 2:].sum(dim=1, keepdim=True)
        high_low_ratio = high_tail / (low_mid + self.eps)
        centroid = (flat_power * flat_radius).sum(dim=(1, 2)).unsqueeze(1) / (
            flat_power.sum(dim=(1, 2)).unsqueeze(1) + self.eps
        )
        return torch.cat([band_features, high_low_ratio, centroid], dim=1).to(dtype=x_frame.dtype)


class BlurPhysicsBranch(nn.Module):
    """Learned blur branch for single ROI frames.

    The branch combines CNN maps with fixed blur-physics descriptors:
    Sobel edge attenuation, directional line-blur residuals, and FFT radial
    frequency-band statistics. It still predicts a learned latent speed cue
    rather than an explicit metric blur length.
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
        self.physics_descriptor = BlurPhysicsDescriptor(in_channels=self.in_channels)
        descriptor_channels = self.physics_descriptor.spatial_descriptor_channels
        c1 = int(base_channels)
        c2 = int(feature_dim)

        self.encoder = nn.Sequential(
            ConvBNAct(descriptor_channels, c1, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(c1, c2, kernel_size=3, stride=2, padding=1),
            DepthwiseSeparableConv(c2, c2, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.physics_proj = nn.Sequential(
            nn.Linear(self.physics_descriptor.summary_dim, c2),
            nn.LayerNorm(c2),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.feature_fuse = nn.Sequential(
            nn.Linear(c2 * 2, c2),
            nn.LayerNorm(c2),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
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

        descriptor, physics_summary = self.physics_descriptor(x_frame)
        cnn_feat = self.encoder(descriptor)  # B,D
        physics_feat = self.physics_proj(physics_summary)  # B,D
        feat = self.feature_fuse(torch.cat([cnn_feat, physics_feat], dim=1))  # B,D

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
