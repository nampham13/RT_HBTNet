from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BlurPhysicsDescriptor(nn.Module):
    """Fixed spatial blur descriptors from edges and directional residuals."""

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

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_frame(frame)
        residual = self._directional_residuals(frame)
        spatial = torch.cat([frame, self._sobel_magnitude(frame), residual], dim=1)
        summary = torch.cat([self._fft_band_features(frame), self._directional_summary(residual)], dim=1)
        return spatial, summary

    def summary_features(self, frame: torch.Tensor) -> torch.Tensor:
        self._validate_frame(frame)
        residual = self._directional_residuals(frame)
        return torch.cat([self._fft_band_features(frame), self._directional_summary(residual)], dim=1)

    def _validate_frame(self, frame: torch.Tensor) -> None:
        if frame.ndim != 4:
            raise ValueError("frame must have shape B,C,H,W")
        if frame.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {frame.shape[1]}")

    @staticmethod
    def _make_directional_kernels(kernel_size: int) -> torch.Tensor:
        center = kernel_size // 2
        idx = torch.arange(kernel_size)
        kernels = torch.zeros(4, 1, kernel_size, kernel_size, dtype=torch.float32)
        kernels[0, 0, center, :] = 1.0
        kernels[1, 0, :, center] = 1.0
        kernels[2, 0, idx, idx] = 1.0
        kernels[3, 0, idx, kernel_size - 1 - idx] = 1.0
        return kernels / float(kernel_size)

    def _sobel_magnitude(self, frame: torch.Tensor) -> torch.Tensor:
        grad_x = F.conv2d(frame, self.sobel_x, padding=1, groups=self.in_channels)
        grad_y = F.conv2d(frame, self.sobel_y, padding=1, groups=self.in_channels)
        return torch.sqrt(grad_x.square() + grad_y.square() + self.eps)

    def _directional_residuals(self, frame: torch.Tensor) -> torch.Tensor:
        blur = F.conv2d(
            frame,
            self.directional_kernels,
            padding=self.directional_kernel_size // 2,
            groups=self.in_channels,
        )
        return (frame.repeat_interleave(self.num_directions, dim=1) - blur).abs()

    def _directional_summary(self, residual: torch.Tensor) -> torch.Tensor:
        batch = residual.shape[0]
        response = residual.reshape(
            batch,
            self.in_channels,
            self.num_directions,
            residual.shape[-2],
            residual.shape[-1],
        )
        energy = response.mean(dim=(1, 3, 4))
        anisotropy = (energy.amax(dim=1, keepdim=True) - energy.amin(dim=1, keepdim=True)) / (
            energy.mean(dim=1, keepdim=True) + self.eps
        )
        return torch.cat([energy, anisotropy], dim=1)

    def _fft_band_features(self, frame: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft2(frame.float(), norm="ortho")
        power = torch.log1p(spectrum.real.square() + spectrum.imag.square())
        height, width = int(frame.shape[-2]), int(frame.shape[-1])
        freq_y = torch.fft.fftfreq(height, d=1.0, device=frame.device)
        freq_x = torch.fft.rfftfreq(width, d=1.0, device=frame.device)
        radius = torch.sqrt(freq_y[:, None].square() + freq_x[None, :].square())
        radius = radius / radius.max().clamp_min(self.eps)
        flat_power = power.flatten(2)
        flat_radius = radius.flatten()
        bands = []
        for low, high in self.fft_bands:
            mask = ((flat_radius >= low) & (flat_radius < high)).to(flat_power.dtype)
            bands.append(
                (flat_power * mask.view(1, 1, -1)).sum(dim=(1, 2)).unsqueeze(1)
                / mask.sum().clamp_min(1.0)
            )
        band_features = torch.cat(bands, dim=1)
        low_mid = band_features[:, :2].sum(dim=1, keepdim=True)
        high_tail = band_features[:, 2:].sum(dim=1, keepdim=True)
        ratio = high_tail / (low_mid + self.eps)
        centroid = (flat_power * flat_radius).sum(dim=(1, 2)).unsqueeze(1) / (
            flat_power.sum(dim=(1, 2)).unsqueeze(1) + self.eps
        )
        return torch.cat([band_features, ratio, centroid], dim=1).to(frame.dtype)
