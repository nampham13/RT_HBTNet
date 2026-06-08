from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small


def _resolve_channels(
    in_ch: int | None,
    out_ch: int | None,
    in_channels: int | None,
    out_channels: int | None,
) -> tuple[int, int]:
    """Support both short names and older in_channels/out_channels names."""

    resolved_in = in_ch if in_ch is not None else in_channels
    resolved_out = out_ch if out_ch is not None else out_channels
    if resolved_in is None or resolved_out is None:
        raise ValueError("Both input and output channels must be provided")
    return int(resolved_in), int(resolved_out)


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU(inplace=True)
    if name.lower() == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm2d -> activation.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: 2D convolution kernel size.
        stride: 2D convolution stride.
        padding: 2D convolution padding. Defaults to ``kernel_size // 2``.
    """

    def __init__(
        self,
        in_ch: int | None = None,
        out_ch: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        groups: int = 1,
        activation: str = "silu",
    ) -> None:
        in_ch, out_ch = _resolve_channels(in_ch, out_ch, in_channels, out_channels)
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            _activation(activation),
        )


class DepthwiseSeparableConv(nn.Sequential):
    """Lightweight depthwise Conv2d + pointwise Conv2d block.

    The depthwise convolution handles spatial filtering per channel, and the
    pointwise convolution cheaply mixes channels for real-time use.
    """

    def __init__(
        self,
        in_ch: int | None = None,
        out_ch: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        *,
        in_channels: int | None = None,
        out_channels: int | None = None,
        activation: str = "silu",
    ) -> None:
        in_ch, out_ch = _resolve_channels(in_ch, out_ch, in_channels, out_channels)
        if padding is None:
            padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_ch,
                in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            _activation(activation),
        )


class SmallFrameEncoder(nn.Module):
    """Tiny CNN frame encoder.

    Input:
        ``x`` with shape ``B,C,H,W``.
    Output:
        feature tensor with shape ``B,D``.
    """

    def __init__(
        self,
        in_ch: int = 1,
        feature_dim: int = 64,
        base_channels: int = 24,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = int(base_channels * 2)
        self.backbone = nn.Sequential(
            ConvBNAct(in_ch, c1, kernel_size=3, stride=2, padding=1, activation=activation),
            DepthwiseSeparableConv(c1, c2, kernel_size=3, stride=2, padding=1, activation=activation),
            DepthwiseSeparableConv(c2, c2, kernel_size=3, stride=2, padding=1, activation=activation),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.proj = nn.Linear(c2, int(feature_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W -> pooled: B,2*base_channels
        pooled = self.backbone(x)
        # pooled: B,2*base_channels -> features: B,D
        return self.proj(pooled)


class TemporalShift(nn.Module):
    """Shift feature channels across time without parameters or FLOPs.

    Input and output use video feature layout ``B,C,T,H,W``. A small channel
    fold is shifted one step backward and another fold one step forward, while
    the remaining channels stay aligned with the current frame.
    """

    def __init__(self, fold_div: int = 8) -> None:
        super().__init__()
        if int(fold_div) <= 0:
            raise ValueError("fold_div must be positive")
        self.fold_div = int(fold_div)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("x must have shape B,C,T,H,W")

        _, channels, timesteps, _, _ = x.shape
        fold = channels // self.fold_div
        if fold == 0 or timesteps <= 1:
            return x

        out = torch.zeros_like(x)
        out[:, :fold, :-1] = x[:, :fold, 1:]
        out[:, fold : 2 * fold, 1:] = x[:, fold : 2 * fold, :-1]
        out[:, 2 * fold :] = x[:, 2 * fold :]
        return out


class Conv2Plus1DBlock(nn.Module):
    """Residual separable ``(2+1)D`` convolution block for video features.

    The block applies a spatial ``1xKxK`` convolution followed by a temporal
    ``Kt x 1 x 1`` convolution. Depthwise spatial/temporal filters plus
    pointwise mixing keep the block compact enough for real-time ROI clips.
    """

    def __init__(
        self,
        channels: int,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "silu",
        use_tsm: bool = True,
        tsm_fold_div: int = 8,
    ) -> None:
        super().__init__()
        channels = int(channels)
        spatial_padding = spatial_kernel_size // 2
        temporal_padding = temporal_kernel_size // 2
        self.shift = TemporalShift(tsm_fold_div) if use_tsm else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(1, spatial_kernel_size, spatial_kernel_size),
                padding=(0, spatial_padding, spatial_padding),
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            _activation(activation),
            nn.Dropout3d(float(dropout)),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(temporal_kernel_size, 1, 1),
                padding=(temporal_padding, 0, 0),
                groups=channels,
                bias=False,
            ),
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.Dropout3d(float(dropout)),
        )
        self.out_act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("x must have shape B,C,T,H,W")
        shifted = self.shift(x)
        return self.out_act(x + self.net(shifted))


class MultiScaleTemporalPool(nn.Module):
    """Pool video features over time and multiple spatial ROI scales."""

    def __init__(
        self,
        channels: int,
        output_dim: int | None = None,
        scales: tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("at least one pooling scale is required")
        self.scales = tuple(int(scale) for scale in scales)
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("pooling scales must be positive")

        channels = int(channels)
        output_dim = int(output_dim or channels)
        pooled_dim = channels * sum(scale * scale for scale in self.scales)
        self.proj = nn.Sequential(
            nn.Linear(pooled_dim, output_dim),
            nn.LayerNorm(output_dim),
            _activation(activation),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("x must have shape B,C,T,H,W")

        pooled = [
            F.adaptive_avg_pool3d(x, output_size=(1, scale, scale)).flatten(1)
            for scale in self.scales
        ]
        return self.proj(torch.cat(pooled, dim=1))


def _last_conv_out_channels(module: nn.Module) -> int:
    """Return the output channels of the last Conv2d inside a module."""

    out_channels: int | None = None
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            out_channels = int(layer.out_channels)
    if out_channels is None:
        raise ValueError("module does not contain a Conv2d layer")
    return out_channels


class MobileNetV3SmallFrameEncoder(nn.Module):
    """Truncated MobileNetV3-Small frame encoder for small conveyor ROIs.

    The default truncation keeps MobileNetV3-Small features up to total stride
    8. For a 64x128 ROI this leaves an 8x16 feature map before pooling, which
    preserves more low-level texture and blur detail than a full classifier
    backbone.
    """

    def __init__(
        self,
        in_ch: int = 1,
        feature_dim: int = 64,
        truncate_at: int = 4,
    ) -> None:
        super().__init__()
        if truncate_at < 1:
            raise ValueError("truncate_at must keep at least one MobileNetV3 feature block")

        backbone = mobilenet_v3_small(weights=None)
        if truncate_at > len(backbone.features):
            raise ValueError(f"truncate_at={truncate_at} exceeds MobileNetV3 feature count {len(backbone.features)}")

        self.features = nn.Sequential(*list(backbone.features.children())[: int(truncate_at)])
        first_conv = self.features[0][0]
        if not isinstance(first_conv, nn.Conv2d):
            raise TypeError("unexpected MobileNetV3 first block layout")
        if int(in_ch) != int(first_conv.in_channels):
            self.features[0][0] = nn.Conv2d(
                int(in_ch),
                int(first_conv.out_channels),
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                dilation=first_conv.dilation,
                groups=1,
                bias=False,
                padding_mode=first_conv.padding_mode,
            )

        out_channels = _last_conv_out_channels(self.features)
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels, int(feature_dim), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(feature_dim)),
            nn.Hardswish(inplace=True),
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        init_lightweight(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W -> features: B,D
        return self.pool(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,C,H,W -> feature maps: B,D,H',W'
        return self.proj(self.features(x))

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,D,H',W' -> pooled features: B,D
        return self.pool(x)


class TemporalConvBlock(nn.Module):
    """Residual temporal Conv1d block.

    Input and output use shape ``B,D,T`` where ``T`` is sequence length.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            _activation(activation),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(channels),
            nn.Dropout(float(dropout)),
        )
        self.out_act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,D,T -> residual branch: B,D,T -> out: B,D,T
        return self.out_act(x + self.net(x))


class MLPHead(nn.Module):
    """Small prediction head.

    Input:
        ``x`` with shape ``B,D``.
    Output:
        tensor with shape ``B,out_dim``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or in_dim)
        self.net = nn.Sequential(
            nn.Linear(int(in_dim), hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,D -> y: B,out_dim
        return self.net(x)


def init_lightweight(module: nn.Module) -> None:
    """Kaiming init for Conv/Linear layers used in the lightweight blocks."""

    for layer in module.modules():
        if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
