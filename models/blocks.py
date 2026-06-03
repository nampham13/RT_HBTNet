from __future__ import annotations

import torch
from torch import nn


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
        if isinstance(layer, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
