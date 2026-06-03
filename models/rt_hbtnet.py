from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .blur_physics_branch import BlurPhysicsBranch
from .fusion import ConfidenceAwareFusion
from .temporal_texture_branch import TemporalTextureBranch


class RTHBTNet(nn.Module):
    """Full RT-HBTNet model.

    Input:
        ``x_seq`` with shape ``B,T,C,H,W``.

    Output:
        A dictionary containing fused speed/confidence and branch diagnostics.
        All speed and confidence tensors use shape ``B,1``.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        temporal_hidden: int = 64,
        dropout: float = 0.1,
        use_context: bool = False,
        fusion_eps: float = 1.0e-6,
        min_confidence: float = 0.0,
    ) -> None:
        super().__init__()
        del use_context  # Reserved for future context inputs.
        self.texture_branch = TemporalTextureBranch(
            in_channels=int(in_channels),
            base_channels=int(base_channels),
            temporal_hidden=int(temporal_hidden),
            dropout=float(dropout),
        )
        self.blur_branch = BlurPhysicsBranch(
            in_channels=int(in_channels),
            base_channels=int(base_channels),
            feature_dim=int(temporal_hidden),
            dropout=float(dropout),
        )
        self.fusion = ConfidenceAwareFusion(eps=float(fusion_eps), min_confidence=float(min_confidence))

    def forward(self, x_seq: torch.Tensor) -> dict[str, torch.Tensor]:
        if x_seq.ndim != 5:
            raise ValueError("x_seq must have shape B,T,C,H,W")

        texture = self.texture_branch(x_seq)  # full sequence: B,T,C,H,W
        x_key = x_seq[:, -1]  # last frame: B,C,H,W
        blur = self.blur_branch(x_key)

        fused = self.fusion(
            speed_tex=texture["speed_tex"],  # B,1
            conf_tex=texture["conf_tex"],  # B,1
            speed_blur=blur["speed_blur"],  # B,1
            conf_blur=blur["conf_blur"],  # B,1
        )

        return {
            "speed": fused["speed"],  # B,1
            "conf_final": fused["conf_final"],  # B,1
            "speed_tex": texture["speed_tex"],  # B,1
            "conf_tex": texture["conf_tex"],  # B,1
            "speed_blur": blur["speed_blur"],  # B,1
            "conf_blur": blur["conf_blur"],  # B,1
            "w_tex": fused["w_tex"],  # B,1
            "w_blur": fused["w_blur"],  # B,1
        }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def build_model_from_config(config: dict[str, Any]) -> RTHBTNet:
    """Build ``RTHBTNet`` from the project config dictionary."""

    model_cfg = config.get("model", {})
    fusion_cfg = config.get("fusion", {})
    return RTHBTNet(
        in_channels=int(model_cfg.get("in_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 24)),
        temporal_hidden=int(model_cfg.get("temporal_hidden", model_cfg.get("feature_dim", 64))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_context=bool(model_cfg.get("use_context", False)),
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
