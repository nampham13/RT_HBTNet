from __future__ import annotations

import torch
from torch import nn


class ConfidenceAwareFusion(nn.Module):
    """Fuse texture and blur speeds using confidence-ratio weights."""

    def __init__(self, eps: float = 1.0e-6, min_confidence: float = 0.0) -> None:
        super().__init__()
        self.eps = float(eps)
        self.min_confidence = float(min_confidence)

    def forward(
        self,
        speed_tex: torch.Tensor,
        conf_tex: torch.Tensor,
        speed_blur: torch.Tensor,
        conf_blur: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Fuse branch predictions.

        Args:
            speed_tex: Texture speed tensor with shape ``B,1``.
            conf_tex: Texture confidence tensor with shape ``B,1`` in ``[0,1]``.
            speed_blur: Blur speed tensor with shape ``B,1``.
            conf_blur: Blur confidence tensor with shape ``B,1`` in ``[0,1]``.

        Returns:
            Dictionary containing fused speed/confidence and branch weights.
        """

        conf_tex_safe = torch.clamp(conf_tex, min=self.min_confidence)
        conf_blur_safe = torch.clamp(conf_blur, min=self.min_confidence)
        denom = conf_tex_safe + conf_blur_safe + self.eps  # B,1
        w_tex = conf_tex_safe / denom  # B,1
        w_blur = conf_blur_safe / denom  # B,1
        speed_fused = w_tex * speed_tex + w_blur * speed_blur  # B,1
        conf_final = torch.maximum(conf_tex, conf_blur)  # B,1

        return {
            "speed": speed_fused,
            "conf_final": conf_final,
            "w_tex": w_tex,
            "w_blur": w_blur,
        }


# Backward-compatible alias for earlier prototype imports.
ConfidenceFusion = ConfidenceAwareFusion
