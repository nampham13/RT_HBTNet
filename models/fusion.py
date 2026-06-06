from __future__ import annotations

import torch
from torch import nn


class ConfidenceAwareFusion(nn.Module):
    """Fuse texture and blur speeds using confidence and optional context bias."""

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
        context_bias: torch.Tensor | None = None,
        obs_quality: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Fuse branch predictions.

        Args:
            speed_tex: Texture speed tensor with shape ``B,1``.
            conf_tex: Texture confidence tensor with shape ``B,1`` in ``[0,1]``.
            speed_blur: Blur speed tensor with shape ``B,1``.
            conf_blur: Blur confidence tensor with shape ``B,1`` in ``[0,1]``.
            context_bias: Optional branch-bias logits shaped ``B,2``. Column 0
                biases texture trust and column 1 biases blur trust.
            obs_quality: Optional observation-quality tensor shaped ``B,1`` in
                ``[0,1]``. It scales the final confidence, not the speed.

        Returns:
            Dictionary containing fused speed/confidence and branch weights.
        """

        conf_tex_safe = torch.clamp(conf_tex, min=self.min_confidence) + self.eps
        conf_blur_safe = torch.clamp(conf_blur, min=self.min_confidence) + self.eps
        branch_scores = torch.log(torch.cat([conf_tex_safe, conf_blur_safe], dim=1))  # B,2
        if context_bias is not None:
            if context_bias.shape != branch_scores.shape:
                raise ValueError(
                    f"context_bias must have shape {tuple(branch_scores.shape)}, got {tuple(context_bias.shape)}"
                )
            branch_scores = branch_scores + context_bias

        branch_weights = torch.softmax(branch_scores, dim=1)  # B,2
        w_tex = branch_weights[:, :1]  # B,1
        w_blur = branch_weights[:, 1:2]  # B,1
        speed_fused = w_tex * speed_tex + w_blur * speed_blur  # B,1
        conf_final = torch.maximum(conf_tex, conf_blur)  # B,1
        if obs_quality is not None:
            if obs_quality.shape != conf_final.shape:
                raise ValueError(f"obs_quality must have shape {tuple(conf_final.shape)}, got {tuple(obs_quality.shape)}")
            conf_final = conf_final * torch.clamp(obs_quality, min=0.0, max=1.0)

        return {
            "speed": speed_fused,
            "conf_final": conf_final,
            "w_tex": w_tex,
            "w_blur": w_blur,
        }


# Backward-compatible alias for earlier prototype imports.
ConfidenceFusion = ConfidenceAwareFusion
