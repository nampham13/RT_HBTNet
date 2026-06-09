from __future__ import annotations

import math

import torch
from torch import nn

from .blocks import init_lightweight


class LightweightCrossAttention(nn.Module):
    """Bidirectional cross-attention over compact texture and blur feature tokens."""

    def __init__(
        self,
        feature_dim: int = 64,
        num_tokens: int = 4,
        token_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_tokens = int(num_tokens)
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        self.token_dim = int(token_dim or max(8, min(32, self.feature_dim // 2)))
        token_total = self.num_tokens * self.token_dim

        self.texture_norm = nn.LayerNorm(self.feature_dim)
        self.blur_norm = nn.LayerNorm(self.feature_dim)
        self.tex_q = nn.Linear(self.feature_dim, token_total)
        self.tex_k = nn.Linear(self.feature_dim, token_total)
        self.tex_v = nn.Linear(self.feature_dim, token_total)
        self.blur_q = nn.Linear(self.feature_dim, token_total)
        self.blur_k = nn.Linear(self.feature_dim, token_total)
        self.blur_v = nn.Linear(self.feature_dim, token_total)
        self.tex_from_blur_proj = nn.Sequential(
            nn.Linear(token_total, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        self.blur_from_tex_proj = nn.Sequential(
            nn.Linear(token_total, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
        )
        init_lightweight(self)

    def forward(
        self,
        texture_features: torch.Tensor,
        blur_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if texture_features.ndim != 2 or blur_features.ndim != 2:
            raise ValueError("texture_features and blur_features must have shape B,D")
        if texture_features.shape != blur_features.shape:
            raise ValueError("texture_features and blur_features must have matching shapes")
        if texture_features.shape[-1] != self.feature_dim:
            raise ValueError(f"expected feature dim {self.feature_dim}, got {texture_features.shape[-1]}")

        texture = self.texture_norm(texture_features)
        blur = self.blur_norm(blur_features)
        tex_from_blur_tokens, tex_to_blur_attn = self._attend(
            query=self._tokens(self.tex_q(texture)),
            key=self._tokens(self.blur_k(blur)),
            value=self._tokens(self.blur_v(blur)),
        )
        blur_from_tex_tokens, blur_to_tex_attn = self._attend(
            query=self._tokens(self.blur_q(blur)),
            key=self._tokens(self.tex_k(texture)),
            value=self._tokens(self.tex_v(texture)),
        )
        tex_from_blur = self.tex_from_blur_proj(tex_from_blur_tokens.flatten(1))
        blur_from_tex = self.blur_from_tex_proj(blur_from_tex_tokens.flatten(1))
        return tex_from_blur, blur_from_tex, tex_to_blur_attn, blur_to_tex_attn

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], self.num_tokens, self.token_dim)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(float(self.token_dim))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, value), attn


class ConfidenceAwareFusion(nn.Module):
    """Fuse texture and blur speeds with confidence-guided cross-attention."""

    def __init__(
        self,
        eps: float = 1.0e-6,
        min_confidence: float = 0.0,
        feature_dim: int = 64,
        num_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.min_confidence = float(min_confidence)
        self.feature_dim = int(feature_dim)
        self.cross_attention = LightweightCrossAttention(
            feature_dim=self.feature_dim,
            num_tokens=int(num_tokens),
            dropout=float(dropout),
        )
        gate_in_dim = self.feature_dim * 4 + 2
        self.cross_gate = nn.Sequential(
            nn.Linear(gate_in_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, 2),
        )
        init_lightweight(self.cross_gate)
        self._init_neutral_cross_gate()

    def forward(
        self,
        speed_tex: torch.Tensor,
        conf_tex: torch.Tensor,
        speed_blur: torch.Tensor,
        conf_blur: torch.Tensor,
        context_bias: torch.Tensor | None = None,
        obs_quality: torch.Tensor | None = None,
        texture_features: torch.Tensor | None = None,
        blur_features: torch.Tensor | None = None,
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
            texture_features: Optional texture feature vector shaped ``B,D``.
            blur_features: Optional blur feature vector shaped ``B,D``.

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

        attention_bias = None
        tex_to_blur_attn = None
        blur_to_tex_attn = None
        if texture_features is not None or blur_features is not None:
            if texture_features is None or blur_features is None:
                raise ValueError("texture_features and blur_features must be provided together")
            tex_from_blur, blur_from_tex, tex_to_blur_attn, blur_to_tex_attn = self.cross_attention(
                texture_features,
                blur_features,
            )
            cross_summary = torch.cat(
                [
                    texture_features,
                    blur_features,
                    tex_from_blur,
                    blur_from_tex,
                    torch.cat([conf_tex, conf_blur], dim=1),
                ],
                dim=1,
            )
            attention_bias = self.cross_gate(cross_summary)
            branch_scores = branch_scores + attention_bias

        branch_weights = torch.softmax(branch_scores, dim=1)  # B,2
        w_tex = branch_weights[:, :1]  # B,1
        w_blur = branch_weights[:, 1:2]  # B,1
        speed_fused = w_tex * speed_tex + w_blur * speed_blur  # B,1
        conf_final = torch.maximum(conf_tex, conf_blur)  # B,1
        if obs_quality is not None:
            if obs_quality.shape != conf_final.shape:
                raise ValueError(f"obs_quality must have shape {tuple(conf_final.shape)}, got {tuple(obs_quality.shape)}")
            conf_final = conf_final * torch.clamp(obs_quality, min=0.0, max=1.0)

        out = {
            "speed": speed_fused,
            "conf_final": conf_final,
            "w_tex": w_tex,
            "w_blur": w_blur,
        }
        if attention_bias is not None:
            out.update(
                {
                    "fusion_attention_bias_tex": attention_bias[:, :1],
                    "fusion_attention_bias_blur": attention_bias[:, 1:2],
                    "tex_to_blur_attention": tex_to_blur_attn,
                    "blur_to_tex_attention": blur_to_tex_attn,
                }
            )
        return out

    def _init_neutral_cross_gate(self) -> None:
        final = self.cross_gate[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)


# Backward-compatible alias for earlier prototype imports.
ConfidenceFusion = ConfidenceAwareFusion
