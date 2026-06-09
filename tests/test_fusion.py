from __future__ import annotations

import torch

from models.fusion import ConfidenceAwareFusion, LightweightCrossAttention


def test_fusion_prefers_high_confidence_branch() -> None:
    fusion = ConfidenceAwareFusion()
    out = fusion(
        speed_tex=torch.tensor([[2.0]]),
        conf_tex=torch.tensor([[0.9]]),
        speed_blur=torch.tensor([[5.0]]),
        conf_blur=torch.tensor([[0.1]]),
    )

    assert out["speed"].item() < 3.5
    assert abs(out["speed"].item() - 2.0) < abs(out["speed"].item() - 5.0)


def test_context_bias_can_shift_branch_trust() -> None:
    fusion = ConfidenceAwareFusion()
    out = fusion(
        speed_tex=torch.tensor([[2.0]]),
        conf_tex=torch.tensor([[0.9]]),
        speed_blur=torch.tensor([[8.0]]),
        conf_blur=torch.tensor([[0.1]]),
        context_bias=torch.tensor([[-4.0, 4.0]]),
        obs_quality=torch.tensor([[0.25]]),
    )

    assert out["w_blur"].item() > out["w_tex"].item()
    assert out["speed"].item() > 5.0
    assert torch.isclose(out["conf_final"], torch.tensor([[0.225]])).all()


def test_lightweight_cross_attention_returns_token_attention() -> None:
    attention = LightweightCrossAttention(feature_dim=16, num_tokens=4, token_dim=8, dropout=0.0).eval()
    texture_features = torch.rand(2, 16)
    blur_features = torch.rand(2, 16)

    with torch.no_grad():
        tex_from_blur, blur_from_tex, tex_to_blur_attn, blur_to_tex_attn = attention(
            texture_features,
            blur_features,
        )

    assert tex_from_blur.shape == (2, 16)
    assert blur_from_tex.shape == (2, 16)
    assert tex_to_blur_attn.shape == (2, 4, 4)
    assert blur_to_tex_attn.shape == (2, 4, 4)
    assert torch.allclose(tex_to_blur_attn.sum(dim=-1), torch.ones(2, 4), atol=1.0e-6)
    assert torch.allclose(blur_to_tex_attn.sum(dim=-1), torch.ones(2, 4), atol=1.0e-6)


def test_fusion_uses_cross_attention_when_features_are_provided() -> None:
    fusion = ConfidenceAwareFusion(feature_dim=16, num_tokens=4, dropout=0.0).eval()

    with torch.no_grad():
        out = fusion(
            speed_tex=torch.tensor([[2.0], [3.0]]),
            conf_tex=torch.tensor([[0.7], [0.2]]),
            speed_blur=torch.tensor([[5.0], [7.0]]),
            conf_blur=torch.tensor([[0.3], [0.8]]),
            texture_features=torch.rand(2, 16),
            blur_features=torch.rand(2, 16),
        )

    assert out["speed"].shape == (2, 1)
    assert out["fusion_attention_bias_tex"].shape == (2, 1)
    assert out["fusion_attention_bias_blur"].shape == (2, 1)
    assert out["tex_to_blur_attention"].shape == (2, 4, 4)
    assert out["blur_to_tex_attention"].shape == (2, 4, 4)
    assert torch.allclose(out["w_tex"] + out["w_blur"], torch.ones(2, 1), atol=1.0e-6)
