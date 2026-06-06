from __future__ import annotations

import torch

from models.fusion import ConfidenceAwareFusion


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
