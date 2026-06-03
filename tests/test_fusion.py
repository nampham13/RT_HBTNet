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
