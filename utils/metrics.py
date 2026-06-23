from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


ArrayLike = np.ndarray | torch.Tensor | list[float] | tuple[float, ...]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a tensor-like value to a NumPy float array."""

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def mae(pred: ArrayLike, target: ArrayLike) -> float:
    """Mean absolute error: ``mean(abs(pred - target))``."""

    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    return float(np.mean(np.abs(pred_np - target_np)))


def rmse(pred: ArrayLike, target: ArrayLike) -> float:
    """Root mean squared error: ``sqrt(mean((pred - target)^2))``."""

    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    return float(np.sqrt(np.mean((pred_np - target_np) ** 2)))


def mape(pred: ArrayLike, target: ArrayLike, eps: float = 1.0e-6) -> float:
    """Mean absolute percentage error.

    Formula: ``mean(abs((pred - target) / (target + eps))) * 100``.
    """

    pred_np = _to_numpy(pred)
    target_np = _to_numpy(target)
    return float(np.mean(np.abs((pred_np - target_np) / (target_np + float(eps)))) * 100.0)


@dataclass
class AverageMeter:
    """Track running average for scalar values."""

    name: str = "meter"
    val: float = 0.0
    avg: float = 0.0
    sum: float = 0.0
    count: int = 0

    def reset(self) -> None:
        """Reset all accumulated state."""

        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add ``value`` repeated ``n`` times to the running average."""

        self.val = float(value)
        self.sum += float(value) * int(n)
        self.count += int(n)
        self.avg = self.sum / max(self.count, 1)

    def as_dict(self) -> dict[str, float | int | str]:
        """Return the meter state as a plain dictionary."""

        return {
            "name": self.name,
            "val": self.val,
            "avg": self.avg,
            "sum": self.sum,
            "count": self.count,
        }


def endpoint_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> float:
    """Mean endpoint error for dense two-dimensional vector fields."""

    error = torch.sqrt(torch.sum((pred.float() - target.float()) ** 2, dim=1))
    if valid_mask is not None:
        mask = valid_mask.float()
        if mask.ndim == 4:
            mask = mask[:, 0]
        denominator = mask.sum().clamp_min(1.0)
        return float((error * mask).sum().detach().cpu() / denominator.detach().cpu())
    return float(error.mean().detach().cpu())


def alpha_error_report(preds: ArrayLike, targets: ArrayLike) -> dict[str, Any]:
    """Return exposure-fraction regression metrics."""

    preds_np = _to_numpy(preds)
    targets_np = _to_numpy(targets)
    absolute = np.abs(preds_np - targets_np)
    return {
        "alpha_mae": float(np.mean(absolute)),
        "alpha_rmse": float(np.sqrt(np.mean((preds_np - targets_np) ** 2))),
        "alpha_median_ae": float(np.median(absolute)),
        "num_samples": int(np.size(preds_np)),
    }
