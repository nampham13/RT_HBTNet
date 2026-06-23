from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EMAFilter:
    """Exponential moving average for scalar exposure estimates."""

    alpha: float = 0.2
    previous: float | None = None

    def update(self, value: float) -> float:
        current = float(value)
        if self.previous is None:
            self.previous = current
        else:
            self.previous = self.alpha * current + (1.0 - self.alpha) * self.previous
        return float(self.previous)

    def reset(self) -> None:
        self.previous = None
