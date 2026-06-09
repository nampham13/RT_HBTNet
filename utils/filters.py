from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EMAFilter:
    """Exponential moving average filter for scalar speed values."""

    alpha: float = 0.25
    previous: float | None = None

    def update(self, value: float) -> float:
        """Update the filter and return a smoothed Python float.

        First frame behavior: the first measurement is returned unchanged.
        Formula: ``smooth = alpha * current + (1 - alpha) * previous``.
        """

        current = float(value)
        if self.previous is None:
            self.previous = current
        else:
            self.previous = self.alpha * current + (1.0 - self.alpha) * self.previous
        return float(self.previous)

    def reset(self) -> None:
        """Clear the stored previous value."""

        self.previous = None


@dataclass
class SimpleKalmanFilter1D:
    """Simple scalar Kalman filter for conveyor speed stabilization."""

    process_noise: float = 0.01
    measurement_noise: float = 0.1
    velocity: float | None = None
    uncertainty: float = 1.0

    def update(self, measurement: float) -> float:
        """Update with one scalar speed measurement and return a Python float.

        The first measurement initializes the velocity state directly. Later
        updates use a one-state constant-velocity Kalman correction.
        """

        z = float(measurement)
        if self.velocity is None:
            self.velocity = z
            return float(self.velocity)

        self.uncertainty += float(self.process_noise)
        kalman_gain = self.uncertainty / (self.uncertainty + float(self.measurement_noise))
        self.velocity = self.velocity + kalman_gain * (z - self.velocity)
        self.uncertainty = (1.0 - kalman_gain) * self.uncertainty
        return float(self.velocity)

    def reset(self) -> None:
        """Reset the scalar velocity and uncertainty state."""

        self.velocity = None
        self.uncertainty = 1.0


class SpeedStabilizer:
    """Config-driven speed stabilizer using EMA, Kalman, or no filtering."""

    def __init__(self, config: dict[str, Any]) -> None:
        stab_cfg = config.get("stabilization", config)
        filter_type = str(stab_cfg.get("type", "ema")).lower()
        self.filter_type = filter_type

        if filter_type == "ema":
            self.filter = EMAFilter(alpha=float(stab_cfg.get("ema_alpha", 0.25)))
        elif filter_type == "kalman":
            kalman_cfg = stab_cfg.get("kalman", {})
            self.filter = SimpleKalmanFilter1D(
                process_noise=float(kalman_cfg.get("process_noise", 0.01)),
                measurement_noise=float(kalman_cfg.get("measurement_noise", 0.1)),
            )
        elif filter_type in ("none", "raw", "off"):
            self.filter = None
        else:
            raise ValueError(f"Unsupported stabilization type: {filter_type}")

    def update(self, value: float) -> float:
        """Return stabilized speed as a Python float."""

        current = float(value)
        if self.filter is None:
            return current
        return float(self.filter.update(current))

    def reset(self) -> None:
        """Reset the underlying filter state if filtering is enabled."""

        if self.filter is not None:
            self.filter.reset()


# Backward-compatible alias for earlier prototype imports.
KalmanFilter1D = SimpleKalmanFilter1D


class StabilizerFactory:
    """Factory facade for config-driven speed stabilizers."""

    @staticmethod
    def create(config: dict[str, Any]) -> SpeedStabilizer:
        return SpeedStabilizer(config)
