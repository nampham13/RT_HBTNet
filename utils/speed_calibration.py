from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SpeedCalibrator:
    """Convert raw model speed to metric conveyor speed.

    Modes:
        ``none``: return raw model speed unchanged.
        ``scale``: return ``raw_speed * scale``.
        ``known_speed``: calibrate scale from known real conveyor speed and
        raw model estimates, then apply the learned scale.
    """

    mode: str = "none"
    scale: float = 1.0
    known_speed_mps: float | None = None
    raw_reference: float | None = None

    def calibrate_from_known_speed(self, raw_estimates: list[float] | np.ndarray, known_speed_mps: float) -> float:
        """Estimate scale from raw predictions and known real speed.

        Uses the median raw estimate for robustness. Returns the learned scale.
        """

        raw = np.asarray(raw_estimates, dtype=np.float32)
        raw = raw[np.isfinite(raw)]
        if raw.size == 0:
            raise ValueError("raw_estimates must contain at least one finite value")

        raw_median = float(np.median(raw))
        if abs(raw_median) < 1.0e-8:
            raise ValueError("median raw estimate is too close to zero for calibration")

        self.mode = "known_speed"
        self.known_speed_mps = float(known_speed_mps)
        self.raw_reference = raw_median
        self.scale = float(known_speed_mps) / raw_median
        return self.scale

    def apply(self, raw_speed: float) -> float:
        """Apply calibration and return a Python float speed."""

        if self.mode == "none":
            return float(raw_speed)
        if self.mode in ("scale", "known_speed"):
            return float(raw_speed) * float(self.scale)
        raise ValueError(f"Unsupported calibration mode: {self.mode}")

    def save(self, path: str | Path) -> None:
        """Save calibration settings to JSON."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "mode": self.mode,
            "scale": self.scale,
            "known_speed_mps": self.known_speed_mps,
            "raw_reference": self.raw_reference,
        }
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SpeedCalibrator":
        """Load calibration settings from JSON."""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            mode=str(data.get("mode", "none")),
            scale=float(data.get("scale", 1.0)),
            known_speed_mps=(
                None if data.get("known_speed_mps") is None else float(data.get("known_speed_mps"))
            ),
            raw_reference=None if data.get("raw_reference") is None else float(data.get("raw_reference")),
        )


def estimate_pixels_per_meter(marker_spacing_m: float, marker_spacing_px: float) -> float:
    """Estimate pixel-to-meter scale from a known marker distance."""

    if marker_spacing_m <= 0:
        raise ValueError("marker_spacing_m must be positive")
    return float(marker_spacing_px) / float(marker_spacing_m)


def robust_roi_fusion(
    predictions: list[dict[str, Any]],
    min_confidence: float = 0.05,
) -> dict[str, float | int | str | None]:
    """Fuse multi-ROI predictions with confidence rejection and median speed.

    Args:
        predictions: List of ROI prediction dictionaries containing ``speed``,
            ``conf_final``, ``conf_tex``, ``conf_blur``, ``w_tex``, and
            ``w_blur`` scalar-like values.
        min_confidence: Reject ROIs with ``conf_final`` below this threshold.

    Returns:
        Dictionary with ``speed_median``, ``confidence_mean``, ``valid_count``,
        and ``status``. If every ROI is low-confidence, ``speed_median`` is
        ``None`` and status is ``LOW CONFIDENCE``.
    """

    if not predictions:
        return {
            "speed_median": None,
            "confidence_mean": 0.0,
            "valid_count": 0,
            "status": "NO PREDICTIONS",
        }

    valid: list[dict[str, Any]] = []
    all_confidences: list[float] = []
    for pred in predictions:
        conf = float(pred.get("conf_final", 0.0))
        all_confidences.append(conf)
        if conf >= float(min_confidence):
            valid.append(pred)

    if not valid:
        return {
            "speed_median": None,
            "confidence_mean": float(np.mean(all_confidences)) if all_confidences else 0.0,
            "valid_count": 0,
            "status": "LOW CONFIDENCE",
        }

    speeds = [float(pred["speed"]) for pred in valid]
    confidences = [float(pred.get("conf_final", 0.0)) for pred in valid]
    return {
        "speed_median": float(np.median(speeds)),
        "confidence_mean": float(np.mean(confidences)),
        "valid_count": len(valid),
        "status": "OK",
    }
