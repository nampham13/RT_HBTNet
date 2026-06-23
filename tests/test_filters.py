from __future__ import annotations

from utils.filters import EMAFilter


def test_ema_first_update() -> None:
    filt = EMAFilter(alpha=0.5)
    assert filt.update(10.0) == 10.0


def test_ema_smoothing() -> None:
    filt = EMAFilter(alpha=0.5)
    filt.update(10.0)
    assert filt.update(20.0) == 15.0
def test_ema_reset() -> None:
    filt = EMAFilter(alpha=0.5)
    filt.update(10.0)
    filt.reset()
    assert filt.update(20.0) == 20.0
