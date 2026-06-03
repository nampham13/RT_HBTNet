from __future__ import annotations

from rt_hbtnet.utils.filters import EMAFilter, SimpleKalmanFilter1D


def test_ema_first_update() -> None:
    filt = EMAFilter(alpha=0.5)
    assert filt.update(10.0) == 10.0


def test_ema_smoothing() -> None:
    filt = EMAFilter(alpha=0.5)
    filt.update(10.0)
    assert filt.update(20.0) == 15.0


def test_kalman_does_not_crash() -> None:
    filt = SimpleKalmanFilter1D(process_noise=0.01, measurement_noise=0.1)
    value = filt.update(1.0)
    value = filt.update(1.2)
    assert isinstance(value, float)
