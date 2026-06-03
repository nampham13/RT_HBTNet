from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _range(config: dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    values = config.get(key, default)
    return float(values[0]), float(values[1])


def random_brightness(img: np.ndarray, rng: np.random.Generator, scale_range: tuple[float, float]) -> np.ndarray:
    """Apply random brightness scaling to a uint8 image."""

    scale = float(rng.uniform(*scale_range))
    return np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def random_gamma(img: np.ndarray, rng: np.random.Generator, gamma_range: tuple[float, float]) -> np.ndarray:
    """Apply random gamma correction."""

    gamma = max(float(rng.uniform(*gamma_range)), 1.0e-6)
    normalized = img.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    return np.clip(corrected * 255.0, 0, 255).astype(np.uint8)


def add_gaussian_noise(img: np.ndarray, rng: np.random.Generator, std_range: tuple[float, float]) -> np.ndarray:
    """Add Gaussian noise where std is specified in normalized [0,1] units."""

    std = float(rng.uniform(*std_range))
    if std <= 0:
        return img
    noise = rng.normal(0.0, std * 255.0, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_motion_blur(
    img: np.ndarray,
    rng: np.random.Generator,
    prob: float,
    kernel_range: tuple[int, int],
) -> np.ndarray:
    """Apply horizontal motion blur with random odd kernel length."""

    if float(rng.random()) > float(prob):
        return img
    lo, hi = int(kernel_range[0]), int(kernel_range[1])
    k = int(rng.integers(max(1, lo), max(lo + 1, hi + 1)))
    if k % 2 == 0:
        k += 1
    if k <= 1:
        return img
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)


def add_dust_particles(img: np.ndarray, rng: np.random.Generator, prob: float) -> np.ndarray:
    """Draw small dust-like light/dark particles."""

    if float(rng.random()) > float(prob):
        return img
    out = img.copy()
    height, width = out.shape[:2]
    count = int(rng.integers(8, max(9, height * width // 450)))
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, 3))
        value = int(rng.integers(20, 245))
        color = value if out.ndim == 2 else (value, value, value)
        cv2.circle(out, (x, y), radius, color, -1)
    return out


def random_contrast(img: np.ndarray, rng: np.random.Generator, contrast_range: tuple[float, float]) -> np.ndarray:
    """Apply random contrast reduction or mild contrast scaling."""

    alpha = float(rng.uniform(*contrast_range))
    mean = float(np.mean(img))
    out = (img.astype(np.float32) - mean) * alpha + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def random_jpeg_artifacts(
    img: np.ndarray,
    rng: np.random.Generator,
    prob: float,
    quality_range: tuple[int, int],
) -> np.ndarray:
    """Simulate JPEG compression artifacts by encode/decode."""

    if float(rng.random()) > float(prob):
        return img
    lo, hi = int(quality_range[0]), int(quality_range[1])
    quality = int(rng.integers(max(1, lo), min(100, hi) + 1))
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    return img if decoded is None else decoded


def apply_low_light_blur_augmentations(
    img: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> np.ndarray:
    """Apply configurable low-light and blur augmentations.

    Args:
        img: uint8 grayscale or BGR image.
        rng: NumPy random generator.
        config: Full project config or an ``augmentation`` sub-dictionary.
    """

    aug = config.get("augmentation", config)
    if not aug.get("enabled", True):
        return img

    out = img.copy()
    out = random_brightness(out, rng, _range(aug, "brightness", (0.2, 1.2)))
    out = random_gamma(out, rng, _range(aug, "gamma", (0.6, 1.8)))
    out = random_contrast(out, rng, _range(aug, "contrast", (0.4, 1.0)))
    out = add_gaussian_noise(out, rng, _range(aug, "gaussian_noise_std", (0.0, 0.08)))
    out = add_dust_particles(out, rng, float(aug.get("dust_prob", 0.3)))
    out = random_motion_blur(
        out,
        rng,
        prob=float(aug.get("motion_blur_prob", 0.5)),
        kernel_range=tuple(int(v) for v in aug.get("motion_blur_kernel", (3, 15))),
    )
    out = random_jpeg_artifacts(
        out,
        rng,
        prob=float(aug.get("jpeg_prob", 0.0)),
        quality_range=tuple(int(v) for v in aug.get("jpeg_quality", (35, 95))),
    )
    return out
