from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


ROI = tuple[int, int, int, int]


def parse_roi(value: Any) -> ROI | None:
    """Parse one ROI as ``(x, y, w, h)``, or return ``None`` for full frame."""

    if value in (None, "", "null"):
        return None
    if isinstance(value, dict):
        return (
            int(value["x"]),
            int(value["y"]),
            int(value["w"]),
            int(value["h"]),
        )
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(int(v) for v in value)  # type: ignore[return-value]
    raise ValueError("ROI must be null, a dict with x/y/w/h, or a 4-item list")


def _clamp_roi(frame_shape: tuple[int, ...], roi: Iterable[int]) -> ROI:
    """Clamp ``[x, y, w, h]`` to a valid image crop rectangle.

    If an ROI lies partially or fully outside the image, it is clamped to at
    least one pixel inside the frame so downstream resize calls stay valid.
    """

    x, y, w, h = (int(v) for v in roi)
    if w <= 0 or h <= 0:
        raise ValueError("ROI width and height must be positive")

    frame_h, frame_w = frame_shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        raise ValueError("frame must have positive height and width")

    x0 = max(0, min(x, frame_w - 1))
    y0 = max(0, min(y, frame_h - 1))
    x1 = max(x0 + 1, min(x + w, frame_w))
    y1 = max(y0 + 1, min(y + h, frame_h))
    return x0, y0, x1 - x0, y1 - y0


def crop_fixed_rois(frame: np.ndarray, rois: list[list[int]] | list[ROI]) -> list[np.ndarray]:
    """Crop fixed ROIs from an image.

    Args:
        frame: Image array with shape ``H,W,C`` or ``H,W``.
        rois: List of ``[x, y, w, h]`` rectangles in pixel coordinates.

    Returns:
        List of cropped ROI images. Each crop keeps the input channel layout.
    """

    crops: list[np.ndarray] = []
    for roi in rois:
        x, y, w, h = _clamp_roi(frame.shape, roi)
        crops.append(frame[y : y + h, x : x + w].copy())
    return crops


def resize_roi(roi: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize one ROI image to ``height,width`` using OpenCV."""

    return cv2.resize(roi, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def extract_rois(frame: np.ndarray, config: dict[str, Any]) -> list[np.ndarray]:
    """Extract and resize ROIs from a frame according to config.

    The function accepts either the full project config or just the ``roi``
    section. For the default config it returns images with shape
    ``resize_height,resize_width,C`` before preprocessing.
    """

    roi_cfg = config.get("roi", config)
    mode = roi_cfg.get("mode", "fixed")
    width = int(roi_cfg.get("resize_width", frame.shape[1]))
    height = int(roi_cfg.get("resize_height", frame.shape[0]))

    if mode == "fixed":
        rois = roi_cfg.get("rois", [])
        if not rois:
            return [resize_roi(frame, width=width, height=height)]
        crops = crop_fixed_rois(frame, rois)
        return [resize_roi(crop, width=width, height=height) for crop in crops]

    if mode in ("full", "full_frame", "none"):
        return [resize_roi(frame, width=width, height=height)]

    raise ValueError(f"Unsupported ROI mode: {mode}")


def crop_roi(frame: np.ndarray, roi: ROI | None) -> np.ndarray:
    """Backward-compatible helper for cropping one optional ROI."""

    if roi is None:
        return frame
    return crop_fixed_rois(frame, [roi])[0]


def draw_roi(frame: np.ndarray, roi: ROI | None, color: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """Draw one ROI rectangle on a copy of the frame."""

    out = frame.copy()
    if roi is not None:
        x, y, w, h = _clamp_roi(frame.shape, roi)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    return out
