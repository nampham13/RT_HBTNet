from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np


ROI = tuple[int, int, int, int]
AUTO_MOTION_MODES = {"auto", "auto_motion", "motion"}


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


def is_auto_motion_mode(config: dict[str, Any]) -> bool:
    """Return true when ROI config requests motion-based auto detection."""

    roi_cfg = config.get("roi", config)
    return str(roi_cfg.get("mode", "fixed")).lower() in AUTO_MOTION_MODES


def _auto_motion_config(config: dict[str, Any]) -> dict[str, Any]:
    roi_cfg = config.get("roi", config)
    return dict(roi_cfg.get("auto_motion", {}))


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        gray = frame
    elif frame.ndim == 3 and frame.shape[2] == 1:
        gray = frame[:, :, 0]
    elif frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported frame shape for ROI detection: {frame.shape}")

    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


def _normalize_to_u8(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.uint8)
    max_value = float(np.max(values))
    if max_value <= 1.0e-6:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = np.clip(values.astype(np.float32) * (255.0 / max_value), 0, 255)
    return scaled.astype(np.uint8)


def _odd_kernel_size(value: int, minimum: int = 3) -> int:
    size = max(int(value), int(minimum))
    return size if size % 2 == 1 else size + 1


def _expand_roi(frame_shape: tuple[int, ...], roi: ROI, margin_ratio: float) -> ROI:
    x, y, w, h = roi
    margin_x = int(round(w * float(margin_ratio)))
    margin_y = int(round(h * float(margin_ratio)))
    return _clamp_roi(frame_shape, (x - margin_x, y - margin_y, w + 2 * margin_x, h + 2 * margin_y))


def detect_motion_rois(frames: Iterable[np.ndarray], config: dict[str, Any]) -> list[ROI]:
    """Detect conveyor ROIs from short-term motion and texture energy.

    The detector is intentionally lightweight: it accumulates frame differences
    and Sobel texture energy, thresholds the combined score map, then returns
    stable connected-component boxes. It is meant for warm-up ROI selection,
    not per-frame object tracking.
    """

    frame_list = [frame for frame in frames if frame is not None]
    if len(frame_list) < 2:
        return []

    first_shape = frame_list[0].shape
    frame_h, frame_w = first_shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return []

    auto_cfg = _auto_motion_config(config)
    max_rois = max(1, int(auto_cfg.get("max_rois", 1)))
    min_area_ratio = float(auto_cfg.get("min_area_ratio", 0.02))
    max_area_ratio = float(auto_cfg.get("max_area_ratio", 0.85))
    margin_ratio = float(auto_cfg.get("margin_ratio", 0.12))
    motion_threshold = float(auto_cfg.get("motion_threshold", 18.0))
    score_percentile = float(auto_cfg.get("score_percentile", 70.0))
    blur_kernel = _odd_kernel_size(int(auto_cfg.get("blur_kernel", 5)))
    morph_kernel = _odd_kernel_size(int(auto_cfg.get("morph_kernel", 7)))

    motion_acc = np.zeros((frame_h, frame_w), dtype=np.float32)
    texture_acc = np.zeros((frame_h, frame_w), dtype=np.float32)
    prev_gray = cv2.GaussianBlur(_to_gray_u8(frame_list[0]), (blur_kernel, blur_kernel), 0)
    usable_pairs = 0

    for frame in frame_list[1:]:
        if frame.shape[:2] != (frame_h, frame_w):
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        gray = cv2.GaussianBlur(_to_gray_u8(frame), (blur_kernel, blur_kernel), 0)
        motion_acc += cv2.absdiff(gray, prev_gray).astype(np.float32)

        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        texture_acc += cv2.magnitude(grad_x, grad_y)

        prev_gray = gray
        usable_pairs += 1

    if usable_pairs <= 0:
        return []

    motion_u8 = _normalize_to_u8(motion_acc / float(usable_pairs))
    texture_u8 = _normalize_to_u8(texture_acc / float(usable_pairs))
    score_u8 = cv2.addWeighted(motion_u8, 0.75, texture_u8, 0.25, 0.0)

    nonzero_scores = score_u8[score_u8 > 0]
    if nonzero_scores.size == 0:
        return []

    adaptive_threshold = float(np.percentile(nonzero_scores, np.clip(score_percentile, 0.0, 100.0)))
    threshold_value = max(float(auto_cfg.get("score_threshold", 20.0)), adaptive_threshold)
    motion_mask = motion_u8 >= motion_threshold
    score_mask = score_u8 >= threshold_value
    mask = np.where(score_mask & motion_mask, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel, morph_kernel))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(frame_h * frame_w)
    min_area = frame_area * min_area_ratio
    max_area = frame_area * max_area_ratio
    candidates: list[tuple[float, ROI]] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area < min_area or contour_area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        roi = _expand_roi(first_shape, (x, y, w, h), margin_ratio)
        candidates.append((contour_area, roi))

    if not candidates:
        nonzero = cv2.findNonZero(mask)
        if nonzero is None:
            return []
        x, y, w, h = cv2.boundingRect(nonzero)
        box_area = float(w * h)
        if min_area <= box_area <= max_area:
            candidates.append((box_area, _expand_roi(first_shape, (x, y, w, h), margin_ratio)))

    if max_rois == 1 and candidates:
        xs = [roi[0] for _, roi in candidates]
        ys = [roi[1] for _, roi in candidates]
        x1s = [roi[0] + roi[2] for _, roi in candidates]
        y1s = [roi[1] + roi[3] for _, roi in candidates]
        return [_clamp_roi(first_shape, (min(xs), min(ys), max(x1s) - min(xs), max(y1s) - min(ys)))]

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [roi for _, roi in candidates[:max_rois]]


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

    if mode == "fixed" or mode in AUTO_MOTION_MODES:
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
