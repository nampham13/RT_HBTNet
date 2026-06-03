from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from rt_hbtnet.utils.roi import ROI, crop_roi, resize_roi


def _data_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("data", config)


def _roi_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("roi", config)


def _target_size(config: dict[str, Any]) -> tuple[int, int]:
    """Return target size as ``(height, width)``."""

    roi_cfg = _roi_config(config)
    data_cfg = _data_config(config)
    if "resize_height" in roi_cfg and "resize_width" in roi_cfg:
        return int(roi_cfg["resize_height"]), int(roi_cfg["resize_width"])

    image_size = data_cfg.get("image_size", {"height": 64, "width": 128})
    if isinstance(image_size, dict):
        return int(image_size["height"]), int(image_size["width"])
    return int(image_size[0]), int(image_size[1])


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale if needed.

    OpenCV captures color frames in BGR order. Grayscale input with shape
    ``H,W`` is returned unchanged; ``H,W,1`` is squeezed to ``H,W``.
    """

    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported image shape for grayscale conversion: {img.shape}")


def apply_clahe(gray: np.ndarray, clip_limit: float, tile_grid_size: int | tuple[int, int]) -> np.ndarray:
    """Apply CLAHE to a grayscale image for low-light enhancement.

    Args:
        gray: Grayscale image with shape ``H,W``.
        clip_limit: CLAHE clip limit.
        tile_grid_size: Integer grid size or ``(width, height)`` tuple.
    """

    if gray.ndim != 2:
        raise ValueError("CLAHE expects a grayscale image with shape H,W")
    if isinstance(tile_grid_size, int):
        grid = (int(tile_grid_size), int(tile_grid_size))
    else:
        grid = (int(tile_grid_size[0]), int(tile_grid_size[1]))

    gray_u8 = gray
    if gray_u8.dtype != np.uint8:
        gray_u8 = np.clip(gray_u8, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid)
    return clahe.apply(gray_u8)


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert an image to ``float32`` in ``[0, 1]``."""

    if img.dtype == np.float32:
        if img.size == 0:
            return img
        if float(img.max()) <= 1.0 and float(img.min()) >= 0.0:
            return img
    return np.clip(img, 0, 255).astype(np.float32) / 255.0


def preprocess_roi(roi: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Run resize, optional grayscale, optional CLAHE, and normalization.

    Args:
        roi: Input ROI image, usually ``H,W,C`` from OpenCV BGR frames.
        config: Full config or data/ROI sub-config.

    Returns:
        Array with shape ``C,H,W``. If ``data.grayscale`` is true, ``C == 1``.
    """

    data_cfg = _data_config(config)
    target_h, target_w = _target_size(config)
    resized = resize_roi(roi, width=target_w, height=target_h)

    grayscale = bool(data_cfg.get("grayscale", True))
    normalize = bool(data_cfg.get("normalize", True))
    clahe_cfg = data_cfg.get("clahe", {})

    if grayscale:
        img = to_grayscale(resized)  # H,W
        if clahe_cfg.get("enabled", False):
            img = apply_clahe(
                img,
                clip_limit=float(clahe_cfg.get("clip_limit", 2.0)),
                tile_grid_size=clahe_cfg.get("tile_grid_size", 8),
            )
        img = normalize_image(img) if normalize else img.astype(np.float32)
        return np.ascontiguousarray(img[None, :, :])  # 1,H,W

    if resized.ndim == 2:
        resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    img = normalize_image(resized) if normalize else resized.astype(np.float32)
    return np.ascontiguousarray(img.transpose(2, 0, 1))  # C,H,W


def stack_sequence(rois_sequence: list[np.ndarray]) -> np.ndarray:
    """Stack preprocessed ROI frames into ``T,C,H,W`` order."""

    if not rois_sequence:
        raise ValueError("rois_sequence must contain at least one frame")
    return np.ascontiguousarray(np.stack(rois_sequence, axis=0).astype(np.float32))  # T,C,H,W


def preprocess_frame(
    frame: np.ndarray,
    image_size: Sequence[int] = (64, 128),
    channels: int = 1,
    roi: ROI | None = None,
) -> np.ndarray:
    """Backward-compatible wrapper returning one preprocessed ``C,H,W`` array."""

    height, width = int(image_size[0]), int(image_size[1])
    cfg = {
        "data": {
            "image_size": {"height": height, "width": width},
            "grayscale": int(channels) == 1,
            "normalize": True,
            "clahe": {"enabled": False},
        },
        "roi": {"resize_height": height, "resize_width": width},
    }
    return preprocess_roi(crop_roi(frame, roi), cfg)


def stack_frames(frames: list[np.ndarray], image_size: Sequence[int], channels: int, roi: ROI | None = None) -> np.ndarray:
    """Preprocess and stack raw video frames into ``T,C,H,W`` order."""

    arrays = [preprocess_frame(frame, image_size=image_size, channels=channels, roi=roi) for frame in frames]
    return stack_sequence(arrays)


def read_video_sequence(
    video_path: str | Path,
    sequence_len: int,
    image_size: Sequence[int],
    channels: int = 1,
    roi: ROI | None = None,
    start_frame: int = 0,
) -> np.ndarray:
    """Read a fixed-length video clip and return ``T,C,H,W`` as ``np.ndarray``."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_frame)))
    frames: list[np.ndarray] = []
    last = None
    for _ in range(int(sequence_len)):
        ok, frame = cap.read()
        if not ok:
            if last is None:
                break
            frame = last.copy()
        last = frame
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {video_path}")
    while len(frames) < sequence_len:
        frames.append(frames[-1].copy())
    return stack_frames(frames, image_size=image_size, channels=channels, roi=roi)
