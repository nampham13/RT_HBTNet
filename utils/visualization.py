from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _format_value(value: Any, precision: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, str):
        return value
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def draw_rois(frame: np.ndarray, rois: list[list[int]] | list[tuple[int, int, int, int]]) -> np.ndarray:
    """Draw fixed ROI rectangles on a frame.

    Args:
        frame: BGR image with shape ``H,W,C``.
        rois: List of ``[x, y, w, h]`` rectangles.

    Returns:
        Copy of the input frame with clamped ROI rectangles drawn.
    """

    out = frame.copy()
    frame_h, frame_w = out.shape[:2]
    for idx, roi in enumerate(rois):
        x, y, w, h = [int(v) for v in roi]
        if w <= 0 or h <= 0 or frame_h <= 0 or frame_w <= 0:
            continue
        x0 = max(0, min(x, frame_w - 1))
        y0 = max(0, min(y, frame_h - 1))
        x1 = max(x0 + 1, min(x + w, frame_w))
        y1 = max(y0 + 1, min(y + h, frame_h))
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 255), 2)
        cv2.putText(out, f"ROI {idx}", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return out


def draw_speed_panel(frame: np.ndarray, info_dict: dict[str, Any]) -> np.ndarray:
    """Draw a readable speed/confidence panel on a BGR frame.

    Expected info keys:
        ``speed_raw``, ``speed_smooth``, ``conf_tex``, ``conf_blur``,
        ``w_tex``, ``w_blur``, ``fps``, and optional ``status``.
    """

    out = frame.copy()
    conf_tex = info_dict.get("conf_tex")
    conf_blur = info_dict.get("conf_blur")
    if "status" in info_dict:
        status = str(info_dict["status"])
    elif conf_tex is not None and conf_blur is not None and max(float(conf_tex), float(conf_blur)) >= 0.3:
        status = "OK"
    else:
        status = "LOW CONFIDENCE"

    status_color = (80, 220, 80) if status.startswith("OK") else (0, 180, 255)
    lines = [
        ("Speed raw", f"{_format_value(info_dict.get('speed_raw'))} m/s"),
        ("Speed smooth", f"{_format_value(info_dict.get('speed_smooth'))} m/s"),
        ("conf_tex", _format_value(conf_tex, 2)),
        ("conf_blur", _format_value(conf_blur, 2)),
        ("w_tex", _format_value(info_dict.get("w_tex"), 2)),
        ("w_blur", _format_value(info_dict.get("w_blur"), 2)),
        ("FPS", _format_value(info_dict.get("fps"), 1)),
    ]

    x0, y0 = 10, 10
    row_h = 24
    width = 360
    height = 48 + row_h * len(lines)
    cv2.rectangle(out, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1)
    cv2.rectangle(out, (x0, y0), (x0 + width, y0 + height), (80, 80, 80), 1)

    cv2.putText(out, status, (x0 + 12, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)
    for idx, (key, value) in enumerate(lines):
        y = y0 + 54 + idx * row_h
        cv2.putText(out, f"{key}:", (x0 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
        cv2.putText(out, value, (x0 + 170, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_dashboard_frame(frame: np.ndarray, roi_preview: np.ndarray | None, info_dict: dict[str, Any]) -> np.ndarray:
    """Compose a simple dashboard frame with main video, ROI preview, and panel."""

    out = draw_speed_panel(frame, info_dict)
    if roi_preview is None:
        return out

    preview = roi_preview
    if preview.ndim == 2:
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    preview_h = 120
    scale = preview_h / max(preview.shape[0], 1)
    preview_w = int(preview.shape[1] * scale)
    preview = cv2.resize(preview, (preview_w, preview_h), interpolation=cv2.INTER_AREA)

    margin = 10
    y0 = max(margin, out.shape[0] - preview_h - margin)
    x0 = margin
    x1 = min(out.shape[1], x0 + preview_w)
    y1 = min(out.shape[0], y0 + preview_h)
    preview = preview[: y1 - y0, : x1 - x0]

    cv2.rectangle(out, (x0 - 1, y0 - 22), (x1 + 1, y1 + 1), (20, 20, 20), -1)
    cv2.putText(out, "ROI preview", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    out[y0:y1, x0:x1] = preview
    cv2.rectangle(out, (x0, y0), (x1, y1), (80, 80, 80), 1)
    return out


def draw_speed_overlay(
    frame: np.ndarray,
    speed_mps: float | None,
    roi: tuple[int, int, int, int] | None = None,
    label: str = "RT-HBTNet",
) -> np.ndarray:
    """Backward-compatible simple overlay helper."""

    rois = [roi] if roi is not None else []
    out = draw_rois(frame, rois)
    return draw_speed_panel(out, {"speed_raw": speed_mps, "speed_smooth": speed_mps, "status": label})
