from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch


RGBColor = tuple[int, int, int]


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        arr = _as_numpy(value)
        return float(arr.reshape(-1)[0])
    except Exception:
        return float(default)


def image_to_rgb(image: Any, *, color_order: str = "bgr") -> np.ndarray:
    """Convert ``C,H,W`` or ``H,W,C`` image data into ``uint8`` RGB."""

    arr = _as_numpy(image)
    if arr.ndim == 3 and arr.shape[0] in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected image with 1 or 3 channels, got shape {arr.shape}")

    arr = arr.astype(np.float32)
    if arr.size and float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0:
        arr = arr * 255.0
    arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)

    if color_order.lower() == "bgr":
        arr = arr[:, :, ::-1]
    return np.ascontiguousarray(arr)


def _flow_to_hwc(flow: Any) -> np.ndarray:
    arr = _as_numpy(flow).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 2:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"Expected flow with shape 2,H,W or H,W,2, got {arr.shape}")
    return np.ascontiguousarray(arr)


def _mask_to_hw(mask: Any | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    arr = _as_numpy(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.shape != shape:
        arr = cv2.resize(arr.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return arr.astype(bool)


def robust_uint8(
    values: Any,
    *,
    mask: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    percentile: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    """Robustly normalize a scalar map into ``uint8``."""

    arr = _as_numpy(values).astype(np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Expected scalar map H,W, got {arr.shape}")

    valid = _mask_to_hw(mask, arr.shape)
    finite = np.isfinite(arr) & valid
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    lo = float(vmin) if vmin is not None else float(np.percentile(arr[finite], percentile[0]))
    hi = float(vmax) if vmax is not None else float(np.percentile(arr[finite], percentile[1]))
    if hi <= lo:
        hi = lo + 1.0e-6
    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    out[~finite] = 0.0
    return np.rint(out * 255.0).astype(np.uint8)


def flow_to_color(
    flow: Any,
    *,
    mask: Any | None = None,
    max_magnitude: float | None = None,
) -> np.ndarray:
    """Visualize optical flow using hue for direction and value for magnitude."""

    flow_hwc = _flow_to_hwc(flow)
    height, width = flow_hwc.shape[:2]
    valid = _mask_to_hw(mask, (height, width))
    u = flow_hwc[:, :, 0]
    v = flow_hwc[:, :, 1]
    mag = np.sqrt(u * u + v * v)
    angle = np.arctan2(v, u)

    finite = np.isfinite(mag) & valid
    if max_magnitude is None:
        max_magnitude = float(np.percentile(mag[finite], 95.0)) if np.any(finite) else 1.0
    max_magnitude = max(float(max_magnitude), 1.0e-6)

    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.rint(((angle + np.pi) / (2.0 * np.pi)) * 179.0).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip(np.rint(mag / max_magnitude * 255.0), 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb[~finite] = np.array([55, 55, 55], dtype=np.uint8)
    return rgb


def heatmap(
    values: Any,
    *,
    mask: Any | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
    gray = robust_uint8(values, mask=mask, vmin=vmin, vmax=vmax)
    bgr = cv2.applyColorMap(gray, cmap)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if mask is not None:
        valid = _mask_to_hw(mask, gray.shape)
        rgb[~valid] = np.array([55, 55, 55], dtype=np.uint8)
    return rgb


def mask_to_rgb(mask: Any) -> np.ndarray:
    arr = _as_numpy(mask)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    valid = arr.astype(bool)
    rgb = np.zeros((*valid.shape, 3), dtype=np.uint8)
    rgb[valid] = np.array([42, 180, 95], dtype=np.uint8)
    rgb[~valid] = np.array([190, 55, 55], dtype=np.uint8)
    return rgb


def local_alpha_ratio(
    motion_flow: Any,
    blur_flow: Any,
    *,
    valid_mask: Any | None = None,
    eps: float = 1.0e-6,
    min_motion_px: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute sign-invariant local alpha implied by blur and motion flow."""

    motion = _flow_to_hwc(motion_flow)
    blur = _flow_to_hwc(blur_flow)
    if motion.shape != blur.shape:
        raise ValueError("motion_flow and blur_flow must have the same shape")

    valid = _mask_to_hw(valid_mask, motion.shape[:2])
    dot = np.abs(np.sum(motion * blur, axis=2))
    motion_energy = np.sum(motion * motion, axis=2)
    ratio = dot / (motion_energy + float(eps))
    ratio_mask = valid & np.isfinite(ratio) & (motion_energy > float(min_motion_px) ** 2)
    ratio = np.clip(ratio, 0.0, 1.0).astype(np.float32)
    ratio[~ratio_mask] = 0.0
    return ratio, ratio_mask


def resize_rgb(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)


def with_label(
    image: np.ndarray,
    label: str,
    *,
    bar_height: int = 24,
    background: RGBColor = (25, 25, 25),
    foreground: RGBColor = (255, 255, 255),
) -> np.ndarray:
    image = np.ascontiguousarray(image)
    canvas = np.full((image.shape[0] + bar_height, image.shape[1], 3), background, dtype=np.uint8)
    canvas[bar_height:, :, :] = image
    cv2.putText(
        canvas,
        str(label)[:80],
        (6, int(bar_height * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        foreground,
        1,
        cv2.LINE_AA,
    )
    return canvas


def text_panel(
    lines: Iterable[str],
    *,
    width: int,
    height: int,
    title: str = "stats",
) -> np.ndarray:
    panel = np.full((int(height), int(width), 3), 248, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (int(width), 24), (25, 25, 25), thickness=-1)
    cv2.putText(panel, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    y = 45
    for raw_line in lines:
        for line in textwrap.wrap(str(raw_line), width=max(20, int(width // 8))):
            if y > height - 8:
                return panel
            cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (35, 35, 35), 1, cv2.LINE_AA)
            y += 18
    return panel


def stack_grid(rows: list[list[np.ndarray]], *, pad: int = 8, background: RGBColor = (238, 238, 238)) -> np.ndarray:
    """Compose uneven RGB panels into a padded grid."""

    if not rows or not any(rows):
        raise ValueError("rows must contain at least one panel")

    normalized_rows: list[np.ndarray] = []
    for row in rows:
        if not row:
            continue
        row_height = max(panel.shape[0] for panel in row)
        row_width = sum(panel.shape[1] for panel in row) + pad * (len(row) - 1)
        canvas = np.full((row_height, row_width, 3), background, dtype=np.uint8)
        x = 0
        for panel in row:
            y = (row_height - panel.shape[0]) // 2
            canvas[y : y + panel.shape[0], x : x + panel.shape[1]] = panel
            x += panel.shape[1] + pad
        normalized_rows.append(canvas)

    width = max(row.shape[1] for row in normalized_rows)
    height = sum(row.shape[0] for row in normalized_rows) + pad * (len(normalized_rows) - 1)
    grid = np.full((height, width, 3), background, dtype=np.uint8)
    y = 0
    for row in normalized_rows:
        grid[y : y + row.shape[0], : row.shape[1]] = row
        y += row.shape[0] + pad
    return grid


def _flow_magnitude(flow: Any) -> np.ndarray:
    flow_hwc = _flow_to_hwc(flow)
    return np.sqrt(np.sum(flow_hwc * flow_hwc, axis=2)).astype(np.float32)


def summarize_sample(sample: dict[str, Any], *, min_motion_px: float = 0.05) -> dict[str, float | str]:
    """Return scalar diagnostics useful for sorting/debugging visualizations."""

    summary: dict[str, float | str] = {
        "scene": str(sample.get("scene", "")),
        "alpha": _to_float(sample.get("alpha", 0.0)),
    }
    if "video_path" in sample:
        summary["video_path"] = str(sample["video_path"])

    if {"motion_flow", "blur_flow", "valid_mask"}.issubset(sample):
        valid = _mask_to_hw(sample["valid_mask"], _flow_to_hwc(sample["motion_flow"]).shape[:2])
        motion_mag = _flow_magnitude(sample["motion_flow"])
        blur_mag = _flow_magnitude(sample["blur_flow"])
        ratio, ratio_mask = local_alpha_ratio(
            sample["motion_flow"],
            sample["blur_flow"],
            valid_mask=sample["valid_mask"],
            min_motion_px=min_motion_px,
        )
        valid_values = valid & np.isfinite(motion_mag)
        ratio_values = ratio_mask & np.isfinite(ratio)
        summary.update(
            {
                "valid_ratio": float(valid.mean()),
                "motion_mag_mean": float(motion_mag[valid_values].mean()) if np.any(valid_values) else 0.0,
                "motion_mag_p95": float(np.percentile(motion_mag[valid_values], 95.0)) if np.any(valid_values) else 0.0,
                "blur_mag_mean": float(blur_mag[valid_values].mean()) if np.any(valid_values) else 0.0,
                "blur_mag_p95": float(np.percentile(blur_mag[valid_values], 95.0)) if np.any(valid_values) else 0.0,
                "local_alpha_mean": float(ratio[ratio_values].mean()) if np.any(ratio_values) else 0.0,
                "local_alpha_p95": float(np.percentile(ratio[ratio_values], 95.0)) if np.any(ratio_values) else 0.0,
                "local_alpha_valid_ratio": float(ratio_values.mean()),
            }
        )
    return summary


def make_sample_visualization(
    sample: dict[str, Any],
    *,
    index: int | None = None,
    panel_width: int = 192,
    panel_height: int = 96,
    max_flow: float | None = None,
    max_blur: float | None = None,
    min_motion_px: float = 0.05,
) -> np.ndarray:
    """Create a contact-sheet style visualization for one dataset sample."""

    frames = _as_numpy(sample["frames"])
    if frames.ndim != 4:
        raise ValueError(f"Expected frames with shape T,C,H,W, got {frames.shape}")
    temporal_len = frames.shape[0]
    center_index = temporal_len // 2

    frame_panels = []
    for t in range(temporal_len):
        offset = t - center_index
        label = "t0" if offset == 0 else f"t{offset:+d}"
        rgb = image_to_rgb(frames[t])
        rgb = resize_rgb(rgb, width=panel_width, height=panel_height)
        frame_panels.append(with_label(rgb, label))

    alpha = _to_float(sample.get("alpha", 0.0))
    scene = str(sample.get("scene", "unknown"))
    header = text_panel(
        [
            f"sample: {index if index is not None else '-'}",
            f"scene: {scene}",
            f"alpha: {alpha:.4f}",
            f"clip shape: {tuple(frames.shape)}",
        ],
        width=max(panel_width * 2, 360),
        height=panel_height + 24,
        title="sample",
    )

    rows: list[list[np.ndarray]] = [[header], frame_panels]

    center = resize_rgb(image_to_rgb(frames[center_index]), width=panel_width, height=panel_height)
    diagnostic_panels = [with_label(center, "center blurred")]

    if {"motion_flow", "blur_flow", "valid_mask"}.issubset(sample):
        motion = sample["motion_flow"]
        blur = sample["blur_flow"]
        valid = sample["valid_mask"]
        motion_mag = _flow_magnitude(motion)
        blur_mag = _flow_magnitude(blur)
        local_alpha, ratio_mask = local_alpha_ratio(
            motion,
            blur,
            valid_mask=valid,
            min_motion_px=min_motion_px,
        )

        diagnostic_panels.extend(
            [
                with_label(
                    resize_rgb(flow_to_color(motion, mask=valid, max_magnitude=max_flow), width=panel_width, height=panel_height),
                    "motion flow d",
                ),
                with_label(
                    resize_rgb(heatmap(motion_mag, mask=valid), width=panel_width, height=panel_height),
                    "|d|",
                ),
                with_label(
                    resize_rgb(flow_to_color(blur, mask=valid, max_magnitude=max_blur), width=panel_width, height=panel_height),
                    "blur flow b",
                ),
                with_label(
                    resize_rgb(heatmap(blur_mag, mask=valid), width=panel_width, height=panel_height),
                    "|b|",
                ),
                with_label(
                    resize_rgb(heatmap(local_alpha, mask=ratio_mask, vmin=0.0, vmax=1.0), width=panel_width, height=panel_height),
                    "local alpha",
                ),
                with_label(
                    resize_rgb(mask_to_rgb(valid), width=panel_width, height=panel_height),
                    "valid mask",
                ),
            ]
        )

        summary = summarize_sample(sample, min_motion_px=min_motion_px)
        stats = text_panel(
            [
                f"valid: {float(summary['valid_ratio']) * 100:.1f}%",
                f"|d| mean/p95: {summary['motion_mag_mean']:.2f}/{summary['motion_mag_p95']:.2f}",
                f"|b| mean/p95: {summary['blur_mag_mean']:.2f}/{summary['blur_mag_p95']:.2f}",
                f"local alpha mean/p95: {summary['local_alpha_mean']:.3f}/{summary['local_alpha_p95']:.3f}",
                f"usable ratio pixels: {float(summary['local_alpha_valid_ratio']) * 100:.1f}%",
            ],
            width=max(panel_width * 2, 360),
            height=panel_height + 24,
            title="diagnostics",
        )
        rows.append(diagnostic_panels)
        rows.append([stats])
    else:
        video_path = str(sample.get("video_path", ""))
        stats = text_panel(
            [
                "real-video sample",
                f"video: {Path(video_path).name if video_path else '-'}",
                "flow/blur-flow targets are unavailable",
                "use this view for frame order, ROI, alpha metadata",
            ],
            width=max(panel_width * 2, 360),
            height=panel_height + 24,
            title="diagnostics",
        )
        rows.append(diagnostic_panels)
        rows.append([stats])

    return stack_grid(rows)


def save_rgb_image(image: np.ndarray, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output), bgr):
        raise OSError(f"Could not write image: {output}")


def write_summary_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "index",
        "image_path",
        "scene",
        "video_path",
        "alpha",
        "valid_ratio",
        "motion_mag_mean",
        "motion_mag_p95",
        "blur_mag_mean",
        "blur_mag_p95",
        "local_alpha_mean",
        "local_alpha_p95",
        "local_alpha_valid_ratio",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)
