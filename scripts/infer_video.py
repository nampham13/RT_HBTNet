from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from models.rt_hbtnet import build_model_from_config
from utils.filters import SpeedStabilizer
from utils.preprocessing import preprocess_roi
from utils.roi import detect_motion_rois, extract_rois, is_auto_motion_mode
from utils.speed_calibration import SpeedCalibrator, robust_roi_fusion
from utils.visualization import draw_rois, make_dashboard_frame


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against CWD first, then the project root."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def parse_roi_arg(value: str | None) -> list[int] | None:
    """Parse ``--roi "x,y,w,h"``."""

    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be formatted as 'x,y,w,h'")
    return [int(float(part)) for part in parts]


def apply_cli_roi(config: dict[str, Any], roi: list[int] | None) -> None:
    """Override config ROI with a CLI fixed ROI if provided."""

    if roi is None:
        return
    roi_cfg = config.setdefault("roi", {})
    roi_cfg["mode"] = "fixed"
    roi_cfg["rois"] = [roi]


def sequence_length_from_config(config: dict[str, Any]) -> int:
    """Get temporal buffer length from config."""

    data_cfg = config.get("data", {})
    return int(data_cfg.get("sequence_length", data_cfg.get("sequence_len", 64)))


def choose_device(config: dict[str, Any]) -> torch.device:
    """Choose CUDA automatically when available."""

    requested = str(config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(config: dict[str, Any], weights: str | Path, device: torch.device) -> torch.nn.Module:
    """Build model and load checkpoint weights."""

    weights_path = resolve_project_path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")

    model = build_model_from_config(config).to(device).eval()
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    """Open a video path or webcam source."""

    source: str | int
    if args.video:
        source = str(resolve_project_path(args.video))
    else:
        source = int(args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video source: {source}")
    return cap


def make_writer(path: str | None, cap: cv2.VideoCapture, target_fps: float) -> cv2.VideoWriter | None:
    """Create output video writer if requested."""

    if not path:
        return None
    output = resolve_project_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output), fourcc, float(fps), (width, height))


def fixed_roi_boxes(config: dict[str, Any]) -> list[list[int]]:
    """Return active fixed-like ROI boxes, or an empty list for full-frame mode."""

    roi_cfg = config.get("roi", {})
    mode = str(roi_cfg.get("mode", "fixed")).lower()
    if mode != "fixed" and not is_auto_motion_mode(config):
        return []
    return [[int(v) for v in roi] for roi in roi_cfg.get("rois", [])]


def warmup_auto_motion_rois(cap: cv2.VideoCapture, config: dict[str, Any], rewind: bool) -> None:
    """Detect and lock auto-motion ROIs before the main inference loop."""

    if not is_auto_motion_mode(config):
        return

    roi_cfg = config.setdefault("roi", {})
    if roi_cfg.get("rois"):
        return

    auto_cfg = roi_cfg.get("auto_motion", {})
    warmup_frames = max(2, int(auto_cfg.get("warmup_frames", 45)))
    start_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    frames: list[np.ndarray] = []

    for _ in range(warmup_frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)

    detected = detect_motion_rois(frames, config)
    if detected:
        roi_cfg["rois"] = [list(roi) for roi in detected]
        print(f"auto ROI detected: {roi_cfg['rois']}")
    else:
        fallback = str(auto_cfg.get("fallback", "full")).lower()
        fallback_rois = auto_cfg.get("fallback_rois", [])
        if fallback == "fixed" and fallback_rois:
            roi_cfg["rois"] = [[int(v) for v in roi] for roi in fallback_rois]
            print(f"auto ROI fallback fixed: {roi_cfg['rois']}")
        else:
            roi_cfg["rois"] = []
            print("auto ROI fallback: full frame")

    if rewind and frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_pos)


def has_roi_error(frame: np.ndarray, rois: list[list[int]]) -> bool:
    """Detect invalid or out-of-frame ROI boxes before clamped cropping."""

    if not rois:
        return False
    height, width = frame.shape[:2]
    for x, y, w, h in rois:
        if w <= 0 or h <= 0:
            return True
        if x < 0 or y < 0 or x + w > width or y + h > height:
            return True
        if x >= width or y >= height:
            return True
    return False


def is_low_light(rois: list[np.ndarray], threshold: float = 35.0) -> bool:
    """Return true when median ROI brightness is below a simple low-light threshold."""

    if not rois:
        return False
    means: list[float] = []
    for roi in rois:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        means.append(float(np.mean(gray)))
    return float(np.median(means)) < float(threshold)


def format_status(base_status: str, warnings: list[str], valid_count: int, total_count: int) -> str:
    """Format status text for the dashboard panel."""

    ordered: list[str] = []
    if base_status and base_status != "OK":
        ordered.append(base_status)
    for warning in warnings:
        if warning not in ordered:
            ordered.append(warning)
    if not ordered:
        ordered.append("OK")
    return f"{' | '.join(ordered)} ({valid_count}/{total_count} ROI)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RT-HBTNet video/webcam inference")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="runs/train/best.pt")
    parser.add_argument("--video", default=None)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--roi", default=None, help='Fixed ROI override formatted as "x,y,w,h"')
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-output", default=None)
    parser.add_argument("--calibration", default="calibration.json")
    parser.add_argument("--known-speed", type=float, default=None)
    parser.add_argument("--calibrate-seconds", type=float, default=10.0)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    apply_cli_roi(config, parse_roi_arg(args.roi))
    sequence_length = sequence_length_from_config(config)
    inference_cfg = config.get("inference", {})
    fusion_cfg = config.get("fusion", {})
    target_fps = float(inference_cfg.get("target_fps", 30))
    display_enabled = bool(inference_cfg.get("show_window", True)) and not args.no_display
    min_confidence = float(fusion_cfg.get("min_confidence", 0.05))
    max_speed_delta = float(config.get("stabilization", {}).get("max_speed_delta", 0.5))

    device = choose_device(config)
    model = load_model(config, args.weights, device)
    stabilizer = SpeedStabilizer(config)
    calibration_path = resolve_project_path(args.calibration)
    calibrator = SpeedCalibrator()
    if args.known_speed is None and calibration_path.exists():
        calibrator = SpeedCalibrator.load(calibration_path)
    cap = open_capture(args)
    warmup_auto_motion_rois(cap, config, rewind=args.video is not None)
    writer = make_writer(args.save_output, cap, target_fps)

    buffers: list[deque[np.ndarray]] = []
    raw_speed: float | None = None
    smooth_speed: float | None = None
    tex_conf: float | None = None
    blur_conf: float | None = None
    w_tex: float | None = None
    w_blur: float | None = None
    valid_count = 0
    status = "BUFFERING"
    roi_preview: np.ndarray | None = None
    calibration_samples: list[float] = []
    calibration_start: float | None = None
    calibration_done = args.known_speed is None
    last_accepted_speed: float | None = None
    fps = 0.0
    last_time = time.perf_counter()
    frames_read = 0

    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if frames_read == 0:
                        raise RuntimeError("No frame read from video source")
                    print("No more frames read; stopping inference")
                    break
                frames_read += 1
                frame_warnings: list[str] = []

                roi_boxes = fixed_roi_boxes(config)
                if has_roi_error(frame, roi_boxes):
                    frame_warnings.append("ROI ERROR")

                try:
                    rois = extract_rois(frame, config)
                except (ValueError, cv2.error) as exc:
                    frame_warnings.append("ROI ERROR")
                    rois = []
                    print(f"ROI extraction failed: {exc}")

                if rois and is_low_light(rois):
                    frame_warnings.append("LOW LIGHT")
                roi_preview = rois[0] if rois else None
                if len(buffers) != len(rois):
                    buffers = [deque(maxlen=sequence_length) for _ in rois]
                    stabilizer.reset()

                for idx, roi in enumerate(rois):
                    buffers[idx].append(preprocess_roi(roi, config))  # C,H,W

                if not rois:
                    status = "ROI ERROR"
                    valid_count = 0
                elif not all(len(buf) == sequence_length for buf in buffers):
                    filled = min((len(buf) for buf in buffers), default=0)
                    status = f"BUFFERING {filled}/{sequence_length}"
                    valid_count = 0
                else:
                    # Batch all ROI sequences together: R,T,C,H,W.
                    batch_np = np.stack([np.stack(list(buf), axis=0) for buf in buffers], axis=0).astype(np.float32)
                    x_seq = torch.from_numpy(batch_np).to(device, non_blocking=True)
                    pred = model(x_seq)

                    speed_values = pred["speed"].detach().cpu().numpy().reshape(-1)
                    conf_final_values = pred["conf_final"].detach().cpu().numpy().reshape(-1)
                    conf_tex_values = pred["conf_tex"].detach().cpu().numpy().reshape(-1)
                    conf_blur_values = pred["conf_blur"].detach().cpu().numpy().reshape(-1)
                    w_tex_values = pred["w_tex"].detach().cpu().numpy().reshape(-1)
                    w_blur_values = pred["w_blur"].detach().cpu().numpy().reshape(-1)
                    roi_predictions = [
                        {
                            "speed": float(speed_values[i]),
                            "conf_final": float(conf_final_values[i]),
                            "conf_tex": float(conf_tex_values[i]),
                            "conf_blur": float(conf_blur_values[i]),
                            "w_tex": float(w_tex_values[i]),
                            "w_blur": float(w_blur_values[i]),
                        }
                        for i in range(len(speed_values))
                    ]
                    roi_fused = robust_roi_fusion(roi_predictions, min_confidence=min_confidence)
                    valid_count = int(roi_fused["valid_count"])
                    status = str(roi_fused["status"])
                    if valid_count == 0:
                        frame_warnings.append("LOW CONFIDENCE")

                    tex_conf = float(np.mean(conf_tex_values))
                    blur_conf = float(np.mean(conf_blur_values))
                    w_tex = float(np.mean(w_tex_values))
                    w_blur = float(np.mean(w_blur_values))
                    if roi_fused["speed_median"] is not None:
                        raw_model_speed = float(roi_fused["speed_median"])
                        raw_speed = calibrator.apply(raw_model_speed)
                        if not calibration_done:
                            if calibration_start is None:
                                calibration_start = time.perf_counter()
                            calibration_samples.append(raw_model_speed)
                            elapsed = time.perf_counter() - calibration_start
                            status = f"CALIBRATING {elapsed:.1f}/{args.calibrate_seconds:.1f}s"
                            if elapsed >= float(args.calibrate_seconds):
                                scale = calibrator.calibrate_from_known_speed(
                                    calibration_samples,
                                    known_speed_mps=float(args.known_speed),
                                )
                                calibrator.save(calibration_path)
                                calibration_done = True
                                stabilizer.reset()
                                last_accepted_speed = None
                                raw_speed = calibrator.apply(raw_model_speed)
                                print(f"saved calibration: {calibration_path} scale={scale:.6f}")

                        if last_accepted_speed is not None:
                            delta = raw_speed - last_accepted_speed
                            if abs(delta) > max_speed_delta:
                                frame_warnings.append("UNSTABLE SPEED")
                                raw_speed = last_accepted_speed + float(np.sign(delta)) * max_speed_delta
                        smooth_speed = stabilizer.update(raw_speed)
                        last_accepted_speed = raw_speed

                now = time.perf_counter()
                inst_fps = 1.0 / max(now - last_time, 1.0e-6)
                last_time = now
                fps = inst_fps if fps <= 0.0 else 0.1 * inst_fps + 0.9 * fps
                if fps < target_fps:
                    frame_warnings.append("LOW FPS")

                output_frame = draw_rois(frame, roi_boxes) if config.get("inference", {}).get("draw_roi", True) else frame.copy()
                output_frame = make_dashboard_frame(
                    output_frame,
                    roi_preview,
                    {
                        "speed_raw": raw_speed,
                        "speed_smooth": smooth_speed,
                        "conf_tex": tex_conf,
                        "conf_blur": blur_conf,
                        "w_tex": w_tex,
                        "w_blur": w_blur,
                        "fps": fps,
                        "status": format_status(status, frame_warnings, valid_count, len(buffers)),
                    },
                )

                display_scale = float(config.get("inference", {}).get("display_scale", 1.0))
                if writer is not None:
                    writer.write(output_frame)
                if display_enabled:
                    shown = output_frame
                    if display_scale != 1.0:
                        shown = cv2.resize(output_frame, None, fx=display_scale, fy=display_scale)
                    cv2.imshow("RT-HBTNet", shown)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if display_enabled:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
