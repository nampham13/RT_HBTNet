from __future__ import annotations

import argparse
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
from utils.filters import EMAFilter
from utils.preprocessing import preprocess_roi


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate video exposure fraction")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="runs/exposure/best.pt")
    parser.add_argument("--video", required=True)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-output", default=None)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    device = choose_device(config)
    model = build_model_from_config(config).to(device).eval()
    checkpoint = torch.load(resolve_project_path(args.weights), map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    cap = cv2.VideoCapture(str(resolve_project_path(args.video)))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sequence_length = int(config.get("data", {}).get("sequence_length", 5))
    buffer: deque[np.ndarray] = deque(maxlen=sequence_length)
    smoother = EMAFilter(alpha=float(config.get("inference", {}).get("ema_alpha", 0.2)))

    writer = None
    if args.save_output:
        output_path = resolve_project_path(args.save_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    alpha_value: float | None = None
    confidence_value: float | None = None
    try:
        with torch.no_grad():
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                buffer.append(preprocess_roi(frame, config))
                if len(buffer) == sequence_length:
                    clip = torch.from_numpy(
                        np.stack(list(buffer), axis=0)[None].astype(np.float32)
                    ).to(device)
                    pred = model(clip)
                    alpha_value = smoother.update(float(pred["alpha"].item()))
                    confidence_value = float(pred["confidence"].item())

                output = frame.copy()
                if alpha_value is None:
                    status = f"Buffering {len(buffer)}/{sequence_length}"
                else:
                    shutter_degrees = 360.0 * alpha_value
                    exposure_ms = 1000.0 * alpha_value / fps
                    status = (
                        f"alpha={alpha_value:.3f}  shutter={shutter_degrees:.1f} deg  "
                        f"exposure={exposure_ms:.2f} ms  conf={confidence_value:.2f}"
                    )
                cv2.putText(
                    output,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if writer is not None:
                    writer.write(output)
                if not args.no_display:
                    cv2.imshow("BT-ShutterNet", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if alpha_value is not None:
        print(f"exposure_fraction={alpha_value:.6f}")
        print(f"shutter_angle_degrees={360.0 * alpha_value:.3f}")
        print(f"exposure_time_ms={1000.0 * alpha_value / fps:.3f}")
        print(f"confidence={confidence_value:.6f}")


if __name__ == "__main__":
    main()
