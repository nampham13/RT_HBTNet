from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against CWD first, then project root."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.parent.exists():
        return cwd_path
    return ROOT / path


def make_texture(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
    """Create a conveyor-like grayscale texture larger than the output frame."""

    texture = rng.normal(120.0, 38.0, size=(height, width)).astype(np.float32)

    # Fine belt fibers.
    for y in range(0, height, 7):
        texture[y : y + 1, :] += rng.uniform(8.0, 24.0)

    # Occasional seams and scratches along the belt.
    for x in range(0, width, 80):
        texture[:, x : x + 3] += rng.uniform(20.0, 60.0)
    for _ in range(max(60, height * width // 900)):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, 4))
        value = float(rng.uniform(35.0, 220.0))
        cv2.circle(texture, (x, y), radius, value, -1)

    texture = cv2.GaussianBlur(texture, (3, 3), 0)
    return np.clip(texture, 0, 255).astype(np.uint8)


def apply_low_light(frame: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Darken and add sensor-like noise."""

    img = frame.astype(np.float32) * float(rng.uniform(0.35, 0.65)) + float(rng.uniform(2.0, 12.0))
    img += rng.normal(0.0, rng.uniform(3.0, 8.0), size=img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_motion_blur(frame: np.ndarray, speed: float) -> np.ndarray:
    """Apply simple horizontal motion blur."""

    k = int(round(max(3.0, min(31.0, speed * 5.0))))
    if k % 2 == 0:
        k += 1
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(frame, -1, kernel)


def write_labels(output_path: Path, speed: float, frames: int) -> None:
    """Write labels.csv compatible with VideoSpeedDataset."""

    labels_path = output_path.parent / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "start_frame", "end_frame", "speed_mps"])
        writer.writerow([output_path.name, 0, max(0, frames - 1), float(speed)])
    print(f"wrote labels: {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic conveyor belt demo video")
    parser.add_argument("--output", default="data/synthetic_conveyor.mp4")
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--low-light", action="store_true")
    parser.add_argument("--blur", action="store_true")
    args = parser.parse_args()

    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    texture_width = int(args.width * 3)
    texture = make_texture(rng, int(args.height), texture_width)
    pixels_per_meter = 64.0
    pixels_per_frame = float(args.speed) * pixels_per_meter / max(float(args.fps), 1.0)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(args.fps), (int(args.width), int(args.height)))
    if not writer.isOpened():
        raise RuntimeError(f"could not create output video: {output_path}")

    try:
        for idx in range(int(args.frames)):
            shift = int(round(idx * pixels_per_frame)) % args.width
            start = shift
            end = start + args.width
            belt = texture[:, start:end]
            if belt.shape[1] < args.width:
                belt = np.concatenate([belt, texture[:, : args.width - belt.shape[1]]], axis=1)

            frame = belt.copy()
            if args.blur:
                frame = apply_motion_blur(frame, float(args.speed))
            if args.low_light:
                frame = apply_low_light(frame, rng)

            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()

    write_labels(output_path, float(args.speed), int(args.frames))
    print(f"wrote video: {output_path}")


if __name__ == "__main__":
    main()
