from __future__ import annotations

import argparse
import struct
from pathlib import Path

import cv2
import numpy as np


def write_flo(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = flow.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("f", 202021.25))
        handle.write(struct.pack("i", width))
        handle.write(struct.pack("i", height))
        handle.write(flow.astype(np.float32).tobytes())


def make_texture(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    grid = np.zeros_like(noise)
    grid[::8, :] = 255
    grid[:, ::8] = 255
    texture = cv2.addWeighted(noise, 0.7, grid, 0.3, 0.0)
    return cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic toy frame+flow data")
    parser.add_argument("--output", default="data/toy_exposure")
    parser.add_argument("--scenes", type=int, default=4)
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    root = Path(args.output)
    for scene_index in range(int(args.scenes)):
        scene = f"toy_{scene_index:02d}"
        dx = float(1 + scene_index)
        dy = float((scene_index % 3) - 1)
        base = make_texture(int(args.height), int(args.width), seed=100 + scene_index)
        flow = np.zeros((int(args.height), int(args.width), 2), dtype=np.float32)
        flow[..., 0] = dx
        flow[..., 1] = dy
        for frame_index in range(1, int(args.frames) + 1):
            transform = np.float32([[1.0, 0.0, dx * frame_index], [0.0, 1.0, dy * frame_index]])
            frame = cv2.warpAffine(
                base,
                transform,
                (int(args.width), int(args.height)),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            stem = f"frame_{frame_index:04d}"
            frame_path = root / "training" / "final" / scene / f"{stem}.png"
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frame_path), frame)
            write_flo(root / "training" / "flow" / scene / f"{stem}.flo", flow)
    print(f"wrote toy exposure data to {root.resolve()}")


if __name__ == "__main__":
    main()
