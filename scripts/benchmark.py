from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from rt_hbtnet.models.rt_hbtnet import build_model_from_config, count_parameters  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against CWD first, then project root."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def choose_device(config: dict[str, Any]) -> torch.device:
    """Choose CUDA automatically when available."""

    requested = str(config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def input_shape(config: dict[str, Any], batch_size: int, sequence_length: int) -> tuple[int, int, int, int, int]:
    """Return benchmark input shape as ``B,T,C,H,W``."""

    image_size = config.get("data", {}).get("image_size", {"height": 64, "width": 128})
    if isinstance(image_size, dict):
        height = int(image_size.get("height", 64))
        width = int(image_size.get("width", 128))
    else:
        height = int(image_size[0])
        width = int(image_size[1])
    channels = int(config.get("model", {}).get("in_channels", 1))
    return int(batch_size), int(sequence_length), channels, height, width


def load_weights(model: torch.nn.Module, weights: str | None, device: torch.device) -> None:
    """Load optional checkpoint weights."""

    if not weights:
        return
    weights_path = resolve_project_path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RT-HBTNet latency")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    device = choose_device(config)
    model = build_model_from_config(config).to(device).eval()
    load_weights(model, args.weights, device)

    shape = input_shape(config, args.batch_size, args.sequence_length)
    dummy = torch.rand(*shape, device=device)

    with torch.no_grad():
        for _ in range(int(args.warmup)):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times_ms: list[float] = []
        for _ in range(int(args.iters)):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - start) * 1000.0)

    avg_ms = statistics.mean(times_ms)
    fps = 1000.0 * int(args.batch_size) / avg_ms

    print("RT-HBTNet Benchmark")
    print("-------------------")
    print(f"Device: {device}")
    print(f"Input shape: {shape}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Warmup iters: {int(args.warmup)}")
    print(f"Test iters: {int(args.iters)}")
    print(f"Average latency: {avg_ms:.3f} ms")
    print(f"FPS: {fps:.2f}")


if __name__ == "__main__":
    main()
