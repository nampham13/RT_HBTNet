from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from rt_hbtnet.models.rt_hbtnet import build_model_from_config


class SpeedOnlyWrapper(torch.nn.Module):
    """Export only the fused speed output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.model(x_seq)["speed"]


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


def dummy_shape(config: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Return dummy input shape ``1,T,C,H,W``."""

    data_cfg = config.get("data", {})
    image_size = data_cfg.get("image_size", {"height": 64, "width": 128})
    if isinstance(image_size, dict):
        height = int(image_size.get("height", 64))
        width = int(image_size.get("width", 128))
    else:
        height = int(image_size[0])
        width = int(image_size[1])
    sequence_length = int(data_cfg.get("sequence_length", data_cfg.get("sequence_len", 64)))
    channels = int(config.get("model", {}).get("in_channels", 1))
    return 1, sequence_length, channels, height, width


def load_weights(model: torch.nn.Module, weights: str | Path) -> None:
    """Load checkpoint weights."""

    weights_path = resolve_project_path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def verify_onnx(output_path: Path) -> None:
    """Verify ONNX graph if the optional onnx package is installed."""

    if importlib.util.find_spec("onnx") is None:
        print("onnx package not installed; skipping verification")
        return

    import onnx  # type: ignore[import-not-found]

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("ONNX verification: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RT-HBTNet to ONNX")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="runs/train/best.pt")
    parser.add_argument("--output", default="rt_hbtnet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    model = build_model_from_config(config).eval()
    load_weights(model, args.weights)

    dummy = torch.rand(*dummy_shape(config), dtype=torch.float32)
    output_path = resolve_project_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        SpeedOnlyWrapper(model),
        dummy,
        str(output_path),
        input_names=["x_seq"],
        output_names=["speed"],
        dynamic_axes={
            "x_seq": {0: "batch", 1: "sequence"},
            "speed": {0: "batch"},
        },
        opset_version=int(args.opset),
    )

    print(f"Exported ONNX: {output_path}")
    verify_onnx(output_path)


if __name__ == "__main__":
    main()
