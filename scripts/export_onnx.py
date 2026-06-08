from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from models.rt_hbtnet import build_model_from_config
from utils.onnx_export import export_model_to_onnx


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


def load_weights(model: torch.nn.Module, weights: str | Path) -> None:
    """Load checkpoint weights."""

    weights_path = resolve_project_path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"weights file not found: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RT-HBTNet to ONNX")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", default="runs/train/best.pt")
    parser.add_argument("--output", default="rt_hbtnet.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    model = build_model_from_config(config).eval()
    load_weights(model, args.weights)

    output_path = resolve_project_path(args.output)
    exported_path = export_model_to_onnx(
        model,
        config,
        output_path,
        opset=int(args.opset),
        verify=not bool(args.no_verify),
    )
    print(f"Exported ONNX: {exported_path}")


if __name__ == "__main__":
    main()
