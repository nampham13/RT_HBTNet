from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from typing import Any

import torch


class SpeedOnlyWrapper(torch.nn.Module):
    """Export only the fused speed output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.model(x_seq)["speed"]


def dummy_shape(config: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Return dummy input shape ``1,T,C,H,W`` for ONNX export."""

    data_cfg = config.get("data", {})
    roi_cfg = config.get("roi", {})
    image_size = data_cfg.get("image_size", {"height": 64, "width": 128})

    if "resize_height" in roi_cfg and "resize_width" in roi_cfg:
        height = int(roi_cfg["resize_height"])
        width = int(roi_cfg["resize_width"])
    elif isinstance(image_size, dict):
        height = int(image_size.get("height", 64))
        width = int(image_size.get("width", 128))
    else:
        height = int(image_size[0])
        width = int(image_size[1])

    sequence_length = int(data_cfg.get("sequence_length", data_cfg.get("sequence_len", 64)))
    channels = int(config.get("model", {}).get("in_channels", 1))
    return 1, sequence_length, channels, height, width


def verify_onnx(output_path: str | Path) -> bool:
    """Verify ONNX graph if the optional onnx package is installed."""

    if importlib.util.find_spec("onnx") is None:
        print("onnx package not installed; skipping verification")
        return False

    import onnx  # type: ignore[import-not-found]

    model = onnx.load(str(output_path))
    onnx.checker.check_model(model)
    print("ONNX verification: OK")
    return True


def export_model_to_onnx(
    model: torch.nn.Module,
    config: dict[str, Any],
    output_path: str | Path,
    opset: int = 17,
    verify: bool = False,
) -> Path:
    """Export a trained RT-HBTNet model to an ONNX speed-output graph."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    was_training = model.training
    model.eval()
    dummy = torch.rand(*dummy_shape(config), dtype=torch.float32, device=device)
    dynamic_axes: dict[str, dict[int, str]] = {
        "x_seq": {0: "batch"},
        "speed": {0: "batch"},
    }

    export_kwargs: dict[str, Any] = {
        "input_names": ["x_seq"],
        "output_names": ["speed"],
        "dynamic_axes": dynamic_axes,
        "opset_version": int(opset),
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    with torch.no_grad():
        torch.onnx.export(
            SpeedOnlyWrapper(model),
            dummy,
            str(output),
            **export_kwargs,
        )

    if was_training:
        model.train()

    if verify:
        verify_onnx(output)
    return output
