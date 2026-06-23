from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from datasets import DatasetFactory
from models.rt_hbtnet import build_model_from_config
from utils.metrics import alpha_error_report


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def bin_report(pred: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float | int]]:
    bins = {
        "alpha_lt_0.1": target < 0.1,
        "alpha_0.1_to_0.3": (target >= 0.1) & (target <= 0.3),
        "alpha_gt_0.3": target > 0.3,
    }
    report: dict[str, dict[str, float | int]] = {}
    for name, mask in bins.items():
        if not np.any(mask):
            report[name] = {"count": 0}
            continue
        errors = np.abs(pred[mask] - target[mask])
        report[name] = {
            "count": int(mask.sum()),
            "mae": float(errors.mean()),
            "median_ae": float(np.median(errors)),
        }
    return report


def risk_coverage(
    pred: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> list[dict[str, float]]:
    order = np.argsort(-confidence)
    errors = np.abs(pred[order] - target[order])
    points = []
    for coverage in (0.25, 0.5, 0.75, 1.0):
        count = max(1, int(round(len(errors) * coverage)))
        points.append({"coverage": coverage, "mae": float(errors[:count].mean())})
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exposure-fraction estimation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", choices=["exposure_flow", "exposure_video"], required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--constant-alpha",
        type=float,
        default=0.5,
        help="Constant baseline fixed before evaluating this split (use the training-set median)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    requested = str(config.get("project", {}).get("device", "auto")).lower()
    device = torch.device(
        "cuda" if requested == "auto" and torch.cuda.is_available()
        else "cpu" if requested == "auto"
        else requested
    )

    dataset = DatasetFactory.create(
        config=config,
        dataset_type=args.dataset,
        root=None if args.data_root is None else resolve_project_path(args.data_root),
        video_root=None if args.data_root is None else resolve_project_path(args.data_root),
        manifest_path=None if args.manifest is None else resolve_project_path(args.manifest),
        split=args.split,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = build_model_from_config(config).to(device).eval()
    checkpoint = torch.load(resolve_project_path(args.weights), map_location=device)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    confidences: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluate"):
            frames = batch["frames"].to(device).float()
            out = model(frames)
            predictions.append(out["alpha"].cpu().view(-1))
            targets.append(batch["alpha"].float().view(-1))
            confidences.append(out["confidence"].cpu().view(-1))

    pred_np = torch.cat(predictions).numpy()
    target_np = torch.cat(targets).numpy()
    confidence_np = torch.cat(confidences).numpy()
    constant = float(args.constant_alpha)
    report = {
        "model": alpha_error_report(pred_np, target_np),
        "constant_baseline": {
            "prediction": constant,
            **alpha_error_report(np.full_like(target_np, constant), target_np),
        },
        "per_alpha_bin": bin_report(pred_np, target_np),
        "risk_coverage": risk_coverage(pred_np, target_np, confidence_np),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = resolve_project_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
