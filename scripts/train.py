from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from datasets import DatasetFactory
from models.rt_hbtnet import build_model_from_config, count_parameters
from utils.metrics import alpha_error_report


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else ROOT / path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def split_dataset_by_group(
    dataset: Dataset[Any],
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[Subset[Any], Subset[Any]]:
    """Create a leakage-safe split where one scene belongs to one partition."""

    group_ids = getattr(dataset, "group_ids", None)
    if not group_ids or len(group_ids) != len(dataset):
        raise ValueError("Dataset must expose one group_ids entry per sample")
    groups = sorted(set(str(group) for group in group_ids))
    if len(groups) < 2:
        raise ValueError("At least two scenes are required for a scene-disjoint split")

    rng = random.Random(int(seed))
    rng.shuffle(groups)
    val_group_count = max(1, int(round(len(groups) * float(val_fraction))))
    val_groups = set(groups[:val_group_count])
    train_indices = [idx for idx, group in enumerate(group_ids) if str(group) not in val_groups]
    val_indices = [idx for idx, group in enumerate(group_ids) if str(group) in val_groups]
    if not train_indices or not val_indices:
        raise ValueError("Scene-disjoint split produced an empty partition")
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def resize_supervision(
    target: torch.Tensor,
    size: tuple[int, int],
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    if target.shape[-2:] == size:
        return target
    if mode == "nearest":
        return F.interpolate(target, size=size, mode=mode)
    return F.interpolate(target, size=size, mode=mode, align_corners=False)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return (values * mask).sum() / (mask.sum() + float(eps))


def sign_invariant_vector_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive = (pred - target).square().sum(dim=1, keepdim=True)
    negative = (pred + target).square().sum(dim=1, keepdim=True)
    return torch.minimum(positive, negative)


def compute_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor | list[str]],
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task supervision plus the blur-temporal physical constraint."""

    alpha_gt = batch["alpha"]
    motion_gt = batch["motion_flow"]
    blur_gt = batch["blur_flow"]
    valid = batch["valid_mask"]
    if not all(isinstance(value, torch.Tensor) for value in (alpha_gt, motion_gt, blur_gt, valid)):
        raise TypeError("alpha, motion_flow, blur_flow, and valid_mask must be tensors")

    output_size = pred["motion_flow"].shape[-2:]
    motion_gt = resize_supervision(motion_gt, output_size)
    blur_gt = resize_supervision(blur_gt, output_size)
    valid = resize_supervision(valid, output_size, mode="nearest")

    motion_sq = (pred["motion_flow"] - motion_gt).square().sum(dim=1, keepdim=True)
    blur_sq = sign_invariant_vector_error(pred["blur_flow"], blur_gt)
    motion_nll = 0.5 * (torch.exp(-pred["motion_logvar"]) * motion_sq + pred["motion_logvar"])
    blur_nll = 0.5 * (torch.exp(-pred["blur_logvar"]) * blur_sq + pred["blur_logvar"])
    motion_loss = masked_mean(motion_nll, valid)
    blur_loss = masked_mean(blur_nll, valid)

    alpha_gt = alpha_gt.view(-1, 1)
    alpha_loss = F.smooth_l1_loss(pred["alpha"], alpha_gt, beta=0.02)
    gt_motion_energy = motion_gt.square().sum(dim=1, keepdim=True)
    ratio_mask = valid * (gt_motion_energy > 0.05**2).to(valid.dtype)
    local_alpha_target = alpha_gt[:, :, None, None].expand_as(pred["alpha_map"])
    local_alpha_error = F.smooth_l1_loss(
        torch.clamp(pred["alpha_map"], 0.0, 1.0),
        local_alpha_target,
        beta=0.02,
        reduction="none",
    )
    local_alpha_loss = masked_mean(local_alpha_error, ratio_mask)

    alpha_field = alpha_gt[:, :, None, None]
    physics_sq = torch.minimum(
        (pred["blur_flow"] - alpha_field * pred["motion_flow"]).square().sum(dim=1, keepdim=True),
        (pred["blur_flow"] + alpha_field * pred["motion_flow"]).square().sum(dim=1, keepdim=True),
    )
    physics_loss = masked_mean(torch.sqrt(physics_sq + 1.0e-6), valid)

    direction_dot = (pred["blur_flow"] * pred["motion_flow"]).sum(dim=1, keepdim=True).abs()
    direction_den = (
        torch.sqrt(pred["blur_flow"].square().sum(dim=1, keepdim=True) + 1.0e-6)
        * torch.sqrt(pred["motion_flow"].square().sum(dim=1, keepdim=True) + 1.0e-6)
    )
    direction_loss = masked_mean(1.0 - direction_dot / direction_den.clamp_min(1.0e-6), valid)

    total = (
        float(weights.get("alpha_weight", 1.0)) * alpha_loss
        + float(weights.get("local_alpha_weight", 0.5)) * local_alpha_loss
        + float(weights.get("motion_weight", 0.25)) * motion_loss
        + float(weights.get("blur_weight", 0.25)) * blur_loss
        + float(weights.get("physics_weight", 0.25)) * physics_loss
        + float(weights.get("direction_weight", 0.05)) * direction_loss
    )
    return total, {
        "alpha_loss": float(alpha_loss.detach().cpu()),
        "local_alpha_loss": float(local_alpha_loss.detach().cpu()),
        "motion_loss": float(motion_loss.detach().cpu()),
        "blur_loss": float(blur_loss.detach().cpu()),
        "physics_loss": float(physics_loss.detach().cpu()),
        "direction_loss": float(direction_loss.detach().cpu()),
    }


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True).float() if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict[str, float],
    epoch: int,
) -> dict[str, float]:
    model.train()
    totals: dict[str, list[float]] = {
        "train_loss": [],
        "alpha_loss": [],
        "local_alpha_loss": [],
        "motion_loss": [],
        "blur_loss": [],
        "physics_loss": [],
        "direction_loss": [],
    }
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for raw_batch in progress:
        batch = move_batch_to_device(raw_batch, device)
        pred = model(batch["frames"])
        loss, parts = compute_loss(pred, batch, loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        totals["train_loss"].append(float(loss.detach().cpu()))
        for key, value in parts.items():
            totals[key].append(value)
        progress.set_postfix(loss=f"{np.mean(totals['train_loss']):.4f}")
    return {key: float(np.mean(values)) for key, values in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    alpha_predictions: list[torch.Tensor] = []
    alpha_targets: list[torch.Tensor] = []
    motion_errors: list[float] = []
    blur_errors: list[float] = []

    for raw_batch in tqdm(loader, desc="val", leave=False):
        batch = move_batch_to_device(raw_batch, device)
        pred = model(batch["frames"])
        alpha_predictions.append(pred["alpha"].detach().cpu().view(-1))
        alpha_targets.append(batch["alpha"].detach().cpu().view(-1))

        size = pred["motion_flow"].shape[-2:]
        motion_gt = resize_supervision(batch["motion_flow"], size)
        blur_gt = resize_supervision(batch["blur_flow"], size)
        valid = resize_supervision(batch["valid_mask"], size, mode="nearest")
        motion_epe = torch.sqrt(
            (pred["motion_flow"] - motion_gt).square().sum(dim=1, keepdim=True) + 1.0e-6
        )
        blur_epe = torch.sqrt(sign_invariant_vector_error(pred["blur_flow"], blur_gt) + 1.0e-6)
        motion_errors.append(float(masked_mean(motion_epe, valid).cpu()))
        blur_errors.append(float(masked_mean(blur_epe, valid).cpu()))

    report = alpha_error_report(torch.cat(alpha_predictions), torch.cat(alpha_targets))
    return {
        **{key: float(value) for key, value in report.items() if key != "num_samples"},
        "motion_epe": float(np.mean(motion_errors)),
        "blur_epe": float(np.mean(blur_errors)),
    }


def write_history(history: list[dict[str, float]], output: Path) -> None:
    if not history:
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BT-ShutterNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default="exposure_flow", choices=["exposure_flow"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--prediction-mode",
        choices=["physics", "direct", "blur_only", "temporal_only"],
        default=None,
    )
    parser.add_argument(
        "--alpha-only",
        action="store_true",
        help="Disable dense cue losses for scalar-regression baselines",
    )
    parser.add_argument("--save-dir", default="runs/exposure")
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    train_cfg = config.setdefault("training", {})
    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    if args.prediction_mode is not None:
        config.setdefault("model", {})["prediction_mode"] = args.prediction_mode
    if args.alpha_only:
        loss_cfg = train_cfg.setdefault("loss", {})
        loss_cfg["motion_weight"] = 0.0
        loss_cfg["blur_weight"] = 0.0
        loss_cfg["physics_weight"] = 0.0
        loss_cfg["direction_weight"] = 0.0
        loss_cfg["local_alpha_weight"] = 0.0

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = choose_device(config)
    data_cfg = config.get("data", {}).get("datasets", {}).get("exposure_flow", {})
    root = resolve_project_path(args.data_root or data_cfg.get("root", "data/raw/sintel"))
    dataset = DatasetFactory.create(
        config=config,
        dataset_type=args.dataset,
        root=root,
        split=args.split or data_cfg.get("split", "training"),
    )
    train_set, val_set = split_dataset_by_group(
        dataset,
        seed=seed,
        val_fraction=float(train_cfg.get("val_fraction", 0.2)),
    )

    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_options)
    val_loader = DataLoader(val_set, shuffle=False, **loader_options)

    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2.0e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(train_cfg.get("epochs", 30))),
    )

    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    print(f"device: {device}")
    print(f"train={len(train_set)} val={len(val_set)} (scene-disjoint)")
    print(f"parameters: {count_parameters(model):,}")

    best_mae = float("inf")
    history: list[dict[str, float]] = []
    epochs = int(train_cfg.get("epochs", 30))
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_metrics = run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train_cfg.get("loss", {}),
            epoch,
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time_s": float(time.perf_counter() - start),
        }
        history.append(row)
        write_history(history, save_dir / "history.csv")
        print(
            f"epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"alpha_mae={row['val_alpha_mae']:.4f} "
            f"motion_epe={row['val_motion_epe']:.3f} "
            f"blur_epe={row['val_blur_epe']:.3f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if val_metrics["alpha_mae"] < best_mae:
            best_mae = val_metrics["alpha_mae"]
            torch.save(checkpoint, save_dir / "best.pt")


if __name__ == "__main__":
    main()
