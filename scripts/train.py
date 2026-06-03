from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from rt_hbtnet.datasets import SyntheticSpeedDataset, VideoSpeedDataset  # noqa: E402
from rt_hbtnet.models.rt_hbtnet import build_model_from_config, count_parameters  # noqa: E402
from rt_hbtnet.utils.metrics import mae, mape, rmse  # noqa: E402


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(config: dict[str, Any]) -> torch.device:
    """Choose CUDA automatically when requested and available."""

    requested = str(config.get("project", {}).get("device", "auto")).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve relative paths against the project root when needed."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT / path


def build_dataset(args: argparse.Namespace, config: dict[str, Any]) -> Dataset:
    """Build either synthetic or labeled-video dataset."""

    if args.synthetic:
        return SyntheticSpeedDataset(config=config)

    labels_path = resolve_project_path(args.labels)
    video_root = resolve_project_path(args.video_root)
    return VideoSpeedDataset(
        labels_csv=labels_path,
        video_root=video_root,
        config=config,
    )


def split_dataset(dataset: Dataset, seed: int) -> tuple[Dataset, Dataset]:
    """Split one dataset into train/validation subsets."""

    total = len(dataset)
    if total < 2:
        raise ValueError("Need at least 2 samples to create a train/val split")
    val_count = max(1, int(round(total * 0.2)))
    train_count = total - val_count
    if train_count < 1:
        train_count, val_count = 1, total - 1
    generator = torch.Generator().manual_seed(int(seed))
    return random_split(dataset, [train_count, val_count], generator=generator)


def compute_loss(
    pred: dict[str, torch.Tensor],
    gt: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute weighted L1 branch and fused-speed losses."""

    gt = gt.view(-1, 1)
    l1 = nn.functional.l1_loss
    main_loss = l1(pred["speed"], gt)
    tex_loss = l1(pred["speed_tex"], gt)
    blur_loss = l1(pred["speed_blur"], gt)
    total = (
        float(weights.get("main_weight", 1.0)) * main_loss
        + float(weights.get("tex_weight", 0.5)) * tex_loss
        + float(weights.get("blur_weight", 0.5)) * blur_loss
    )

    conf_reg_weight = float(weights.get("conf_reg_weight", 0.0))
    conf_reg = torch.zeros((), device=gt.device)
    if conf_reg_weight > 0.0:
        # Confidence target is high when the corresponding branch error is low.
        tex_conf_target = torch.exp(-torch.abs(pred["speed_tex"] - gt)).detach()
        blur_conf_target = torch.exp(-torch.abs(pred["speed_blur"] - gt)).detach()
        conf_reg = l1(pred["conf_tex"], tex_conf_target) + l1(pred["conf_blur"], blur_conf_target)
        total = total + conf_reg_weight * conf_reg

    return total, {
        "main_loss": float(main_loss.detach().cpu()),
        "tex_loss": float(tex_loss.detach().cpu()),
        "blur_loss": float(blur_loss.detach().cpu()),
        "conf_reg": float(conf_reg.detach().cpu()),
    }


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weights: dict[str, float],
    epoch: int,
) -> float:
    """Run one training epoch and return mean loss."""

    model.train()
    losses: list[float] = []
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    for x_seq, y_speed in progress:
        x_seq = x_seq.to(device, non_blocking=True).float()  # B,T,C,H,W
        y_speed = y_speed.to(device, non_blocking=True).float()  # B,1

        pred = model(x_seq)
        loss, loss_parts = compute_loss(pred, y_speed, loss_weights)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(
            loss=f"{np.mean(losses):.4f}",
            main=f"{loss_parts['main_loss']:.4f}",
        )

    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate validation metrics."""

    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for x_seq, y_speed in tqdm(loader, desc="val", leave=False):
        x_seq = x_seq.to(device, non_blocking=True).float()  # B,T,C,H,W
        y_speed = y_speed.to(device, non_blocking=True).float()  # B,1
        pred = model(x_seq)
        preds.append(pred["speed"].detach().cpu().view(-1))
        targets.append(y_speed.detach().cpu().view(-1))

    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    return {
        "mae": mae(pred_all, target_all),
        "rmse": rmse(pred_all, target_all),
        "mape": mape(pred_all, target_all),
    }


def save_config_copy(config: dict[str, Any], save_dir: Path) -> None:
    """Write the resolved training config into the run directory."""

    with (save_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-HBTNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-dir", default="runs/train")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    train_cfg = config.setdefault("training", {})
    loss_cfg = train_cfg.setdefault("loss", {})

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = choose_device(config)

    dataset = build_dataset(args, config)
    train_set, val_set = split_dataset(dataset, seed)
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
    )

    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, save_dir)

    print(f"device: {device}")
    print(f"dataset: {'synthetic' if args.synthetic else 'video'} train={len(train_set)} val={len(val_set)}")
    print(f"parameters: {count_parameters(model)}")

    best_mae = float("inf")
    epochs = int(train_cfg.get("epochs", 20))
    for epoch in range(1, epochs + 1):
        train_loss = run_train_epoch(model, train_loader, optimizer, device, loss_cfg, epoch)
        val_metrics = evaluate(model, val_loader, device)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_mape={val_metrics['mape']:.2f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, save_dir / "last.pt")
        if val_metrics["mae"] < best_mae:
            best_mae = val_metrics["mae"]
            torch.save(checkpoint, save_dir / "best.pt")


if __name__ == "__main__":
    main()
