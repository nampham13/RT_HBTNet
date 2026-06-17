from __future__ import annotations

import argparse
import csv
import random
import time
from collections.abc import Sized
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from datasets import DatasetFactory
from models.rt_hbtnet import build_model_from_config
from utils.metrics import mae, mape, rmse


def sync_if_cuda(device: torch.device) -> None:
    """Synchronize CUDA so timing reflects completed GPU work."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def elapsed_since(start: float, device: torch.device) -> float:
    """Return elapsed wall time after synchronizing pending CUDA kernels."""

    sync_if_cuda(device)
    return time.perf_counter() - start


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
    """Build the selected dataset."""

    dataset_type = args.dataset
    labels_csv = None
    video_root = None
    if dataset_type in {"video", "video_speed", "labeled_video"}:
        labels_csv = resolve_project_path(args.labels)
        video_root = resolve_project_path(args.video_root)

    return DatasetFactory.create(
        config=config,
        dataset_type=dataset_type,
        labels_csv=labels_csv,
        video_root=video_root,
        root=None if args.data_root is None else resolve_project_path(args.data_root),
        manifest_path=None if args.manifest is None else resolve_project_path(args.manifest),
        split=args.split,
    )


def split_dataset(dataset: Sized, seed: int) -> tuple[Subset[Any], Subset[Any]]:
    """Split one dataset into train/validation subsets."""

    total = len(dataset)
    if total < 2:
        raise ValueError("Need at least 2 samples to create a train/val split")
    val_count = max(1, int(round(total * 0.2)))
    train_count = total - val_count
    if train_count < 1:
        train_count, val_count = 1, total - 1
    generator = torch.Generator().manual_seed(int(seed))
    train_set, val_set = random_split(cast(Dataset[Any], dataset), [train_count, val_count], generator=generator)
    return train_set, val_set


def compute_loss(
    pred: dict[str, torch.Tensor],
    gt: torch.Tensor,
    weights: dict[str, float],
    branch: str = "joint",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute branch-aware L1 losses."""

    gt = gt.view(-1, 1)
    l1 = nn.functional.l1_loss
    tex_loss = l1(pred["speed_tex"], gt)
    blur_loss = l1(pred["speed_blur"], gt)
    main_loss = l1(pred["speed"], gt)

    if branch == "temporal":
        total = tex_loss
        main_loss = tex_loss.detach()
    elif branch == "blur":
        total = blur_loss
        main_loss = blur_loss.detach()
    else:
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
        if branch == "temporal":
            conf_reg = l1(pred["conf_tex"], tex_conf_target)
        elif branch == "blur":
            conf_reg = l1(pred["conf_blur"], blur_conf_target)
        else:
            conf_reg = l1(pred["conf_tex"], tex_conf_target) + l1(pred["conf_blur"], blur_conf_target)
        if "obs_quality" in pred:
            # Context quality is high when at least one visual branch has a
            # reliable observation; it is not supervised as a speed estimate.
            quality_target = torch.maximum(tex_conf_target, blur_conf_target)
            if branch == "joint":
                conf_reg = conf_reg + l1(pred["obs_quality"], quality_target)
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
    branch: str = "joint",
    profile_timing: bool = False,
) -> dict[str, float]:
    """Run one training epoch and return mean loss values."""

    model.train()
    losses: list[float] = []
    loss_parts_accum: dict[str, list[float]] = {
        "main_loss": [],
        "tex_loss": [],
        "blur_loss": [],
        "conf_reg": [],
    }
    timing = {
        "train_data_wait_s": 0.0,
        "train_h2d_s": 0.0,
        "train_forward_loss_s": 0.0,
        "train_backward_step_s": 0.0,
    }
    progress = tqdm(loader, desc=f"train {epoch}", leave=False)
    sync_if_cuda(device)
    wait_start = time.perf_counter()
    for x_seq, y_speed in progress:
        if profile_timing:
            timing["train_data_wait_s"] += time.perf_counter() - wait_start

        start = time.perf_counter()
        x_seq = x_seq.to(device, non_blocking=True).float()  # B,T,C,H,W
        y_speed = y_speed.to(device, non_blocking=True).float()  # B,1
        if profile_timing:
            timing["train_h2d_s"] += elapsed_since(start, device)

        start = time.perf_counter()
        pred = model(x_seq)
        loss, loss_parts = compute_loss(pred, y_speed, loss_weights, branch=branch)
        if profile_timing:
            timing["train_forward_loss_s"] += elapsed_since(start, device)

        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if profile_timing:
            timing["train_backward_step_s"] += elapsed_since(start, device)

        losses.append(float(loss.detach().cpu()))
        for name, value in loss_parts.items():
            loss_parts_accum[name].append(float(value))
        progress.set_postfix(
            loss=f"{np.mean(losses):.4f}",
            main=f"{loss_parts['main_loss']:.4f}",
        )
        sync_if_cuda(device)
        wait_start = time.perf_counter()

    return {
        "train_loss": float(np.mean(losses)) if losses else 0.0,
        **{
            name: float(np.mean(values)) if values else 0.0
            for name, values in loss_parts_accum.items()
        },
        **timing,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    branch: str = "joint",
    profile_timing: bool = False,
) -> dict[str, float]:
    """Evaluate validation metrics."""

    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    timing = {
        "val_data_wait_s": 0.0,
        "val_h2d_s": 0.0,
        "val_forward_s": 0.0,
    }
    sync_if_cuda(device)
    wait_start = time.perf_counter()
    for x_seq, y_speed in tqdm(loader, desc="val", leave=False):
        if profile_timing:
            timing["val_data_wait_s"] += time.perf_counter() - wait_start

        start = time.perf_counter()
        x_seq = x_seq.to(device, non_blocking=True).float()  # B,T,C,H,W
        y_speed = y_speed.to(device, non_blocking=True).float()  # B,1
        if profile_timing:
            timing["val_h2d_s"] += elapsed_since(start, device)

        start = time.perf_counter()
        pred = model(x_seq)
        if profile_timing:
            timing["val_forward_s"] += elapsed_since(start, device)

        pred_key = {"joint": "speed", "temporal": "speed_tex", "blur": "speed_blur"}[branch]
        preds.append(pred[pred_key].detach().cpu().view(-1))
        targets.append(y_speed.detach().cpu().view(-1))
        sync_if_cuda(device)
        wait_start = time.perf_counter()

    pred_all = torch.cat(preds)
    target_all = torch.cat(targets)
    return {
        "mae": mae(pred_all, target_all),
        "rmse": rmse(pred_all, target_all),
        "mape": mape(pred_all, target_all),
        **timing,
    }


def save_config_copy(config: dict[str, Any], save_dir: Path) -> None:
    """Write the resolved training config into the run directory."""

    with (save_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def write_history_csv(history: list[dict[str, float]], output_path: Path) -> None:
    """Write per-epoch training history to CSV."""

    if not history:
        return
    fieldnames = list(history[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def plot_training_history(history: list[dict[str, float]], output_path: Path) -> None:
    """Save a PNG chart with training losses and validation metrics."""

    if not history:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    loss_axis = axes[0]
    for key, label in (
        ("train_loss", "total"),
        ("main_loss", "fused"),
        ("tex_loss", "texture"),
        ("blur_loss", "blur"),
        ("conf_reg", "confidence reg"),
    ):
        values = [row[key] for row in history]
        loss_axis.plot(epochs, values, marker="o", linewidth=1.6, label=label)
    loss_axis.set_title("Training losses")
    loss_axis.set_ylabel("L1 loss")
    loss_axis.grid(True, alpha=0.3)
    loss_axis.legend(loc="best")

    metric_axis = axes[1]
    metric_axis.plot(epochs, [row["val_mae"] for row in history], marker="o", linewidth=1.6, label="MAE")
    metric_axis.plot(epochs, [row["val_rmse"] for row in history], marker="o", linewidth=1.6, label="RMSE")
    metric_axis.set_title("Validation speed error")
    metric_axis.set_xlabel("Epoch")
    metric_axis.set_ylabel("m/s")
    metric_axis.grid(True, alpha=0.3)
    metric_axis.legend(loc="upper left")

    mape_axis = metric_axis.twinx()
    mape_axis.plot(
        epochs,
        [row["val_mape"] for row in history],
        color="tab:red",
        linestyle="--",
        linewidth=1.4,
        label="MAPE",
    )
    mape_axis.set_ylabel("MAPE (%)")
    mape_axis.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def configure_branch_training(model: nn.Module, branch: str) -> list[nn.Parameter]:
    """Freeze modules outside the requested branch and return trainable params."""

    if branch == "joint":
        for param in model.parameters():
            param.requires_grad = True
        return [param for param in model.parameters() if param.requires_grad]

    for param in model.parameters():
        param.requires_grad = False

    module_names = ["frame_encoder"]
    if branch == "temporal":
        module_names.extend(["texture_head", "texture_branch"])
    elif branch == "blur":
        module_names.extend(["blur_head", "blur_branch"])
    else:
        raise ValueError(f"Unsupported training branch: {branch}")

    for name in module_names:
        module = getattr(model, name, None)
        if module is None:
            continue
        for param in module.parameters():
            param.requires_grad = True

    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError(f"No trainable parameters selected for branch '{branch}'")
    return trainable


def total_parameters(model: nn.Module) -> int:
    """Count all model parameters, including frozen modules."""

    return sum(param.numel() for param in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RT-HBTNet")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--dataset",
        default=None,
        choices=["video", "paired_blur", "flow_temporal"],
    )
    parser.add_argument("--branch", default=None, choices=["joint", "blur", "temporal"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--labels", default="data/labels.csv")
    parser.add_argument("--video-root", default="data/videos")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--save-dir", default="runs/train")
    parser.add_argument("--no-plots", dest="plots", action="store_false", default=True)
    parser.add_argument("--profile-timing", action="store_true", help="Print per-epoch timing breakdowns")
    args = parser.parse_args()

    config_path = resolve_project_path(args.config)
    config = load_config(config_path)
    train_cfg = config.setdefault("training", {})
    loss_cfg = train_cfg.setdefault("loss", {})
    if args.dataset is None:
        args.dataset = "video"
    if args.branch is None:
        args.branch = str(train_cfg.get("branch", "joint"))
    train_cfg["branch"] = args.branch

    if args.epochs is not None:
        train_cfg["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        train_cfg["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        train_cfg["num_workers"] = int(args.num_workers)
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = choose_device(config)

    dataset = cast(Sized, build_dataset(args, config))
    train_set, val_set = split_dataset(dataset, seed)
    batch_size = int(train_cfg.get("batch_size", 8))
    num_workers = int(train_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = build_model_from_config(config).to(device)
    trainable_params = configure_branch_training(model, args.branch)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(train_cfg.get("lr", 1.0e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1.0e-5)),
    )

    save_dir = resolve_project_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_config_copy(config, save_dir)

    print(f"device: {device}")
    print(f"dataset: {args.dataset} train={len(train_set)} val={len(val_set)}")
    print(f"branch: {args.branch}")
    print(f"parameters: total={total_parameters(model)} trainable={sum(p.numel() for p in trainable_params)}")

    best_mae = float("inf")
    epochs = int(train_cfg.get("epochs", 20))
    history: list[dict[str, float]] = []
    history_path = save_dir / "history.csv"
    chart_path = save_dir / "training_curves.png"
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_start = time.perf_counter()
        train_metrics = run_train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_cfg,
            epoch,
            branch=args.branch,
            profile_timing=bool(args.profile_timing),
        )
        train_elapsed = elapsed_since(train_start, device)

        val_start = time.perf_counter()
        val_metrics = evaluate(model, val_loader, device, branch=args.branch, profile_timing=bool(args.profile_timing))
        val_elapsed = elapsed_since(val_start, device)
        epoch_history = {
            "epoch": float(epoch),
            **train_metrics,
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_mape": float(val_metrics["mape"]),
            "epoch_time_s": 0.0,
            "train_time_s": train_elapsed,
            "val_time_s": val_elapsed,
            "artifact_time_s": 0.0,
        }
        history.append(epoch_history)
        artifact_start = time.perf_counter()
        write_history_csv(history, history_path)
        if args.plots:
            plot_training_history(history, chart_path)
        artifact_elapsed = elapsed_since(artifact_start, device)
        epoch_history["artifact_time_s"] = artifact_elapsed

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_mae={val_metrics['mae']:.4f} "
            f"val_rmse={val_metrics['rmse']:.4f} "
            f"val_mape={val_metrics['mape']:.2f}"
        )
        print(f"saved history: {history_path}")
        if args.plots:
            print(f"saved chart: {chart_path}")

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
        epoch_history["epoch_time_s"] = elapsed_since(epoch_start, device)
        write_history_csv(history, history_path)
        if args.profile_timing:
            print(
                "timing: "
                f"epoch={epoch_history['epoch_time_s']:.2f}s "
                f"train={train_elapsed:.2f}s "
                f"val={val_elapsed:.2f}s "
                f"artifacts={artifact_elapsed:.2f}s"
            )
            print(
                "train detail: "
                f"data_wait={train_metrics['train_data_wait_s']:.2f}s "
                f"h2d={train_metrics['train_h2d_s']:.2f}s "
                f"forward_loss={train_metrics['train_forward_loss_s']:.2f}s "
                f"backward_step={train_metrics['train_backward_step_s']:.2f}s"
            )
            print(
                "val detail: "
                f"data_wait={val_metrics['val_data_wait_s']:.2f}s "
                f"h2d={val_metrics['val_h2d_s']:.2f}s "
                f"forward={val_metrics['val_forward_s']:.2f}s"
            )

if __name__ == "__main__":
    main()
