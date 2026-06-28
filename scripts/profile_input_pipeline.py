from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from datasets import DatasetFactory
from models.rt_hbtnet import build_model_from_config, count_parameters


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_workers(value: str) -> list[int]:
    workers = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            workers.append(int(item))
    if not workers:
        raise ValueError("--workers must contain at least one integer")
    return workers


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_getitem(dataset, samples: int) -> dict[str, float]:
    count = min(int(samples), len(dataset))
    if count <= 0:
        raise ValueError("dataset is empty")

    # Warm up decoder/remap paths.
    _ = dataset[0]

    start = time.perf_counter()
    for index in range(count):
        _ = dataset[index]
    elapsed = time.perf_counter() - start
    return {
        "samples": float(count),
        "seconds": float(elapsed),
        "samples_per_s": float(count / max(elapsed, 1.0e-9)),
        "ms_per_sample": float(elapsed * 1000.0 / max(count, 1)),
    }


def benchmark_loader(dataset, *, batch_size: int, samples: int, num_workers: int) -> tuple[dict[str, float], tuple[int, ...]]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        persistent_workers=int(num_workers) > 0,
        pin_memory=torch.cuda.is_available(),
    )
    target_samples = min(int(samples), len(dataset))
    seen = 0
    batches = 0
    first_shape: tuple[int, ...] | None = None

    start = time.perf_counter()
    for batch in loader:
        frames = batch["frames"]
        if first_shape is None:
            first_shape = tuple(frames.shape)
        seen += int(frames.shape[0])
        batches += 1
        if seen >= target_samples:
            break
    elapsed = time.perf_counter() - start
    if first_shape is None:
        raise ValueError("DataLoader produced no batches")
    return (
        {
            "samples": float(seen),
            "batches": float(batches),
            "seconds": float(elapsed),
            "samples_per_s": float(seen / max(elapsed, 1.0e-9)),
            "batches_per_s": float(batches / max(elapsed, 1.0e-9)),
            "ms_per_batch": float(elapsed * 1000.0 / max(batches, 1)),
        },
        first_shape,
    )


@torch.no_grad()
def benchmark_model_forward(config: dict[str, Any], batch_shape: tuple[int, ...], *, batches: int, device: torch.device) -> dict[str, float]:
    model = build_model_from_config(config).to(device).eval()
    frames = torch.randn(*batch_shape, device=device)

    warmup = min(5, max(1, int(batches) // 4))
    for _ in range(warmup):
        _ = model(frames)
    sync_if_needed(device)

    start = time.perf_counter()
    for _ in range(int(batches)):
        _ = model(frames)
    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    return {
        "parameters": float(count_parameters(model)),
        "batches": float(batches),
        "seconds": float(elapsed),
        "batches_per_s": float(float(batches) / max(elapsed, 1.0e-9)),
        "ms_per_batch": float(elapsed * 1000.0 / max(int(batches), 1)),
    }


def format_metric(row: dict[str, float], keys: list[str]) -> str:
    parts = []
    for key in keys:
        value = row[key]
        if key in {"samples", "batches", "parameters"}:
            parts.append(f"{key}={int(value):,}")
        else:
            parts.append(f"{key}={value:.3f}")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile BT-ShutterNet input pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["exposure_flow"], default="exposure_flow")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", default="0,2,4")
    parser.add_argument("--model-batches", type=int, default=50)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    data_cfg = config.get("data", {}).get("datasets", {}).get("exposure_flow", {})
    root = resolve_project_path(args.data_root or data_cfg.get("root", "data/sintel"))
    dataset = DatasetFactory.create(
        config=config,
        dataset_type=args.dataset,
        root=root,
        split=args.split or data_cfg.get("split", "training"),
    )

    train_cfg = config.get("training", {})
    batch_size = int(args.batch_size or train_cfg.get("batch_size", 8))
    print(f"dataset={args.dataset} root={root}")
    print(f"length={len(dataset):,} groups={len(set(getattr(dataset, 'group_ids', []))):,}")
    print(f"batch_size={batch_size} samples_to_profile={min(args.samples, len(dataset)):,}")

    getitem = benchmark_getitem(dataset, args.samples)
    print("getitem     " + format_metric(getitem, ["samples", "seconds", "samples_per_s", "ms_per_sample"]))

    best_shape: tuple[int, ...] | None = None
    for worker_count in parse_workers(args.workers):
        row, shape = benchmark_loader(
            dataset,
            batch_size=batch_size,
            samples=args.samples,
            num_workers=worker_count,
        )
        if best_shape is None:
            best_shape = shape
        print(
            f"dataloader workers={worker_count:<2d} "
            + format_metric(row, ["samples", "batches", "seconds", "samples_per_s", "ms_per_batch"])
        )

    if best_shape is not None and int(args.model_batches) > 0:
        requested = str(config.get("project", {}).get("device", "auto")).lower()
        if requested == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested)
        model_row = benchmark_model_forward(config, best_shape, batches=int(args.model_batches), device=device)
        print(
            f"model_forward device={device} input_shape={best_shape} "
            + format_metric(model_row, ["parameters", "batches", "seconds", "ms_per_batch", "batches_per_s"])
        )


if __name__ == "__main__":
    main()
