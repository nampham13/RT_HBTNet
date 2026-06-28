from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import yaml

try:
    from ._bootstrap import ROOT
except ImportError:
    from _bootstrap import ROOT

from datasets import DatasetFactory
from utils.visualization import (
    make_sample_visualization,
    save_rgb_image,
    summarize_sample,
    write_summary_csv,
)


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else ROOT / path


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_indices(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    indices: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bounds = [item.strip() for item in part.split(":")]
            if len(bounds) not in {2, 3}:
                raise ValueError(f"Invalid index range: {part}")
            start = int(bounds[0] or 0)
            stop = int(bounds[1])
            step = int(bounds[2] or 1) if len(bounds) == 3 else 1
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(part))
    return indices


def build_dataset(args: argparse.Namespace, config: dict[str, Any]):
    if args.dataset == "exposure_flow":
        data_cfg = config.get("data", {}).get("datasets", {}).get("exposure_flow", {})
        root = resolve_project_path(args.data_root or data_cfg.get("root", "data/sintel"))
        return DatasetFactory.create(
            config=config,
            dataset_type="exposure_flow",
            root=root,
            split=args.split or data_cfg.get("split", "training"),
        )

    root = resolve_project_path(args.data_root or "data/bsd")
    manifest = resolve_project_path(args.manifest or (root / "manifest.csv"))
    return DatasetFactory.create(
        config=config,
        dataset_type="exposure_video",
        root=root,
        video_root=root,
        manifest_path=manifest,
    )


def choose_indices(args: argparse.Namespace, dataset_length: int) -> list[int]:
    explicit = parse_indices(args.indices)
    if explicit is not None:
        indices = explicit
    elif args.random:
        rng = random.Random(int(args.seed))
        pool = list(range(dataset_length))
        rng.shuffle(pool)
        indices = pool[: int(args.count)]
    else:
        start = int(args.start)
        indices = list(range(start, min(dataset_length, start + int(args.count))))

    cleaned = []
    for index in indices:
        if index < 0 or index >= dataset_length:
            raise IndexError(f"Sample index {index} is outside dataset length {dataset_length}")
        cleaned.append(int(index))
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize BT-ShutterNet dataset samples")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=["exposure_flow", "exposure_video"], default="exposure_flow")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default="runs/visualize_data")
    parser.add_argument("--indices", default=None, help="Comma/range list, e.g. 0,5,10:15")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panel-width", type=int, default=192)
    parser.add_argument("--panel-height", type=int, default=96)
    parser.add_argument("--max-flow", type=float, default=None)
    parser.add_argument("--max-blur", type=float, default=None)
    parser.add_argument("--min-motion-px", type=float, default=0.05)
    args = parser.parse_args()

    config = load_config(resolve_project_path(args.config))
    dataset = build_dataset(args, config)
    indices = choose_indices(args, len(dataset))
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for index in indices:
        sample = dataset[index]
        image = make_sample_visualization(
            sample,
            index=index,
            panel_width=int(args.panel_width),
            panel_height=int(args.panel_height),
            max_flow=args.max_flow,
            max_blur=args.max_blur,
            min_motion_px=float(args.min_motion_px),
        )
        image_path = output_dir / f"{args.dataset}_{index:06d}.png"
        save_rgb_image(image, image_path)

        summary = summarize_sample(sample, min_motion_px=float(args.min_motion_px))
        summary.update({"index": index, "image_path": str(image_path)})
        summary_rows.append(summary)
        print(f"wrote {image_path}")

    summary_path = output_dir / "summary.csv"
    write_summary_csv(summary_rows, summary_path)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
