from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .flow_temporal_dataset import FlowTemporalDataset
from .paired_blur_dataset import PairedBlurDataset
from .video_speed_dataset import VideoSpeedDataset


class DatasetFactory:
    """Factory for training/evaluation dataset variants."""

    @staticmethod
    def create(
        *,
        config: dict[str, Any],
        labels_csv: str | Path | None = None,
        video_root: str | Path | None = None,
        dataset_type: str | None = None,
        split: str | None = None,
        root: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> Dataset:
        resolved_type = str(dataset_type or config.get("data", {}).get("dataset", "video"))
        resolved_type = resolved_type.lower()
        if resolved_type in {"video", "video_speed", "labeled_video"}:
            if labels_csv is None:
                raise ValueError("labels_csv is required for video datasets")
            return VideoSpeedDataset(
                labels_csv=labels_csv,
                video_root=video_root,
                config=config,
            )
        if resolved_type in {"paired_blur", "blur_pair", "blur_pairs"}:
            data_cfg = config.get("data", {}).get("datasets", {}).get("paired_blur", {})
            dataset_root = root or data_cfg.get("root")
            dataset_manifest = manifest_path or data_cfg.get("manifest")
            return PairedBlurDataset(
                root=dataset_root,
                manifest_path=dataset_manifest,
                split=split or data_cfg.get("split", "train"),
                config=config,
                dataset_key="paired_blur",
                blur_dir_name=data_cfg.get("blur_dir_name", "blur"),
                sharp_dir_name=data_cfg.get("sharp_dir_name", "sharp"),
            )
        if resolved_type in {"flow_temporal", "temporal_flow", "frame_flow"}:
            data_cfg = config.get("data", {}).get("datasets", {}).get("flow_temporal", {})
            dataset_root = root or data_cfg.get("root")
            if dataset_root is None:
                raise ValueError("root is required for frame+flow temporal datasets")
            return FlowTemporalDataset(
                root=dataset_root,
                split=split or data_cfg.get("split", "training"),
                image_subdir=data_cfg.get("image_subdir", data_cfg.get("pass", "final")),
                flow_subdir=data_cfg.get("flow_subdir", "flow"),
                dataset_key="flow_temporal",
                config=config,
            )
        raise ValueError(f"Unsupported dataset type: {resolved_type}")
