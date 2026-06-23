from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .exposure_flow_dataset import ExposureFlowDataset
from .exposure_video_dataset import ExposureVideoDataset


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
        resolved_type = str(dataset_type or config.get("data", {}).get("dataset", "exposure_flow"))
        resolved_type = resolved_type.lower()
        if resolved_type in {"exposure_flow", "synthetic_exposure", "blur_flow"}:
            data_cfg = config.get("data", {}).get("datasets", {}).get("exposure_flow", {})
            dataset_root = root or data_cfg.get("root")
            if dataset_root is None:
                raise ValueError("root is required for exposure-flow datasets")
            return ExposureFlowDataset(
                root=dataset_root,
                split=split or data_cfg.get("split", "training"),
                image_subdir=data_cfg.get("image_subdir", "final"),
                flow_subdir=data_cfg.get("flow_subdir", "flow"),
                dataset_key="exposure_flow",
                config=config,
            )
        if resolved_type in {"exposure_video", "real_exposure", "bsd"}:
            if manifest_path is None:
                raise ValueError("manifest_path is required for exposure-video datasets")
            return ExposureVideoDataset(
                manifest_path=manifest_path,
                video_root=video_root or root,
                config=config,
            )
        raise ValueError(f"Unsupported dataset type: {resolved_type}")
