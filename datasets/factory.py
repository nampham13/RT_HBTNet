from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .synthetic_dataset import SyntheticSpeedDataset
from .video_speed_dataset import VideoSpeedDataset


class DatasetFactory:
    """Factory for training/evaluation dataset variants."""

    @staticmethod
    def create(
        *,
        synthetic: bool,
        config: dict[str, Any],
        labels_csv: str | Path | None = None,
        video_root: str | Path | None = None,
    ) -> Dataset:
        if synthetic:
            return SyntheticSpeedDataset(config=config)
        if labels_csv is None:
            raise ValueError("labels_csv is required for video datasets")
        return VideoSpeedDataset(
            labels_csv=labels_csv,
            video_root=video_root,
            config=config,
        )
