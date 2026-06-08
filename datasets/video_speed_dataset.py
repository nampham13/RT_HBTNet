from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_roi, stack_sequence
from utils.roi import detect_motion_rois, extract_rois, is_auto_motion_mode


def _default_config(sequence_length: int = 64) -> dict[str, Any]:
    """Default preprocessing config used when no YAML config is provided."""

    return {
        "data": {
            "image_size": {"height": 64, "width": 128},
            "sequence_length": int(sequence_length),
            "grayscale": True,
            "normalize": True,
            "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": 8},
        },
        "roi": {
            "mode": "full",
            "rois": [],
            "resize_width": 128,
            "resize_height": 64,
        },
    }


class VideoSpeedDataset(Dataset):
    """Dataset for labeled conveyor videos.

    Expected CSV columns:
        ``video_path,start_frame,end_frame,speed_mps``

    Each sample returns:
        ``x_seq``: ``torch.Tensor`` with shape ``T,C,H,W``
        ``y_speed``: ``torch.Tensor`` with shape ``1``
    """

    def __init__(
        self,
        labels_csv: str | Path | None = None,
        config: dict[str, Any] | None = None,
        sequence_length: int | None = None,
        roi_index: int = 0,
        video_root: str | Path | None = None,
        manifest_path: str | Path | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        # ``manifest_path`` and legacy kwargs keep older scripts importable.
        csv_path = labels_csv or manifest_path
        if csv_path is None:
            raise ValueError("labels_csv is required")

        self.labels_csv = Path(csv_path)
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"labels.csv not found: {self.labels_csv}")

        self.config = config or self._config_from_legacy(sequence_length, legacy_kwargs)
        self.video_root = Path(video_root) if video_root is not None else None
        self.sequence_length = int(
            sequence_length
            or self.config.get("data", {}).get("sequence_length")
            or legacy_kwargs.get("sequence_len")
            or 64
        )
        self.roi_index = int(roi_index)
        self.records = self._read_labels(self.labels_csv)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[int(index)]
        video_path = self._resolve_video_path(record["video_path"])
        frame_indices = self._sample_frame_indices(record["start_frame"], record["end_frame"])

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video for dataset sample {index}: {video_path}")

        raw_frames: list[np.ndarray] = []
        try:
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Could not read frame {frame_idx} from video '{video_path}' "
                        f"for dataset sample {index}"
                    )

                raw_frames.append(frame)
        finally:
            cap.release()

        sample_config = self._sample_roi_config(raw_frames)
        frames_chw: list[np.ndarray] = []
        for frame in raw_frames:
            rois = extract_rois(frame, sample_config)
            if not rois:
                raise RuntimeError(f"No ROIs extracted from video '{video_path}' for dataset sample {index}")
            if self.roi_index >= len(rois):
                raise IndexError(
                    f"roi_index={self.roi_index} is out of range for {len(rois)} ROI(s) "
                    f"in video '{video_path}'"
                )

            roi = rois[self.roi_index]
            frames_chw.append(preprocess_roi(roi, sample_config))  # C,H,W

        x_seq = torch.from_numpy(stack_sequence(frames_chw)).float()  # T,C,H,W
        y_speed = torch.tensor([float(record["speed_mps"])], dtype=torch.float32)  # 1
        return x_seq, y_speed

    def _sample_roi_config(self, raw_frames: list[np.ndarray]) -> dict[str, Any]:
        if not is_auto_motion_mode(self.config):
            return self.config
        if self.config.get("roi", {}).get("rois"):
            return self.config

        sample_config = deepcopy(self.config)
        roi_cfg = sample_config.setdefault("roi", {})
        roi_cfg["rois"] = [list(roi) for roi in detect_motion_rois(raw_frames, sample_config)]
        return sample_config

    def _resolve_video_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if self.video_root is not None:
            return self.video_root / path
        return self.labels_csv.parent / path

    def _sample_frame_indices(self, start_frame: int, end_frame: int) -> list[int]:
        if end_frame < start_frame:
            raise ValueError(f"end_frame ({end_frame}) must be >= start_frame ({start_frame})")
        if self.sequence_length <= 1:
            return [int(start_frame)]
        indices = np.linspace(start_frame, end_frame, num=self.sequence_length)
        return [int(round(v)) for v in indices]

    @staticmethod
    def _read_labels(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"video_path", "start_frame", "end_frame", "speed_mps"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"labels.csv missing required columns: {sorted(missing)}")

            for row_idx, row in enumerate(reader, start=2):
                try:
                    records.append(
                        {
                            "video_path": str(row["video_path"]),
                            "start_frame": int(float(row["start_frame"])),
                            "end_frame": int(float(row["end_frame"])),
                            "speed_mps": float(row["speed_mps"]),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid labels.csv values on row {row_idx}: {row}") from exc

        if not records:
            raise ValueError(f"labels.csv has no samples: {path}")
        return records

    @staticmethod
    def _config_from_legacy(sequence_length: int | None, kwargs: dict[str, Any]) -> dict[str, Any]:
        image_size = kwargs.get("image_size", (64, 128))
        height, width = int(image_size[0]), int(image_size[1])
        channels = int(kwargs.get("channels", 1))
        cfg = _default_config(sequence_length or int(kwargs.get("sequence_len", 64)))
        cfg["data"]["image_size"] = {"height": height, "width": width}
        cfg["data"]["grayscale"] = channels == 1
        cfg["roi"]["resize_width"] = width
        cfg["roi"]["resize_height"] = height
        default_roi = kwargs.get("default_roi")
        if default_roi is not None:
            cfg["roi"]["mode"] = "fixed"
            cfg["roi"]["rois"] = [list(default_roi)]
        return cfg
