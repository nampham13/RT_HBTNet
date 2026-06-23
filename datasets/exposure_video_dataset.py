from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_roi, stack_sequence


class ExposureVideoDataset(Dataset):
    """Manifest-based real-video dataset with known exposure settings.

    Required CSV columns:

    ``video_path,exposure_time_ms``

    Optional columns:

    ``fps,start_frame,end_frame,scene``
    """

    def __init__(
        self,
        manifest_path: str | Path,
        config: dict[str, Any] | None = None,
        video_root: str | Path | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Exposure manifest not found: {self.manifest_path}")
        self.config = config or {}
        self.video_root = Path(video_root) if video_root is not None else None
        self.sequence_length = int(self.config.get("data", {}).get("sequence_length", 5))
        self.records = self._read_manifest()
        if not self.records:
            raise ValueError(f"Exposure manifest has no samples: {self.manifest_path}")

    @property
    def group_ids(self) -> list[str]:
        return [str(record["scene"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[int(index)]
        video_path = self._resolve_video_path(record["video_path"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        file_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = float(record["fps"] or file_fps)
        if fps <= 0.0:
            cap.release()
            raise ValueError(f"FPS is unavailable for {video_path}")

        start = int(record["start_frame"])
        end = record["end_frame"]
        if end is None:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            end = max(start, frame_count - 1)
        indices = np.linspace(start, int(end), num=self.sequence_length)
        frames = []
        try:
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(float(frame_index))))
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
                frames.append(preprocess_roi(frame, self.config))
        finally:
            cap.release()

        alpha = float(record["exposure_time_ms"]) * 1.0e-3 * fps
        return {
            "frames": torch.from_numpy(stack_sequence(frames)).float(),
            "alpha": torch.tensor([float(np.clip(alpha, 0.0, 1.0))], dtype=torch.float32),
            "scene": str(record["scene"]),
            "video_path": str(video_path),
        }

    def _resolve_video_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if self.video_root is not None:
            return self.video_root / path
        return self.manifest_path.parent / path

    def _read_manifest(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"video_path", "exposure_time_ms"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Exposure manifest missing columns: {sorted(missing)}")
            for row_index, row in enumerate(reader, start=2):
                try:
                    video_path = str(row["video_path"])
                    records.append(
                        {
                            "video_path": video_path,
                            "exposure_time_ms": float(row["exposure_time_ms"]),
                            "fps": float(row["fps"]) if row.get("fps") else None,
                            "start_frame": int(float(row.get("start_frame") or 0)),
                            "end_frame": int(float(row["end_frame"])) if row.get("end_frame") else None,
                            "scene": str(row.get("scene") or Path(video_path).stem),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid exposure manifest row {row_index}: {row}") from exc
        return records
