from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_roi, stack_sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


class FlowTemporalDataset(Dataset):
    """Generic frame+flow dataset for temporal-branch pretraining.

    The dataset family expects RGB frames and dense optical-flow files aligned
    by frame stem. MPI Sintel is one preset, but any dataset exported into this
    layout can reuse the adapter without adding a new Python class.
    """

    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any] | None = None,
        split: str = "training",
        image_subdir: str = "final",
        flow_subdir: str = "flow",
        dataset_key: str = "flow_temporal",
    ) -> None:
        self.root = Path(root)
        self.config = config or {}
        self.split = str(split)
        self.image_subdir = str(image_subdir)
        self.flow_subdir = str(flow_subdir)
        self.dataset_key = str(dataset_key)
        data_cfg = self.config.get("data", {})
        dataset_cfg = data_cfg.get("datasets", {}).get(self.dataset_key, {})
        self.sequence_length = int(data_cfg.get("sequence_length", 64))
        self.stride = int(dataset_cfg.get("stride", max(1, self.sequence_length // 2)))
        self.target_unit = str(dataset_cfg.get("target_unit", "flow_px_per_frame"))
        self.pixels_per_meter = float(data_cfg.get("pixels_per_meter", dataset_cfg.get("pixels_per_meter", 1.0)))
        self.fps = float(dataset_cfg.get("fps", self.config.get("inference", {}).get("target_fps", 30.0)))
        self.records = self._scan_clips()
        if not self.records:
            raise ValueError(f"No frame+flow clips found for dataset family '{self.dataset_key}' under {self.root}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        frame_paths, flow_paths = self.records[int(index)]
        frames = [preprocess_roi(self._load_image(path), self.config) for path in frame_paths]
        x_seq = torch.from_numpy(stack_sequence(frames)).float()
        target = self._flow_target(flow_paths)
        return x_seq, torch.tensor([target], dtype=torch.float32)

    def _scan_clips(self) -> list[tuple[list[Path], list[Path]]]:
        image_root = self.root / self.split / self.image_subdir
        flow_root = self.root / self.split / self.flow_subdir
        if not image_root.exists():
            raise FileNotFoundError(f"Frame directory not found: {image_root}")
        if not flow_root.exists():
            raise FileNotFoundError(f"Flow directory not found: {flow_root}")

        records: list[tuple[list[Path], list[Path]]] = []
        for scene_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            frames = sorted(path for path in scene_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
            if len(frames) < self.sequence_length:
                continue
            scene_flow_dir = flow_root / scene_dir.name
            for start in range(0, len(frames) - self.sequence_length + 1, self.stride):
                clip_frames = frames[start : start + self.sequence_length]
                flow_paths = []
                for frame_path in clip_frames[:-1]:
                    flow_path = scene_flow_dir / f"{frame_path.stem}.flo"
                    if not flow_path.exists():
                        break
                    flow_paths.append(flow_path)
                if len(flow_paths) == self.sequence_length - 1:
                    records.append((clip_frames, flow_paths))
        return records

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read frame: {path}")
        return image

    def _flow_target(self, flow_paths: list[Path]) -> float:
        magnitudes = []
        for flow_path in flow_paths:
            flow = read_middlebury_flo(flow_path)
            mag = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))
            valid = np.isfinite(mag)
            if np.any(valid):
                magnitudes.append(float(np.mean(mag[valid])))
        flow_px = float(np.mean(magnitudes)) if magnitudes else 0.0
        if self.target_unit == "mps":
            return flow_px * self.fps / max(self.pixels_per_meter, 1.0e-6)
        return flow_px


def read_middlebury_flo(path: str | Path) -> np.ndarray:
    """Read a Middlebury/Sintel .flo optical-flow file."""

    flow_path = Path(path)
    with flow_path.open("rb") as handle:
        magic = struct.unpack("f", handle.read(4))[0]
        if abs(magic - 202021.25) > 1.0e-4:
            raise ValueError(f"Invalid .flo magic number in {flow_path}")
        width = struct.unpack("i", handle.read(4))[0]
        height = struct.unpack("i", handle.read(4))[0]
        data = np.frombuffer(handle.read(), dtype=np.float32)
    expected = height * width * 2
    if data.size != expected:
        raise ValueError(f"Invalid .flo payload size in {flow_path}: expected {expected}, got {data.size}")
    return data.reshape(height, width, 2)
