from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_roi, stack_sequence


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


class PairedBlurDataset(Dataset):
    """Generic blur/sharp-pair dataset for blur-branch pretraining.

    The dataset family expects a degraded frame and a reference frame. It is
    intentionally not tied to GOPRO: any public or private dataset with
    blur/sharp-like pairs can be used through either:

    - A manifest with ``blur_path`` plus optional ``sharp_path`` and
      ``target``/``speed_mps``.
    - A directory layout containing sibling folders such as ``blur`` and
      ``sharp`` under each sequence.
    """

    def __init__(
        self,
        root: str | Path | None = None,
        config: dict[str, Any] | None = None,
        split: str = "train",
        manifest_path: str | Path | None = None,
        dataset_key: str = "paired_blur",
        blur_dir_name: str = "blur",
        sharp_dir_name: str = "sharp",
    ) -> None:
        self.root = Path(root) if root is not None else None
        self.config = config or {}
        self.split = str(split)
        self.dataset_key = str(dataset_key)
        self.blur_dir_name = str(blur_dir_name)
        self.sharp_dir_name = str(sharp_dir_name)
        self.sequence_length = int(self.config.get("data", {}).get("sequence_length", 64))
        dataset_cfg = self.config.get("data", {}).get("datasets", {}).get(self.dataset_key, {})
        self.target_scale = float(dataset_cfg.get("target_scale", 10.0))

        if manifest_path is not None:
            self.records = self._read_manifest(Path(manifest_path))
        else:
            if self.root is None:
                raise ValueError("root is required when manifest_path is not provided")
            self.records = self._scan_pairs(self.root, self.split)

        if not self.records:
            raise ValueError(f"No paired-blur samples found for dataset family '{self.dataset_key}'")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[int(index)]
        blur = self._load_image(record["blur_path"])
        x_frame = preprocess_roi(blur, self.config)
        x_seq = torch.from_numpy(stack_sequence([x_frame.copy() for _ in range(self.sequence_length)])).float()

        target_value = record.get("target")
        if target_value is None:
            sharp_path = record.get("sharp_path")
            if sharp_path is None:
                raise ValueError("Paired-blur samples without an explicit target require sharp_path")
            sharp = self._load_image(sharp_path)
            target_value = self._blur_proxy_target(blur, sharp)

        y_target = torch.tensor([float(target_value)], dtype=torch.float32)
        return x_seq, y_target

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return image

    def _blur_proxy_target(self, blur: np.ndarray, sharp: np.ndarray) -> float:
        if blur.shape[:2] != sharp.shape[:2]:
            sharp = cv2.resize(sharp, (blur.shape[1], blur.shape[0]), interpolation=cv2.INTER_AREA)
        blur_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        sharp_gray = cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        reconstruction_gap = float(np.mean(np.abs(sharp_gray - blur_gray)))
        sharp_hf = float(cv2.Laplacian(sharp_gray, cv2.CV_32F).var())
        blur_hf = float(cv2.Laplacian(blur_gray, cv2.CV_32F).var())
        attenuation = max(0.0, sharp_hf - blur_hf) / max(sharp_hf, 1.0e-6)
        return self.target_scale * (0.65 * reconstruction_gap + 0.35 * attenuation)

    def _read_manifest(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"Paired-blur manifest not found: {path}")

        records: list[dict[str, Any]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"blur_path"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Paired-blur manifest missing required columns: {sorted(missing)}")

            for row_idx, row in enumerate(reader, start=2):
                blur_path = self._resolve_path(row["blur_path"], path.parent)
                sharp_value = row.get("sharp_path") or ""
                target_value = row.get("target") or row.get("speed_mps") or ""
                records.append(
                    {
                        "blur_path": blur_path,
                        "sharp_path": self._resolve_path(sharp_value, path.parent) if sharp_value else None,
                        "target": float(target_value) if target_value else None,
                        "row": row_idx,
                    }
                )
        return records

    def _resolve_path(self, value: str, base_dir: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if self.root is not None and (self.root / path).exists():
            return self.root / path
        return base_dir / path

    def _scan_pairs(self, root: Path, split: str) -> list[dict[str, Any]]:
        search_root = root / split if (root / split).exists() else root
        records: list[dict[str, Any]] = []
        for blur_dir in search_root.rglob(self.blur_dir_name):
            if not blur_dir.is_dir():
                continue
            sharp_dir = blur_dir.parent / self.sharp_dir_name
            if not sharp_dir.exists():
                continue
            for blur_path in sorted(blur_dir.rglob("*")):
                if blur_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                rel = blur_path.relative_to(blur_dir)
                sharp_path = sharp_dir / rel
                if sharp_path.exists():
                    records.append({"blur_path": blur_path, "sharp_path": sharp_path, "target": None})
        return records
