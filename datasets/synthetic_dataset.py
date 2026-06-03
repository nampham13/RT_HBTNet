from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.augmentations import apply_low_light_blur_augmentations
from utils.preprocessing import preprocess_roi, stack_sequence


@dataclass(frozen=True)
class SyntheticConfig:
    """Settings for synthetic conveyor ROI generation."""

    num_samples: int = 2000
    sequence_length: int = 64
    height: int = 64
    width: int = 128
    grayscale: bool = True
    speed_range_mps: tuple[float, float] = (0.5, 5.0)
    fps: float = 30.0
    pixels_per_meter: float = 64.0
    seed: int = 42


class SyntheticSpeedDataset(Dataset):
    """Synthetic conveyor-like ROI sequence dataset.

    It creates a random textured belt patch, shifts it horizontally according
    to a synthetic speed, then applies configurable low-light, dust, noise,
    compression, contrast, and blur augmentations. The goal is not perfect
    physics; it is a lightweight signal for testing the training and inference
    pipeline.

    Each sample returns:
        ``x_seq``: ``torch.Tensor`` with shape ``T,C,H,W``
        ``y_speed``: ``torch.Tensor`` with shape ``1``
    """

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        num_samples: int = 2000,
        speed_range_mps: Sequence[float] = (0.5, 5.0),
        seed: int | None = None,
        apply_motion_blur: bool = True,
        sequence_length: int | None = None,
        sequence_len: int | None = None,
        image_size: Sequence[int] | None = None,
        channels: int | None = None,
        fps: float | None = None,
        pixels_per_meter: float | None = None,
        **_: Any,
    ) -> None:
        self.config = config or self._legacy_config(
            sequence_length=sequence_length or sequence_len,
            image_size=image_size,
            channels=channels,
        )
        data_cfg = self.config.get("data", {})
        roi_cfg = self.config.get("roi", {})
        image_cfg = data_cfg.get("image_size", {"height": 64, "width": 128})

        height = int(roi_cfg.get("resize_height", image_cfg["height"] if isinstance(image_cfg, dict) else image_cfg[0]))
        width = int(roi_cfg.get("resize_width", image_cfg["width"] if isinstance(image_cfg, dict) else image_cfg[1]))
        seq_len = int(sequence_length or sequence_len or data_cfg.get("sequence_length", 64))
        grayscale = bool(data_cfg.get("grayscale", True))

        self.cfg = SyntheticConfig(
            num_samples=int(num_samples),
            sequence_length=seq_len,
            height=height,
            width=width,
            grayscale=grayscale,
            speed_range_mps=(float(speed_range_mps[0]), float(speed_range_mps[1])),
            fps=float(fps or self.config.get("inference", {}).get("target_fps", 30.0)),
            pixels_per_meter=float(pixels_per_meter or data_cfg.get("pixels_per_meter", 64.0)),
            seed=int(seed if seed is not None else self.config.get("project", {}).get("seed", 42)),
        )
        if not apply_motion_blur:
            self.config.setdefault("augmentation", {})["motion_blur_prob"] = 0.0

    def __len__(self) -> int:
        return self.cfg.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.cfg.seed + int(index))
        speed_mps = float(rng.uniform(*self.cfg.speed_range_mps))
        frames = self._make_sequence(rng, speed_mps)
        x_seq = torch.from_numpy(stack_sequence(frames)).float()  # T,C,H,W
        y_speed = torch.tensor([speed_mps], dtype=torch.float32)  # 1
        return x_seq, y_speed

    def _make_sequence(self, rng: np.random.Generator, speed_mps: float) -> list[np.ndarray]:
        texture = self._make_texture(rng, self.cfg.height, self.cfg.width)
        pixels_per_frame = speed_mps * self.cfg.pixels_per_meter / max(self.cfg.fps, 1.0)

        frames: list[np.ndarray] = []
        for t in range(self.cfg.sequence_length):
            shift_px = int(round(t * pixels_per_frame))
            frame = np.roll(texture, shift=shift_px, axis=1)  # H,W horizontal belt motion
            frame = self._augment_frame(rng, frame, pixels_per_frame)
            if not self.cfg.grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # H,W,3 BGR
            frames.append(preprocess_roi(frame, self.config))  # C,H,W
        return frames

    @staticmethod
    def _make_texture(rng: np.random.Generator, height: int, width: int) -> np.ndarray:
        base = rng.normal(120.0, 40.0, size=(height, width)).astype(np.float32)

        # Conveyor-like texture: subtle horizontal fibers, vertical seams, speckles.
        for y in range(0, height, int(rng.integers(5, 12))):
            base[y : y + 1, :] += rng.uniform(8.0, 24.0)
        for x in range(0, width, int(rng.integers(12, 28))):
            base[:, x : x + 2] += rng.uniform(18.0, 55.0)
        for _ in range(max(16, int(height * width * 0.01))):
            cx = int(rng.integers(0, width))
            cy = int(rng.integers(0, height))
            radius = int(rng.integers(1, 3))
            color = float(rng.uniform(30.0, 230.0))
            cv2.circle(base, (cx, cy), radius, color, -1)

        base = cv2.GaussianBlur(base, (3, 3), 0)
        return np.clip(base, 0, 255).astype(np.uint8)

    def _augment_frame(self, rng: np.random.Generator, frame: np.ndarray, pixels_per_frame: float) -> np.ndarray:
        del pixels_per_frame  # Motion blur is configured by augmentation kernel range.
        return apply_low_light_blur_augmentations(frame, rng, self.config)

    @staticmethod
    def _legacy_config(
        sequence_length: int | None,
        image_size: Sequence[int] | None,
        channels: int | None,
    ) -> dict[str, Any]:
        height, width = (64, 128) if image_size is None else (int(image_size[0]), int(image_size[1]))
        return {
            "project": {"seed": 42},
            "data": {
                "image_size": {"height": height, "width": width},
                "sequence_length": int(sequence_length or 64),
                "grayscale": int(channels or 1) == 1,
                "normalize": True,
                "clahe": {"enabled": True, "clip_limit": 2.0, "tile_grid_size": 8},
            },
            "roi": {
                "mode": "full",
                "rois": [],
                "resize_width": width,
                "resize_height": height,
            },
            "augmentation": {
                "enabled": True,
                "brightness": [0.2, 1.2],
                "gamma": [0.6, 1.8],
                "gaussian_noise_std": [0.0, 0.08],
                "motion_blur_prob": 0.5,
                "motion_blur_kernel": [3, 15],
                "dust_prob": 0.3,
                "contrast": [0.4, 1.0],
                "jpeg_prob": 0.2,
                "jpeg_quality": [35, 95],
            },
            "inference": {"target_fps": 30},
        }
