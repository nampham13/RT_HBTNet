from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.preprocessing import preprocess_roi, stack_sequence

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_middlebury_flo(path: str | Path) -> np.ndarray:
    """Read a Middlebury ``.flo`` optical-flow file."""

    import struct

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
        raise ValueError(f"Invalid .flo payload size in {flow_path}")
    return data.reshape(height, width, 2)


def resize_flow(flow: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize optical flow while preserving displacement in pixel units."""

    source_h, source_w = flow.shape[:2]
    resized = cv2.resize(flow, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= float(width) / max(float(source_w), 1.0)
    resized[..., 1] *= float(height) / max(float(source_h), 1.0)
    return resized.astype(np.float32)


def synthesize_motion_blur(
    image: np.ndarray,
    flow: np.ndarray,
    alpha: float,
    integration_samples: int = 9,
    gamma: float = 2.2,
) -> np.ndarray:
    """Render centered exposure blur from an image and inter-frame flow.

    The local constant-velocity model uses ``alpha * flow`` as the exposure
    trajectory. Sampling is centered on the input frame, so the total path
    length is exactly the requested exposure fraction times inter-frame flow.
    """

    if image.shape[:2] != flow.shape[:2]:
        raise ValueError("image and flow must have matching spatial dimensions")
    if integration_samples < 2:
        raise ValueError("integration_samples must be at least 2")

    alpha = float(np.clip(alpha, 0.0, 1.0))
    image_f = image.astype(np.float32) / 255.0
    linear = np.power(np.clip(image_f, 0.0, 1.0), float(gamma))
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )

    accumulation = np.zeros_like(linear, dtype=np.float32)
    for tau in np.linspace(-0.5 * alpha, 0.5 * alpha, int(integration_samples), dtype=np.float32):
        map_x = grid_x - tau * flow[..., 0]
        map_y = grid_y - tau * flow[..., 1]
        warped = cv2.remap(
            linear,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        accumulation += warped

    blurred_linear = accumulation / float(integration_samples)
    blurred = np.power(np.clip(blurred_linear, 0.0, 1.0), 1.0 / float(gamma))
    return np.clip(np.rint(blurred * 255.0), 0, 255).astype(np.uint8)


class ExposureFlowDataset(Dataset):
    """Physics-supervised exposure-fraction dataset built from frame+flow data.

    Expected layout follows MPI Sintel-style folders:

    ``<root>/<split>/<image_subdir>/<scene>/frame_XXXX.png``
    ``<root>/<split>/<flow_subdir>/<scene>/frame_XXXX.flo``

    Each sample contains a synthetically blurred clip, exposure fraction,
    center-frame inter-frame flow, exposure blur flow, and validity mask.
    """

    def __init__(
        self,
        root: str | Path,
        config: dict[str, Any] | None = None,
        split: str = "training",
        image_subdir: str = "final",
        flow_subdir: str = "flow",
        dataset_key: str = "exposure_flow",
    ) -> None:
        self.root = Path(root)
        self.config = config or {}
        self.split = str(split)
        self.image_subdir = str(image_subdir)
        self.flow_subdir = str(flow_subdir)
        self.dataset_key = str(dataset_key)

        data_cfg = self.config.get("data", {})
        dataset_cfg = data_cfg.get("datasets", {}).get(self.dataset_key, {})
        self.sequence_length = int(data_cfg.get("sequence_length", 5))
        if self.sequence_length < 3 or self.sequence_length % 2 == 0:
            raise ValueError("sequence_length must be an odd integer >= 3")
        self.stride = int(dataset_cfg.get("stride", 1))
        self.samples_per_clip = int(dataset_cfg.get("samples_per_clip", 2))
        self.integration_samples = int(dataset_cfg.get("integration_samples", 9))
        self.gamma = float(dataset_cfg.get("gamma", 2.2))
        self.alpha_min = float(dataset_cfg.get("alpha_min", 0.02))
        self.alpha_max = float(dataset_cfg.get("alpha_max", 0.95))
        self.alpha_values = tuple(float(v) for v in dataset_cfg.get("alpha_values", ()))
        self.max_flow = float(dataset_cfg.get("max_flow", 400.0))
        self.seed = int(self.config.get("project", {}).get("seed", 42))

        roi_cfg = self.config.get("roi", {})
        image_size = data_cfg.get("image_size", {"height": 64, "width": 128})
        self.target_height = int(roi_cfg.get("resize_height", image_size.get("height", 64)))
        self.target_width = int(roi_cfg.get("resize_width", image_size.get("width", 128)))

        self.records = self._scan_clips()
        if not self.records:
            raise ValueError(f"No exposure-flow clips found under {self.root}")

    @property
    def group_ids(self) -> list[str]:
        return [str(record["scene"]) for record in self.records for _ in range(self.samples_per_clip)]

    def __len__(self) -> int:
        return len(self.records) * self.samples_per_clip

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record_index = int(index) // self.samples_per_clip
        variant = int(index) % self.samples_per_clip
        record = self.records[record_index]
        alpha = self._sample_alpha(record_index, variant)

        frames: list[np.ndarray] = []
        flows: list[np.ndarray] = []
        for frame_path, flow_path in zip(record["frame_paths"], record["flow_paths"]):
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read frame: {frame_path}")
            flow = read_middlebury_flo(flow_path)
            valid = np.isfinite(flow).all(axis=2)
            clean_flow = np.where(valid[..., None], flow, 0.0).astype(np.float32)
            frames.append(synthesize_motion_blur(
                image,
                clean_flow,
                alpha=alpha,
                integration_samples=self.integration_samples,
                gamma=self.gamma,
            ))
            flows.append(clean_flow)

        x_seq = torch.from_numpy(
            stack_sequence([preprocess_roi(frame, self.config) for frame in frames])
        ).float()

        center = self.sequence_length // 2
        center_flow_original = flows[center]
        valid_original = (
            np.isfinite(center_flow_original).all(axis=2)
            & (np.linalg.norm(center_flow_original, axis=2) <= self.max_flow)
        )
        center_flow = resize_flow(center_flow_original, self.target_width, self.target_height)
        valid_mask = cv2.resize(
            valid_original.astype(np.uint8),
            (self.target_width, self.target_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.float32)

        motion_flow = torch.from_numpy(center_flow.transpose(2, 0, 1)).float()
        blur_flow = motion_flow * float(alpha)
        return {
            "frames": x_seq,
            "alpha": torch.tensor([alpha], dtype=torch.float32),
            "motion_flow": motion_flow,
            "blur_flow": blur_flow,
            "valid_mask": torch.from_numpy(valid_mask[None]).float(),
            "scene": str(record["scene"]),
        }

    def _sample_alpha(self, record_index: int, variant: int) -> float:
        if self.alpha_values:
            return self.alpha_values[(record_index * self.samples_per_clip + variant) % len(self.alpha_values)]
        rng = np.random.default_rng(self.seed + record_index * 1009 + variant * 9176)
        return float(rng.uniform(self.alpha_min, self.alpha_max))

    def _scan_clips(self) -> list[dict[str, Any]]:
        image_root = self.root / self.split / self.image_subdir
        flow_root = self.root / self.split / self.flow_subdir
        if not image_root.exists():
            raise FileNotFoundError(f"Frame directory not found: {image_root}")
        if not flow_root.exists():
            raise FileNotFoundError(f"Flow directory not found: {flow_root}")

        records: list[dict[str, Any]] = []
        for scene_dir in sorted(path for path in image_root.iterdir() if path.is_dir()):
            frames = sorted(path for path in scene_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
            if len(frames) < self.sequence_length:
                continue
            scene_flow_dir = flow_root / scene_dir.name
            for start in range(0, len(frames) - self.sequence_length + 1, self.stride):
                clip_frames = frames[start : start + self.sequence_length]
                flow_paths = [scene_flow_dir / f"{frame_path.stem}.flo" for frame_path in clip_frames]
                if all(path.exists() for path in flow_paths):
                    records.append(
                        {
                            "scene": scene_dir.name,
                            "frame_paths": clip_frames,
                            "flow_paths": flow_paths,
                        }
                    )
        return records
