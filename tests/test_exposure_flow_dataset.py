from __future__ import annotations

import struct

import cv2
import numpy as np

from datasets import DatasetFactory
from datasets.exposure_flow_dataset import synthesize_motion_blur


def _write_image(path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0, 255, 24, dtype=np.uint8)
    image = np.tile(x[None, :, None], (16, 1, 3))
    image = np.roll(image, offset, axis=1)
    cv2.imwrite(str(path), image)


def _write_flo(path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = flow.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("f", 202021.25))
        handle.write(struct.pack("i", width))
        handle.write(struct.pack("i", height))
        handle.write(flow.astype(np.float32).tobytes())


def _config() -> dict:
    return {
        "project": {"seed": 7},
        "data": {
            "sequence_length": 3,
            "image_size": {"height": 8, "width": 12},
            "grayscale": True,
            "normalize": True,
            "clahe": {"enabled": False},
            "datasets": {
                "exposure_flow": {
                    "stride": 1,
                    "samples_per_clip": 1,
                    "integration_samples": 5,
                    "alpha_values": [0.5],
                }
            },
        },
        "roi": {
            "mode": "full",
            "rois": [],
            "resize_height": 8,
            "resize_width": 12,
        },
    }


def test_exposure_flow_dataset_returns_physics_targets(tmp_path) -> None:
    scene = "scene_1"
    flow = np.zeros((16, 24, 2), dtype=np.float32)
    flow[..., 0] = 4.0
    for idx in range(1, 5):
        stem = f"frame_{idx:04d}"
        _write_image(tmp_path / "training" / "final" / scene / f"{stem}.png", idx)
        _write_flo(tmp_path / "training" / "flow" / scene / f"{stem}.flo", flow)

    dataset = DatasetFactory.create(
        config=_config(),
        dataset_type="exposure_flow",
        root=tmp_path,
    )
    sample = dataset[0]

    assert sample["frames"].shape == (3, 1, 8, 12)
    assert sample["motion_flow"].shape == (2, 8, 12)
    assert sample["blur_flow"].shape == (2, 8, 12)
    assert sample["valid_mask"].shape == (1, 8, 12)
    assert np.isclose(float(sample["alpha"].item()), 0.5)
    assert np.allclose(
        sample["blur_flow"].numpy(),
        0.5 * sample["motion_flow"].numpy(),
    )
    assert dataset.group_ids[0] == scene


def test_zero_flow_blur_renderer_preserves_image() -> None:
    image = np.full((8, 10, 3), 127, dtype=np.uint8)
    flow = np.zeros((8, 10, 2), dtype=np.float32)
    blurred = synthesize_motion_blur(image, flow, alpha=0.8, integration_samples=5)
    assert np.max(np.abs(blurred.astype(np.int16) - image.astype(np.int16))) <= 1
