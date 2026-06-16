from __future__ import annotations

import struct

import cv2
import numpy as np

from datasets.factory import DatasetFactory
from datasets.flow_temporal_dataset import read_middlebury_flo


def _base_config(sequence_length: int = 3) -> dict:
    return {
        "data": {
            "sequence_length": sequence_length,
            "grayscale": True,
            "normalize": True,
            "clahe": {"enabled": False},
            "image_size": {"height": 8, "width": 10},
            "datasets": {},
        },
        "roi": {
            "mode": "full",
            "rois": [],
            "resize_width": 10,
            "resize_height": 8,
        },
    }


def _write_image(path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((12, 16, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_flo(path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, _ = flow.shape
    with path.open("wb") as handle:
        handle.write(struct.pack("f", 202021.25))
        handle.write(struct.pack("i", width))
        handle.write(struct.pack("i", height))
        handle.write(flow.astype(np.float32).tobytes())


def test_paired_blur_dataset_scans_pairs(tmp_path) -> None:
    blur_path = tmp_path / "train" / "seq1" / "blur" / "000001.png"
    sharp_path = tmp_path / "train" / "seq1" / "sharp" / "000001.png"
    _write_image(blur_path, 80)
    _write_image(sharp_path, 140)

    dataset = DatasetFactory.create(
        dataset_type="paired_blur",
        root=tmp_path,
        split="train",
        config=_base_config(sequence_length=2),
    )

    x_seq, y_target = dataset[0]

    assert len(dataset) == 1
    assert x_seq.shape == (2, 1, 8, 10)
    assert y_target.shape == (1,)
    assert float(y_target.item()) > 0.0


def test_flow_temporal_dataset_reads_clip_and_flow_target(tmp_path) -> None:
    scene = "alley_1"
    for idx in range(1, 4):
        _write_image(tmp_path / "training" / "final" / scene / f"frame_{idx:04d}.png", 40 + idx)
    flow = np.zeros((5, 6, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    flow[..., 1] = 4.0
    _write_flo(tmp_path / "training" / "flow" / scene / "frame_0001.flo", flow)
    _write_flo(tmp_path / "training" / "flow" / scene / "frame_0002.flo", flow)

    dataset = DatasetFactory.create(
        dataset_type="flow_temporal",
        root=tmp_path,
        config=_base_config(sequence_length=3),
    )

    x_seq, y_target = dataset[0]

    assert len(dataset) == 1
    assert x_seq.shape == (3, 1, 8, 10)
    assert np.isclose(float(y_target.item()), 5.0)
    assert read_middlebury_flo(tmp_path / "training" / "flow" / scene / "frame_0001.flo").shape == (5, 6, 2)
