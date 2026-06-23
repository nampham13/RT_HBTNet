from __future__ import annotations

import cv2
import numpy as np

from datasets.exposure_video_dataset import ExposureVideoDataset


def test_real_exposure_manifest_converts_time_and_fps_to_alpha(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"placeholder")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "video_path,exposure_time_ms,fps,start_frame,end_frame,scene\n"
        "clip.mp4,8,25,0,4,scene_a\n",
        encoding="utf-8",
    )
    frames = [np.full((12, 18, 3), idx * 20, dtype=np.uint8) for idx in range(5)]

    class FakeCapture:
        def __init__(self, source):
            self.index = 0

        def isOpened(self):
            return True

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 25.0
            if prop_id == cv2.CAP_PROP_FRAME_COUNT:
                return len(frames)
            return 0.0

        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self.index = int(value)
            return True

        def read(self):
            return True, frames[self.index].copy()

        def release(self):
            return None

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)
    config = {
        "data": {
            "sequence_length": 3,
            "image_size": {"height": 8, "width": 10},
            "grayscale": True,
            "normalize": True,
            "clahe": {"enabled": False},
        },
        "roi": {"mode": "full", "resize_height": 8, "resize_width": 10},
    }
    dataset = ExposureVideoDataset(manifest, config=config)
    sample = dataset[0]

    assert sample["frames"].shape == (3, 1, 8, 10)
    assert np.isclose(float(sample["alpha"].item()), 0.2)
    assert dataset.group_ids == ["scene_a"]
