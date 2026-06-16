from __future__ import annotations

import cv2
import numpy as np

from datasets.video_speed_dataset import VideoSpeedDataset


def test_video_speed_dataset_caches_video_decode(monkeypatch, tmp_path) -> None:
    video_path = tmp_path / "cached_video.mp4"
    video_path.write_bytes(b"placeholder")

    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "video_path,start_frame,end_frame,speed_mps\n"
        f"{video_path.name},0,3,1.25\n",
        encoding="utf-8",
    )

    frames = [
        np.full((12, 18, 3), fill_value=value, dtype=np.uint8)
        for value in (10, 40, 80, 120)
    ]

    class FakeCapture:
        open_count = 0

        def __init__(self, source):
            type(self).open_count += 1
            self._frames = [frame.copy() for frame in frames]
            self._index = 0
            self._opened = True

        def isOpened(self):
            return self._opened

        def set(self, prop_id, value):
            if prop_id == cv2.CAP_PROP_POS_FRAMES:
                self._index = int(value)
                return True
            return False

        def read(self):
            if self._index >= len(self._frames):
                return False, None
            frame = self._frames[self._index].copy()
            self._index += 1
            return True, frame

        def release(self):
            self._opened = False

    monkeypatch.setattr(cv2, "VideoCapture", FakeCapture)

    config = {
        "data": {
            "sequence_length": 4,
            "grayscale": True,
            "normalize": True,
            "clahe": {"enabled": False},
            "image_size": {"height": 8, "width": 10},
        },
        "roi": {
            "mode": "full",
            "rois": [],
            "resize_width": 10,
            "resize_height": 8,
        },
    }

    dataset = VideoSpeedDataset(labels_csv=labels_path, config=config, video_root=tmp_path)

    first_x, first_y = dataset[0]
    second_x, second_y = dataset[0]

    assert FakeCapture.open_count == 1
    assert first_x.shape == (4, 1, 8, 10)
    assert second_x.shape == (4, 1, 8, 10)
    assert np.isclose(float(first_y.item()), 1.25)
    assert np.isclose(float(second_y.item()), 1.25)