from __future__ import annotations

from utils.onnx_export import dummy_shape


def test_onnx_dummy_shape_uses_roi_resize_size() -> None:
    config = {
        "data": {
            "image_size": {"height": 64, "width": 128},
            "sequence_length": 9,
        },
        "roi": {
            "resize_height": 32,
            "resize_width": 48,
        },
        "model": {
            "in_channels": 2,
        },
    }

    assert dummy_shape(config) == (1, 9, 2, 32, 48)
