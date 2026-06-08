from __future__ import annotations

import cv2
import numpy as np

from utils.roi import detect_motion_rois, extract_rois, is_auto_motion_mode


def _moving_texture_frames() -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in range(12):
        frame = np.zeros((96, 128, 3), dtype=np.uint8)
        x0 = 24 + idx * 2
        x1 = 82 + idx * 2
        cv2.rectangle(frame, (x0, 36), (x1, 60), (120, 120, 120), -1)
        for x in range(x0, x1, 6):
            cv2.line(frame, (x, 36), (x, 60), (220, 220, 220), 1)
        frames.append(frame)
    return frames


def test_detect_motion_rois_finds_moving_texture_region() -> None:
    config = {
        "roi": {
            "mode": "auto_motion",
            "auto_motion": {
                "min_area_ratio": 0.001,
                "max_area_ratio": 0.8,
                "margin_ratio": 0.05,
                "score_threshold": 15.0,
                "score_percentile": 55.0,
            },
        }
    }

    rois = detect_motion_rois(_moving_texture_frames(), config)

    assert len(rois) == 1
    x, y, w, h = rois[0]
    assert x <= 32
    assert y <= 38
    assert x + w >= 90
    assert y + h >= 58


def test_detect_motion_rois_ignores_static_texture() -> None:
    frame = np.zeros((96, 128, 3), dtype=np.uint8)
    for x in range(20, 100, 4):
        cv2.line(frame, (x, 24), (x, 72), (220, 220, 220), 1)
    frames = [frame.copy() for _ in range(8)]
    config = {
        "roi": {
            "mode": "auto_motion",
            "auto_motion": {
                "min_area_ratio": 0.001,
                "score_threshold": 15.0,
                "score_percentile": 55.0,
            },
        }
    }

    assert detect_motion_rois(frames, config) == []


def test_auto_motion_extract_rois_uses_detected_boxes() -> None:
    frame = np.zeros((48, 80, 3), dtype=np.uint8)
    frame[10:30, 20:50] = 255
    config = {
        "roi": {
            "mode": "auto_motion",
            "rois": [[20, 10, 30, 20]],
            "resize_width": 32,
            "resize_height": 16,
        }
    }

    rois = extract_rois(frame, config)

    assert is_auto_motion_mode(config)
    assert len(rois) == 1
    assert rois[0].shape == (16, 32, 3)


def test_auto_motion_extract_rois_falls_back_to_full_frame() -> None:
    frame = np.zeros((48, 80, 3), dtype=np.uint8)
    config = {
        "roi": {
            "mode": "auto_motion",
            "rois": [],
            "resize_width": 32,
            "resize_height": 16,
        }
    }

    rois = extract_rois(frame, config)

    assert len(rois) == 1
    assert rois[0].shape == (16, 32, 3)
