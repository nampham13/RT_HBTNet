# Data

The prototype can run without real conveyor footage by using
`SyntheticSpeedDataset`.

For real data, use a CSV manifest with at least these columns:

```csv
video_path,speed_mps
path/to/video_001.mp4,0.42
path/to/video_002.mp4,0.88
```

Optional columns:

- `start_frame`: first frame used for the training sequence.
- `roi_x`, `roi_y`, `roi_w`, `roi_h`: per-video crop rectangle.

Frames are loaded as fixed-length ROI clips with tensor shape
`T,C,H,W`, normalized to `[0, 1]`.
