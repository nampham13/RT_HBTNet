# Data

The repository data layout follows the site-speed convention used in the main
README:

```text
manifests/site_speed/labels.csv
raw/site_speed/videos/*.mp4
```

`data/labels.csv` is kept as a tiny synthetic demo manifest for smoke checks.

For real data, use a CSV manifest with these required columns:

```csv
video_path,start_frame,end_frame,speed_mps
path/to/video_001.mp4,0,300,0.42
path/to/video_002.mp4,50,350,0.88
```

Frames are loaded as fixed-length ROI clips with tensor shape `T,C,H,W`,
normalized to `[0, 1]`.
