# Minimal data layout

Giữ đơn giản nhất có thể. Bạn chỉ cần nhớ 3 thư mục:

```text
data/
  sintel/          # train/val synthetic blur từ frame + optical flow
  bsd/             # evaluate video thật có exposure metadata
  toy_exposure/    # smoke test, tự sinh bằng script
```

Trong đó:

- `data/sintel/` dùng cho training chính.
- `data/bsd/` dùng cho benchmark/evaluation thật.
- `data/toy_exposure/` chỉ dùng kiểm tra code, không báo cáo trong paper.

## 1. Training data: `data/sintel`

Code hiện tại đọc trực tiếp layout kiểu MPI Sintel:

```text
data/sintel/
  training/
    final/
      <scene>/
        frame_0001.png
        frame_0002.png
        ...
    flow/
      <scene>/
        frame_0001.flo
        frame_0002.flo
        ...
```

Ví dụ:

```text
data/sintel/training/final/alley_1/frame_0001.png
data/sintel/training/flow/alley_1/frame_0001.flo
```

Training command:

```powershell
python scripts/train.py --config configs/default.yaml
```

Nếu bạn để dataset ở chỗ khác:

```powershell
python scripts/train.py --config configs/default.yaml --data-root path/to/sintel
```

Loader sẽ tự sinh blur với nhiều giá trị exposure fraction `alpha`, nên không
cần label speed.

Để train nhanh hơn, loader mặc định render blur trực tiếp ở resolution train
(`render_at_target_resolution: true`). Đây là resolution mà model thật sự nhìn
thấy, nên target `motion_flow` và `blur_flow` vẫn khớp với input.

Visualize vài sample trước khi train:

```powershell
python scripts/visualize_data.py ^
  --config configs/default.yaml ^
  --dataset exposure_flow ^
  --count 8 ^
  --output-dir runs/visualize_data/sintel
```

Profile tốc độ ingest nếu train chậm:

```powershell
python scripts/profile_input_pipeline.py ^
  --config configs/default.yaml ^
  --data-root data/sintel ^
  --samples 64 ^
  --workers 0,2,4
```

## 2. Real benchmark: `data/bsd`

Với Beam-Splitter Dataset hoặc video thật có exposure time:

```text
data/bsd/
  videos/
    clip_0001.mp4
    clip_0002.mp4
  manifest.csv
```

`manifest.csv`:

```csv
video_path,exposure_time_ms,fps,start_frame,end_frame,scene
videos/clip_0001.mp4,8.0,15,0,99,bsd_scene_001
videos/clip_0002.mp4,4.0,30,0,149,bsd_scene_002
```

Chỉ hai cột bắt buộc:

```csv
video_path,exposure_time_ms
```

Các cột `fps,start_frame,end_frame,scene` là optional, nhưng nên có `fps` và
`scene` để evaluation rõ ràng hơn.

Evaluation command:

```powershell
python scripts/evaluate.py ^
  --config configs/default.yaml ^
  --weights runs/exposure/best.pt ^
  --dataset exposure_video ^
  --data-root data/bsd ^
  --manifest data/bsd/manifest.csv ^
  --output runs/exposure/bsd_report.json
```

Visualize video samples và metadata alpha:

```powershell
python scripts/visualize_data.py ^
  --config configs/default.yaml ^
  --dataset exposure_video ^
  --data-root data/bsd ^
  --manifest data/bsd/manifest.csv ^
  --count 8 ^
  --output-dir runs/visualize_data/bsd
```

## 3. Smoke test: `data/toy_exposure`

Tạo toy data:

```powershell
python scripts/make_toy_exposure_data.py
```

Chạy thử:

```powershell
python scripts/train.py --config configs/smoke.yaml --epochs 1 --num-workers 0
```

Visualize toy data:

```powershell
python scripts/visualize_data.py ^
  --config configs/smoke.yaml ^
  --dataset exposure_flow ^
  --count 4 ^
  --output-dir runs/visualize_data/toy
```

Toy data chỉ để xem pipeline có chạy không. Không dùng làm benchmark result.

## 4. Nếu dùng dataset khác

Đừng tạo thêm nhiều nhánh thư mục. Chọn một trong hai cách:

1. Nếu dataset có optical flow, convert về layout `data/sintel/...`.
2. Nếu dataset là video thật có exposure time, đặt vào `data/bsd/...` và viết
   `manifest.csv`.

Vậy là đủ cho hướng hiện tại.

## 5. Không commit data thật

Các thư mục dataset lớn được ignore bởi Git:

```text
data/sintel/
data/bsd/
data/toy_exposure/
```

Chỉ commit code, config, README và protocol.
