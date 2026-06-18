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

# Todo
- tình cảnh hiện tại: 
    - không có 1 bộ dataset hoàn chỉnh có đủ các yếu tố:
    Video Conveyor
    +
    Motion Blur
    +
    Low Light
    +
    Ground Truth Speed

    - việc huấn luyện đã được hỗ trợ cho từng nhánh riêng biệt chạy mạc định thì sẽ train cả 2 nhánh 1 lúc, hoặc có thể train độc lập và fine-tune joint 

- cần phải làm 
    - tìm ra 2 bộ dữ liệu để train cho 2 nhánh:
        + nhánh blurs: dataset yêu cầu phải có blurs kernel và speed ground truth
        + nhánh temporal: dataset yêu cầu phải có speed ground truth và vật thể không bị nhòe
        + data fine-tune: video từ hầm mỏ sẽ được gắn nhãn tốc độ sau
        + Lưu ý: tất cả đều phải là video
    - sửa lại input đầu vào: hiện tại input đầu vào cho nhánh blurs đang là ảnh từ tập GOPRO => không phù hợp. nhánh temporal đã sử dụng video làm input.
    - cấu trúc thư mục data đang quá spagetti, cần thiết kế lại.
    - huấn luyện lại mô hình với 2 nhánh cùng lúc, hoặc có thể train độc lập rồi fine-tune joint.
    - phân tích data và kết quả huấn luyện.