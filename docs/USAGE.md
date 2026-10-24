# Sử dụng

Python ≥3.10. `pip install trueocr` cài cả code và model; inference chạy
offline trên CPU. Jupyter cần thêm `pip install "trueocr[notebook]"`.

## Python

```python
from trueocr import load_model

model = load_model(threads=2)
result = model.predict("anh.jpg")

print(result.text)
for line in result.lines:
    print(line["text"], line["bbox"], line["confidence"])

result.save("output")
```

Load model một lần, dùng lại cho các ảnh sau. `predict()` nhận `str`, `Path`,
encoded image bytes hoặc `PIL.Image`. Đường dẫn tương đối tính từ thư mục
đang chạy. PDF cần render thành ảnh trước.

| Mode | Input | Pipeline |
|---|---|---|
| `page` — mặc định | Ảnh nhiều dòng | Orientation, deskew, text detection, recognition, reading order |
| `line` | Crop sát một dòng nằm ngang | Recognition trực tiếp, bỏ qua detector và orientation/deskew |

```python
result = model.predict("crop.png", mode="line")
```

## Kết quả

| Attribute | Nội dung |
|---|---|
| `result.text` | Text các dòng, nối bằng `\n` |
| `result.lines` | Danh sách `index`, `text`, `bbox`, `confidence`, `low_confidence` |
| `result.image` | Ảnh sau preprocessing, chưa vẽ bbox |
| `result.data` | Dữ liệu JSON: text, lines, model, mode, image size, rotation, latency |
| `result.files` | Đường dẫn `text`, `json`, `preview` sau khi lưu |

`bbox` có dạng `[x0, y0, x1, y1]`. Ở mode `page`, tọa độ thuộc
`result.image` sau rotation/deskew; ở mode `line`, thuộc ảnh sau áp EXIF.
Kiểm tra `result.data["bbox_coordinate_system"]` khi ghép với ảnh gốc.

`confidence` là mean CTC peak probability, chưa được calibrate thành xác suất
dòng đúng. `low_confidence` chỉ đánh dấu dòng dưới threshold; text vẫn được
trả về. Text detection không tìm được dòng thì `text == ""` và `lines == []`.

`result.save("output")` ghi `.txt` UTF-8, `.json`, `_boxes.png` với timestamp
trong tên file và trả về dict đường dẫn. `result.show()` hiển thị text,
bbox và confidence trong Jupyter.

Mặc định `predict()` không ghi file. Có thể bật cho cả model hoặc từng ảnh:

```python
model = load_model(save_outputs=True, output_dir="output")
result = model.predict("anh.jpg", save=False)  # Override cho lần gọi này
```

## Config

Truyền trực tiếp vào `load_model(...)` hoặc qua `OCRConfig`. Keyword argument
override field tương ứng trong config.

| Parameter | Default | Ý nghĩa |
|---|---|---|
| `model_dir` | Model kèm package | Thư mục weights và `meta.json` |
| `mode` | `"page"` | `page` hoặc `line` |
| `threads` | `min(4, số CPU)` | Thread mỗi ONNX session; mode page còn dùng cho deskew |
| `batch_size` | `32` | Số dòng tối đa mỗi batch recognition |
| `auto_rotate` | `True` | Bật orientation; deskew vẫn chạy khi tắt |
| `low_confidence` | `0.80` | Threshold đánh dấu dòng cần xem lại |
| `save_outputs` | `False` | Tự lưu sau mỗi `predict()` |
| `output_dir` | `"trueocr_outputs"` | Thư mục lưu mặc định |

```python
from trueocr import OCRConfig, load_model

config = OCRConfig(mode="line", threads=1)
model = load_model(config, batch_size=64)
```

## Batch crop

Khi đã có crop từng dòng, gọi recognizer để bỏ qua phần đọc trang:

```python
model = load_model(mode="line", threads=1)
predictions = model.recognizer.predict(["dong_1.png", "dong_2.png"], batch_size=32)
for prediction in predictions:
    print(prediction.text, prediction.confidence)
```

Recognizer nhận path, bytes hoặc NumPy `uint8` grayscale/BGR/BGRA. Kết quả
giữ thứ tự đầu vào. Có thể khởi tạo riêng bằng
`Recognizer(model_dir, threads=1)` từ `trueocr`.

## CLI

```bash
trueocr anh.jpg
trueocr crop.png --mode line --threads 1
trueocr *.jpg --json --out output/
trueocr --help
```

`--json` in một JSON mỗi ảnh trên một dòng. `--out` lưu thêm text, JSON và ảnh
bbox. Khi một ảnh lỗi, CLI ghi lỗi ra stderr và tiếp tục ảnh sau; exit code
là `1` nếu có ảnh lỗi, `0` nếu tất cả thành công.

## Model riêng

Model mặc định `trueocr-vi-r01` có charset 222 ký tự Việt/Latin cơ bản,
CTC greedy decoding, không dùng language model.

```python
model = load_model(model_dir="/path/to/models")
```

CLI dùng `--model-dir`. Model phải tương thích với ONNX input/output contract và
preprocessing của TrueOCR; thay charset đơn thuần không thêm được ký tự mới.

| File | Vai trò |
|---|---|
| `rec.onnx` | Line recognition, CTC greedy decoding |
| `det.onnx` | Text detection |
| `orient.onnx` | Orientation |
| `order.onnx` | Reading order |
| `meta.json` | Charset và preprocessing parameters |

Mode `line` chỉ cần `rec.onnx` và `meta.json`. Mode `page` cần đủ năm file.
Chất lượng, tốc độ và phạm vi đánh giá nằm trong [benchmark](BENCHMARKS.md).
