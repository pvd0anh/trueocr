# Hướng dẫn sử dụng

## Model

Gói kèm sẵn weights trong `trueocr/models/` (~8 MB), cài xong dùng được ngay.
Thư mục model gồm:

| File | Vai trò |
|---|---|
| `rec.onnx` | Nhận dạng ký tự trên crop từng dòng (CTC) |
| `det.onnx` | Phát hiện bbox từng dòng chữ |
| `orient.onnx` | Xác định hướng xoay của ảnh |
| `order.onnx` | Sắp thứ tự đọc các dòng |
| `meta.json` | Tên model, bảng ký tự, tham số tiền xử lý |

Chế độ `line` chỉ cần `rec.onnx` và `meta.json`. Dùng model khác thì truyền
đường dẫn bằng `model_dir=` (Python) hoặc `--model-dir` (CLI).

## Nhận dạng từng dòng với Recognizer

```python
from trueocr import Recognizer

ocr = Recognizer("/duong/dan/models", threads=1)
predictions = ocr.predict([bgr_array, png_bytes, "crop.jpg"], batch_size=32)
print(predictions[0].text, predictions[0].confidence)
```

## Hai chế độ

**`page`** (mặc định), cho ảnh nhiều dòng:

1. Đọc ảnh, áp EXIF, ghép nền trắng cho ảnh trong suốt, chuyển grayscale.
2. Ước lượng độ nghiêng (tối đa 45°), `orient.onnx` xác định hướng xoay, rồi xoay và khử nghiêng.
3. `det.onnx` phát hiện bbox từng dòng; ảnh chữ nhỏ được phát hiện lại ở độ phân giải cao hơn.
4. Cắt từng dòng, `rec.onnx` nhận dạng, giải mã CTC greedy, chuẩn hóa Unicode NFC.
5. `order.onnx` gom khối và sắp thứ tự đọc; ghép các dòng bằng ký tự xuống dòng.

**`line`**, cho ảnh đã crop sát một dòng nằm ngang: bỏ qua bước 2, 3 và 5.

```python
model.predict("crop.png", mode="line")
```

## Kết quả

`result.data` (cũng là JSON được lưu) gồm:

- `text`: toàn bộ text, mỗi dòng một hàng.
- `lines`: `index`, `text`, `confidence`, `bbox` `[x0, y0, x1, y1]`, `low_confidence`.
- `rotation_clockwise_degrees`, `skew_degrees`: góc đã xoay và khử nghiêng.
- `bbox_coordinate_system`: ở chế độ `page`, bbox thuộc ảnh đã xoay/khử nghiêng
  (`result.image`); ở chế độ `line`, bbox thuộc ảnh gốc sau khi áp EXIF.
- `model`, `mode`, `image_size`, `elapsed_ms`, `created_at`.

`confidence` là trung bình xác suất đỉnh CTC, chưa hiệu chuẩn thành xác suất
text đúng. `low_confidence` chỉ đánh dấu dòng cần xem lại, text vẫn được giữ.

## Cấu hình

```python
from trueocr import OCRConfig, load_model

model = load_model(OCRConfig(model_dir="/duong/dan/models", threads=2, low_confidence=0.9))
model = load_model(model_dir="/duong/dan/models", mode="line")  # ghi đè từng trường
```

| Trường | Mặc định | Ý nghĩa |
|---|---|---|
| `model_dir` | `trueocr/models` trong gói | Thư mục model |
| `mode` | `"page"` | `"page"` hoặc `"line"` |
| `threads` | `min(4, số CPU)` | Số CPU thread của ONNX Runtime |
| `batch_size` | `32` | Số dòng mỗi lần chạy recognizer |
| `auto_rotate` | `True` | Dùng `orient.onnx` để xoay ảnh |
| `low_confidence` | `0.80` | Ngưỡng đánh dấu `low_confidence` |
| `save_outputs` | `False` | Tự lưu kết quả sau mỗi `predict` |
| `output_dir` | `trueocr_outputs` | Nơi lưu khi `save_outputs=True` |

## Chất lượng và giới hạn

Model `trueocr-vi-r01` có 1.060.035 tham số, file `rec.onnx` FP32 4,08 MiB.
Trên 22.280 crop test nội bộ (giấy tờ tùy thân và hóa đơn), model đọc đúng toàn
dòng 96,72%, tỷ lệ lỗi ký tự (CER) 0,66%. Tốc độ trung vị 1,62 ms/crop (p95
2,46 ms) với 1 thread CPU Intel Xeon Platinum 8358P, batch 1.

- Recognizer chỉ đọc tiếng Việt và ký tự Latin cơ bản (222 ký tự), không dùng language model.
- Crop một dòng cần sát chữ; lề quá rộng hoặc nhiều dòng có thể sinh thêm ký tự.
- Model có thể nhận sai dấu tiếng Việt, nhất là ở font và bố cục khác dữ liệu huấn luyện.
- Chất lượng chế độ `page` phụ thuộc cả bước phát hiện và cắt dòng, không suy
  ra trực tiếp từ điểm trên crop.
