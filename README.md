# TrueOCR

OCR tiếng Việt trên CPU, sử dụng ONNX Runtime. Hỗ trợ ảnh nhiều dòng và ảnh
crop từng dòng, kèm sẵn model khoảng 8 MB.

- Tự xoay, khử nghiêng, phát hiện dòng và sắp thứ tự đọc.
- Trả về văn bản, tọa độ từng dòng và điểm tin cậy.
- Sử dụng qua Python, CLI hoặc Jupyter; không cần GPU hay PyTorch.

## Cài đặt

Yêu cầu Python 3.10 trở lên.

```bash
pip install trueocr
```

Để hiển thị kết quả trong Jupyter: `pip install "trueocr[notebook]"`.

## Sử dụng với Python

```python
from trueocr import load_model

model = load_model()
result = model.predict("anh.jpg")

print(result.text)
for line in result.lines:
    print(line["text"], line["bbox"], line["confidence"])

result.save("ket_qua")  # Lưu văn bản UTF-8, JSON và ảnh khung dòng
```

Nạp model một lần để dùng cho nhiều ảnh. `predict()` nhận đường dẫn, bytes
ảnh đã mã hóa hoặc ảnh PIL. Trong Jupyter, gọi `result.show()` để xem kết quả.

Chế độ mặc định `page` xử lý ảnh nhiều dòng. Với ảnh crop sát một dòng nằm ngang:

```python
result = model.predict("crop.png", mode="line")
```

Có thể tùy chỉnh khi nạp: `load_model(threads=2, low_confidence=0.9)`.
Dùng model riêng bằng `load_model(model_dir="/duong/dan/models")`.

## Sử dụng với CLI

```bash
trueocr anh.jpg
trueocr crop.png --mode line
trueocr *.jpg --json --out ket_qua/
```

Xem các tùy chọn bằng `trueocr --help`.

## Tài liệu và phát triển

- [Hướng dẫn sử dụng](https://github.com/pvd0anh/trueocr/blob/main/docs/USAGE.md): cấu hình, model, cấu trúc kết quả và giới hạn.
- [Notebook mẫu](https://github.com/pvd0anh/trueocr/blob/main/examples/quickstart.ipynb).
- [Hướng dẫn phát hành](https://github.com/pvd0anh/trueocr/blob/main/docs/RELEASING.md): tăng phiên bản, kiểm tra và cập nhật lên PyPI.

Cài từ source và chạy toàn bộ test, bao gồm các test cần model:

```bash
pip install -e ".[test]"
TRUEOCR_MODEL_DIR="$PWD/trueocr/models" python -m pytest
```

## Giấy phép

[Apache-2.0](https://github.com/pvd0anh/trueocr/blob/main/LICENSE).
