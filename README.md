# TrueOCR

OCR tiếng Việt tốc độ cao trên CPU. Đọc ảnh nhiều dòng hoặc crop một
dòng; trả về text, bbox và confidence. Model khoảng 8 MB kèm trong package,
chạy offline, không cần GPU hay PyTorch.

## Tốc độ CPU

Xeon Platinum 8358P, **1 thread, batch 1**, cùng 128 crop × 3 lượt/model @TrueOCR 0.1.0.

![p50 latency của 8 recognizer trên CPU; thanh ngắn hơn là nhanh hơn.](https://raw.githubusercontent.com/pvd0anh/trueocr/main/docs/assets/cpu-latency.png)

| Model | p50, ms/crop ↓ | p95, ms/crop ↓ | p50 / TrueOCR |
|---|---:|---:|---:|
| TrueOCR 0.1.0 | 1.54 | 2.35 | 1.0× |
| Tesseract fast | 5.89 | 15.29 | 3.8× |
| PaddleOCR v6 tiny | 10.76 | 12.29 | 7.0× |
| Tesseract best | 20.24 | 49.06 | 13.2× |
| PaddleOCR v5 Latin mobile | 29.43 | 30.79 | 19.1× |
| EasyOCR latin_g2 | 30.03 | 68.90 | 19.5× |
| VietOCR vgg_seq2seq | 71.77 | 98.01 | 46.7× |
| VietOCR vgg_transformer | 137.85 | 231.45 | 89.7× |

Đo latency của crop recognition, gồm preprocessing, inference và text decoding.
Không gồm model loading, image I/O hay text detection cho cả trang.
[Benchmark setup, baseline config và số liệu từng request](https://github.com/pvd0anh/trueocr/blob/main/docs/BENCHMARKS.md).

## Quick Start

Python ≥3.10.

```bash
pip install trueocr
```

```python
from trueocr import load_model

model = load_model()  # Load một lần, dùng lại cho nhiều ảnh
result = model.predict("anh.jpg")
print(result.text)
```

Mặc định đọc ảnh nhiều dòng. Với crop một dòng:
`model.predict("crop.png", mode="line")`. Lấy bbox và confidence qua
`result.lines`; lưu text, JSON và ảnh bbox bằng `result.save("output")`.

CLI: `trueocr anh.jpg --out output/`.

## Chất lượng và giới hạn

Model dùng cho chữ in tiếng Việt. Handwriting, scene text và font lạ còn yếu;
crop quá rộng hoặc chứa nhiều dòng dễ sai. [Kết quả theo dataset](https://github.com/pvd0anh/trueocr/blob/main/docs/BENCHMARKS.md#recognition-accuracy).

[API và config](https://github.com/pvd0anh/trueocr/blob/main/docs/USAGE.md) · [Notebook](https://github.com/pvd0anh/trueocr/blob/main/examples/quickstart.ipynb) ·
[Release](https://github.com/pvd0anh/trueocr/blob/main/docs/RELEASING.md) · [Apache-2.0](https://github.com/pvd0anh/trueocr/blob/main/LICENSE)
