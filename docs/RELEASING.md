# Phát hành lên PyPI

Đã phát hành thành công `trueocr` lên PyPI ngày 23/10/2026; phiên bản trong
source tại thời điểm ghi chú là `0.1.0`.

Các bước dưới đây chạy từ thư mục gốc dự án, trong cùng một terminal.

### 1. Tăng phiên bản và kiểm tra code

Sửa `__version__` trong `trueocr/__init__.py`, ví dụ `0.1.0` → `0.1.1`.
`pyproject.toml` tự lấy phiên bản từ biến này. Sửa lỗi nhỏ tăng số cuối;
thêm tính năng tăng số giữa, ví dụ `0.2.0`.

PyPI không cho ghi đè file đã phát hành, kể cả sau khi xóa file đó.
Nếu cần phát hành lại bản đã upload, hãy tăng phiên bản.
Xem [quy định về tên file của PyPI](https://pypi.org/help/#file-name-reuse).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build twine
python -m pip install -e ".[test]"
TRUEOCR_MODEL_DIR="$PWD/trueocr/models" python -m pytest
```

Chỉ tiếp tục khi test thành công. Đặt `TRUEOCR_MODEL_DIR` như trên để các test
cần model được chạy, tránh bỏ qua chúng.

### 2. Build và kiểm tra model

Lấy phiên bản từ source, build vào thư mục riêng để tránh upload nhầm bản cũ:

```bash
RELEASE_VERSION=$(python -c 'from trueocr import __version__; print(__version__)')
RELEASE_DIR="dist/pypi-$RELEASE_VERSION"
python -m build --outdir "$RELEASE_DIR"
python -m twine check "$RELEASE_DIR"/*

unzip -l "$RELEASE_DIR"/*.whl | grep 'models/'
tar -tzf "$RELEASE_DIR"/*.tar.gz | grep 'models/'
```

Twine phải báo `PASSED`. Cả wheel (`.whl`) và source archive (`.tar.gz`) phải
có đủ `det.onnx`, `rec.onnx`, `orient.onnx`, `order.onnx`, `meta.json`.
Model được theo dõi trong Git và đóng gói bằng cấu hình
`[tool.setuptools.package-data]` trong `pyproject.toml`. Khi thay model,
nhớ commit cả file model tương ứng.

### 3. Upload bản mới

```bash
python -m twine upload --username __token__ "$RELEASE_DIR"/*
```

Khi được hỏi mật khẩu/API token, nhập token PyPI bắt đầu bằng `pypi-`.
Có thể tạo token chỉ có quyền với dự án `trueocr` tại
[PyPI Account settings](https://pypi.org/manage/account/#api-tokens).
Không ghi token vào source hoặc commit lên Git.

Nếu báo `403`, kiểm tra token và quyền với dự án. Nếu báo file đã tồn tại,
tăng phiên bản rồi build lại. Nếu upload bị gián đoạn, kiểm tra file nào
đã lên PyPI và chỉ upload file còn thiếu của cùng lần build.

### 4. Kiểm tra gói đã phát hành

Cài trong môi trường mới, bên ngoài source để kiểm tra đúng gói từ PyPI:

```bash
RELEASE_CHECK_DIR=$(mktemp -d)
python -m venv "$RELEASE_CHECK_DIR/venv"
"$RELEASE_CHECK_DIR/venv/bin/python" -m pip install --index-url https://pypi.org/simple/ "trueocr==$RELEASE_VERSION"
(
  cd "$RELEASE_CHECK_DIR"
  ./venv/bin/python -c 'import trueocr; print(trueocr.__version__); trueocr.load_model(); print("Nạp model thành công")'
)
```

Người dùng đang cài bản cũ có thể cập nhật bằng `pip install --upgrade trueocr`.
Commit code, phiên bản và model đã phát hành lên Git; có thể đánh tag
`v<phiên bản>` để dễ tìm lại source của từng bản.

Tham khảo [quy trình build và upload của PyPA](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
