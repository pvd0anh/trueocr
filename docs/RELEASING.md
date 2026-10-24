# Release

Chạy từ repo root. Version lấy từ `trueocr.__version__`; tăng version
trước khi build. PyPI không cho overwrite artifact đã release.

## Tests

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip build twine
python -m pip install -e ".[test]"
TRUEOCR_MODEL_DIR="$PWD/trueocr/models" python -m pytest
```

Đặt `TRUEOCR_MODEL_DIR` để chạy cả test inference; thiếu biến này, các test
cần weights sẽ bị skip. Khi đổi model hoặc preprocessing, cập nhật
[benchmark](BENCHMARKS.md) trước khi công bố số liệu cho version mới.

## Build

```bash
RELEASE_VERSION=$(python -c 'from trueocr import __version__; print(__version__)')
RELEASE_DIR="dist/pypi-$RELEASE_VERSION"
python -m build --outdir "$RELEASE_DIR"
python -m twine check "$RELEASE_DIR"/*
unzip -l "$RELEASE_DIR"/*.whl
tar -tzf "$RELEASE_DIR"/*.tar.gz
```

Cả wheel và sdist phải có `rec.onnx`, `det.onnx`, `orient.onnx`, `order.onnx`,
`meta.json` trong `trueocr/models/`. Weights được theo dõi trong Git;
`pyproject.toml` khai báo chúng qua `tool.setuptools.package-data`.

## Upload

```bash
python -m twine upload --username __token__ "$RELEASE_DIR"/*
```

Nhập token PyPI khi được hỏi; không lưu token vào repo. Nếu artifact đã tồn
tại, tăng version và build lại. Nếu upload dở, kiểm tra PyPI rồi chỉ upload
artifact còn thiếu của cùng lần build.

## Verify release

Cài ngoài source tree để tránh import nhầm code local:

```bash
RELEASE_CHECK_DIR=$(mktemp -d)
python -m venv "$RELEASE_CHECK_DIR/venv"
"$RELEASE_CHECK_DIR/venv/bin/python" -m pip install \
  --index-url https://pypi.org/simple/ "trueocr==$RELEASE_VERSION"
(
  cd "$RELEASE_CHECK_DIR"
  ./venv/bin/python -c 'import trueocr; print(trueocr.__version__); trueocr.load_model()'
)
```

Chốt commit và tag `v<version>` tương ứng với artifact. Quy trình chuẩn:
[PyPA](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
