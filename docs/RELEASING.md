# Release

Chạy từ repo root. Version lấy từ `trueocr.__version__`; tăng version
trước khi build. PyPI không cho overwrite artifact đã release.

## GitHub Actions

[CI](../.github/workflows/ci.yml) chạy toàn bộ test với model đi kèm trên
Python 3.10 và 3.14 khi push lên `main` hoặc mở PR vào `main`.
[Publish to PyPI](../.github/workflows/release.yml) chạy khi push tag `v*`:

1. Checkout tag và kiểm tra tag khớp `trueocr.__version__`.
2. Chạy test, build wheel/sdist, kiểm tra metadata và đủ năm file model.
3. Cài wheel, kiểm tra inference và CLI ngoài source tree.
4. Lưu artifact rồi upload lên PyPI bằng Trusted Publishing, không cần API token.

Push code lên `main` chỉ chạy CI; phát hành cần tag phiên bản.

### Cấu hình PyPI một lần

Repository dùng GitHub environment `pypi`. Tại
[Publishing của dự án TrueOCR](https://pypi.org/manage/project/trueocr/settings/publishing/),
thêm GitHub Trusted Publisher với các giá trị:

| Trường | Giá trị |
|---|---|
| Owner | `pvd0anh` |
| Repository name | `trueocr` |
| Workflow name | `release.yml` |
| Environment name | `pypi` |

Workflow name chỉ là tên file, không phải đường dẫn hoặc tên hiển thị.
Environment phải khớp cả trong workflow và cấu hình PyPI. GitHub environment
`pypi` cho phép tag `v*` và branch `main` (để chạy thủ công).
Xem [hướng dẫn Trusted Publisher của PyPI](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

### Phát hành phiên bản mới

Tăng `trueocr.__version__`, commit và push code trước. Sau đó tạo tag trùng
phiên bản đã commit:

```bash
RELEASE_VERSION=$(python -c 'from trueocr import __version__; print(__version__)')
git tag -a "v$RELEASE_VERSION" -m "TrueOCR $RELEASE_VERSION"
git push origin "v$RELEASE_VERSION"
```

Theo dõi job tại [Actions](https://github.com/pvd0anh/trueocr/actions).
Nếu test, build hoặc kiểm tra gói thất bại, job publish không chạy.

### Phát hành tag đã có

Với tag tạo trước khi thêm workflow, vào **Actions → Publish to PyPI →
Run workflow**, chọn branch `main` và điền tag cần phát hành. Hoặc dùng CLI:

```bash
gh workflow run release.yml --ref main -f tag=v0.1.1
```

Workflow checkout đúng tag đã chọn, không thay đổi tag hoặc tự tăng version.
Chỉ chạy cho phiên bản chưa phát hành trên PyPI. Nếu lỗi Trusted Publishing,
kiểm tra bốn trường cấu hình ở trên rồi chạy lại job bị lỗi.

## Phát hành từ máy local

Các bước dưới đây dùng khi cần build hoặc upload thủ công bằng token PyPI.

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
