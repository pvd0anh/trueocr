"""trueocr ẢNH...: in text (mặc định) hoặc JSON của từng ảnh."""

import argparse
import json
import sys

from .ocr import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(prog="trueocr", description="OCR tiếng Việt trên CPU.")
    parser.add_argument("images", nargs="+", help="đường dẫn ảnh")
    parser.add_argument("--model-dir", help="thư mục chứa rec.onnx, det.onnx, orient.onnx, order.onnx và meta.json")
    parser.add_argument("--mode", choices=["page", "line"], default="page", help="page: ảnh nhiều dòng; line: crop một dòng")
    parser.add_argument("--json", action="store_true", help="in kết quả JSON, mỗi ảnh một dòng")
    parser.add_argument("--out", help="lưu .txt, .json và ảnh khung dòng vào thư mục này")
    parser.add_argument("--threads", type=int, help="số CPU thread")
    args = parser.parse_args(argv)

    options = {"mode": args.mode}
    if args.model_dir:
        options["model_dir"] = args.model_dir
    if args.threads:
        options["threads"] = args.threads
    if args.out:
        options.update(save_outputs=True, output_dir=args.out)
    model = load_model(**options)

    failed = 0
    for image in args.images:
        try:
            result = model.predict(image)
        except (OSError, ValueError) as error:
            print(f"{image}: {error}", file=sys.stderr)
            failed = 1
            continue
        if args.json:
            print(json.dumps(result.data, ensure_ascii=False))
        else:
            if len(args.images) > 1:
                print(f"==> {image} <==")
            print(result.text)
    return failed
