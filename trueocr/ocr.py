"""Nạp model một lần, predict ảnh và lưu/hiển thị kết quả OCR."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
import html
import io
import json
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .page import VERTICAL, PageEngine
from .recognizer import Recognizer


@dataclass(frozen=True)
class OCRConfig:
    model_dir: Path | None = None  # None: thư mục models/ trong package
    output_dir: Path = Path("trueocr_outputs")
    mode: str = "page"  # "page": ảnh nhiều dòng; "line": crop một dòng nằm ngang.
    threads: int = min(4, os.cpu_count() or 1)
    batch_size: int = 32
    auto_rotate: bool = True
    low_confidence: float = 0.80  # Chỉ đánh dấu; không bỏ text.
    save_outputs: bool = False  # True: tự lưu .txt, .json và ảnh khung dòng sau mỗi predict.

    def __post_init__(self):
        if self.mode not in {"page", "line"}:
            raise ValueError('mode phải là "page" hoặc "line".')
        if self.threads < 1 or self.batch_size < 1:
            raise ValueError("threads và batch_size phải lớn hơn 0.")
        if not 0 <= self.low_confidence <= 1:
            raise ValueError("low_confidence phải nằm trong khoảng [0, 1].")


def _read_image(source):
    """Nhận path, bytes hoặc PIL; áp EXIF và ghép nền trắng cho ảnh alpha."""
    if isinstance(source, Image.Image):
        image = ImageOps.exif_transpose(source).copy()
    else:
        if isinstance(source, (str, Path)):
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
            source = path.read_bytes()
        with Image.open(io.BytesIO(bytes(source))) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
    rgba = image.convert("RGBA")
    return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")


def _annotate(image, lines):
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for line in lines:
        box = line["bbox"]
        color = "#e07020" if line["low_confidence"] else "#16a34a"
        draw.rectangle(box, outline=color, width=max(1, image.width // 600))
        draw.text((box[0], max(0, box[1] - 12)), str(line["index"]), fill=color)
    return preview


@dataclass
class OCRResult:
    data: dict
    image: Image.Image = field(repr=False)
    output_dir: Path = field(repr=False)
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def text(self):
        return self.data["text"]

    @property
    def lines(self):
        return self.data["lines"]

    def save(self, output_dir=None):
        """Lưu text UTF-8, JSON và ảnh khung dòng; trả các đường dẫn đã ghi."""
        folder = Path(self.output_dir if output_dir is None else output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^\w.-]+", "_", Path(self.data["input_name"]).stem, flags=re.UNICODE)[:100] or "image"
        prefix = folder / f"{stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        files = {
            "text": Path(f"{prefix}.txt"),
            "json": Path(f"{prefix}.json"),
            "preview": Path(f"{prefix}_boxes.png"),
        }
        files["text"].write_text(self.text, encoding="utf-8")
        files["json"].write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        _annotate(self.image, self.lines).save(files["preview"])
        self.files = files
        return files

    def show(self):
        """Hiển thị toàn bộ text, khung dòng, confidence và file kết quả (Jupyter)."""
        from IPython.display import FileLink, HTML, display

        print(f"Ảnh: {self.data['input_name']} | {len(self.lines)} dòng | {self.data['elapsed_ms']:.1f} ms")
        if not self.text.strip():
            print('Chưa đọc được chữ. Nếu ảnh là crop một dòng, thử model.predict(ảnh, mode="line").')
        print("\n===== TEXT OCR =====\n")
        print(self.text)
        preview = _annotate(self.image, self.lines)
        preview.thumbnail((1200, 1200))
        display(preview)
        rows = "".join(
            f"<tr><td>{line['index']}</td><td>{html.escape(line['text'])}</td>"
            f"<td>{line['confidence']:.4f}</td><td>{line['bbox']}</td>"
            f"<td>{'Xem lại' if line['low_confidence'] else ''}</td></tr>"
            for line in self.lines
        )
        display(HTML("<table><tr><th>Dòng</th><th>Text</th><th>Confidence</th><th>Bbox</th><th></th></tr>" + rows + "</table>"))
        if self.files:
            print("\nĐã lưu (UTF-8):")
        for kind, path in self.files.items():
            print(f"{kind}: {path}")
            display(FileLink(os.path.relpath(path, Path.cwd())))


class TrueOCR:
    def __init__(self, config: OCRConfig = OCRConfig()):
        self.config = config
        default_dir = Path(__file__).parent / "models"
        self.model_dir = Path(config.model_dir or default_dir).expanduser().resolve()
        self.recognizer = Recognizer(self.model_dir, threads=config.threads)
        self._page_engine = None
        if config.mode == "page":
            self._get_page_engine()

    def _get_page_engine(self):
        if self._page_engine is None:
            self._page_engine = PageEngine(self.model_dir, threads=self.config.threads)
        return self._page_engine

    def predict(self, source, *, mode=None, save=None):
        """OCR một path/bytes/PIL; trả OCRResult với .text, .lines và .files."""
        mode = self.config.mode if mode is None else mode
        if mode not in {"page", "line"}:
            raise ValueError('mode phải là "page" hoặc "line".')
        image = _read_image(source)
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        engine = self._get_page_engine() if mode == "page" else None
        started = time.perf_counter()
        rotation, skew = 0, 0.0
        if mode == "page":
            gray, quarter_turns, skew = engine.upright(gray, turn=self.config.auto_rotate)
            rotation = -90 if quarter_turns == VERTICAL else 90 * quarter_turns
            image = Image.fromarray(gray).convert("RGB")
            crops, boxes = [], []
            h, w = gray.shape
            for box in engine.detect(gray):
                x0, y0 = np.floor(box[:2]).astype(int)
                x1, y1 = np.ceil(box[2:4]).astype(int)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1, y1])
                    # Recognizer tự thêm lề của model, tránh thêm lề hai lần.
                    crops.append(gray[y0:y1, x0:x1])
        else:
            h, w = gray.shape
            boxes, crops = [[0, 0, w, h]], [gray]
        predictions = self.recognizer.predict(crops, batch_size=self.config.batch_size)
        if mode == "page":
            order = engine.reading_order(np.asarray(boxes, np.float32).reshape(-1, 4), [p.text for p in predictions], (w, h))
        else:
            order = range(len(boxes))
        lines = []
        for i in order:
            pred = predictions[i]
            lines.append({
                "index": len(lines) + 1,
                "text": pred.text,
                "confidence": pred.confidence,
                "bbox": [int(v) for v in boxes[i]],
                "low_confidence": pred.confidence < self.config.low_confidence,
            })
        result = OCRResult(
            data={
                "input_name": Path(source).name if isinstance(source, (str, Path)) else "image",
                "created_at": datetime.now().astimezone().isoformat(),
                "model": self.recognizer.name,
                "model_dir": str(self.model_dir),
                "mode": mode,
                "text": "\n".join(line["text"] for line in lines),
                "lines": lines,
                "image_size": list(image.size),
                "bbox_coordinate_system": "processed_image" if mode == "page" else "input_image_after_exif",
                "rotation_clockwise_degrees": int(rotation),
                "skew_degrees": float(skew),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "low_confidence_threshold": self.config.low_confidence,
            },
            image=image,
            output_dir=self.config.output_dir,
        )
        if self.config.save_outputs if save is None else save:
            result.save()
        return result


def load_model(config: OCRConfig | None = None, **options):
    """Nạp model một lần; dùng lại đối tượng cho các lần predict tiếp theo.

    load_model(model_dir="/duong/dan/models", mode="line") ghi đè từng trường của OCRConfig.
    """
    return TrueOCR(replace(config or OCRConfig(), **options))
