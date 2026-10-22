"""Vietnamese text-line recognizer (CTC) on ONNX Runtime, CPU only."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import onnxruntime as ort

from .preprocess import pad_batch, prepare_image, width_batches

MODEL_FILES = "rec.onnx, det.onnx, orient.onnx, order.onnx và meta.json"


def need(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy {path}. Truyền model_dir tới thư mục chứa {MODEL_FILES}.")
    return path


def session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    # No memory arena: with batches of changing shape it kept growing and bought no speed.
    options.enable_cpu_mem_arena = False
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(need(path)), options, providers=["CPUExecutionProvider"])


@dataclass
class Prediction:
    text: str
    confidence: float  # mean CTC peak probability, not calibrated


class Recognizer:
    """Reads cropped text lines: uint8 gray/BGR/BGRA arrays, encoded bytes or paths."""

    def __init__(self, model_dir, threads: int = 1):
        directory = Path(model_dir)
        meta = json.loads(need(directory / "meta.json").read_text(encoding="utf-8"))
        self.name = meta.get("name", directory.name)
        self.chars = meta["charset"]
        self.max_width = meta.get("max_width", 768)
        self.pad_y, self.pad_x = meta.get("pad_y", 0.2), meta.get("pad_x", 0.3)
        self.session = session(directory / "rec.onnx", threads)

    def predict(self, images, batch_size: int = 64) -> list[Prediction]:
        crops = [prepare_image(image, max_width=self.max_width, pad_y=self.pad_y, pad_x=self.pad_x) for image in images]
        results = [None] * len(crops)
        for selected in width_batches([crop.shape[1] for crop in crops], max_batch=batch_size):
            tensor, widths = pad_batch([crops[i] for i in selected])
            ids, conf, lengths = self.session.run(["ids", "conf", "lengths"], {"x": tensor, "widths": widths})
            for j, index in enumerate(selected):
                seq = ids[j, : int(lengths[j])]
                keep = (seq != 0) & np.concatenate(([True], seq[1:] != seq[:-1]))  # greedy CTC, blank = 0
                text = "".join(self.chars[token - 1] for token in seq[keep])
                text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
                confidence = float(conf[j, : int(lengths[j])][keep].mean()) if keep.any() else 0.0
                results[index] = Prediction(text, confidence)
        return results

    def predict_one(self, image) -> Prediction:
        return self.predict([image])[0]
