"""Page geometry on ONNX Runtime: deskew, quarter-turn orientation, text-line
detection and reading order.

Modified from Apache-2.0 licensed code.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .order import decode_blocks_order, line_features
from .recognizer import need, session

DESKEW_MAX = 45.0  # degrees of tilt searched (quarter turns cover the rest)
DESKEW_MIN = 0.3  # smaller tilts are left alone (the search step is 0.1)
ORIENT_SIDE = 768  # orient.onnx input: thumbnail long side, padded square
# A turn is applied only above this probability: real pages are almost all
# upright, and a wrong turn loses the page.
ORIENT_MIN = 0.95
VERTICAL = 4  # orient.onnx class: the page is set in vertical columns
# Small text: when the median detected line is under ADAPTIVE_MIN_H detector
# pixels, detect again at a scale that makes it about ADAPTIVE_TARGET_H.
ADAPTIVE_MIN_H = 9.0
ADAPTIVE_TARGET_H = 12.0
ADAPTIVE_MAX_SIDE = 2048
OUT_STRIDE = 2  # det.onnx output is at half the input resolution
INPUT_MULTIPLE = 32  # det.onnx input height and width divide by this
CORE_X_SHRINK = 0.15  # a line's core is shorter than the line by this x its height, each end


def _ink_thumb(gray: np.ndarray, side: int) -> np.ndarray:
    """uint8 ink map (255 = ink) of the image shrunk to ``side`` pixels."""
    h, w = gray.shape
    s = side / max(h, w)
    small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return 255 - small


def _row_sharpness(ink: np.ndarray, a: float) -> float:
    """How sharply the horizontal ink profile changes row to row, the image
    turned by ``a`` degrees: highest when the text rows are level."""
    size = (ink.shape[1], ink.shape[0])
    r = cv2.warpAffine(ink, cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), a, 1.0), size)
    p = r.sum(1, dtype=np.int64).astype(np.float64)
    return float(((p[1:] - p[:-1]) ** 2).sum())


def estimate_skew(gray: np.ndarray, side: int = 800, pool=None) -> float:
    """Degrees (counter-clockwise) that straighten the text rows: the angle whose
    horizontal ink profile changes most sharply row to row.  Searched over
    +-DESKEW_MAX in 2 degree steps on a half-size thumbnail; the two best angles
    and 0 are then refined in 0.5 and 0.1 degree steps on the full thumbnail.
    pool: a thread pool that scores the angles in parallel (same result)."""
    run = pool.map if pool is not None else map
    fine = _ink_thumb(gray, side)
    if fine.mean() < 0.003 * 255:  # a blank image has no rows to straighten
        return 0.0
    coarse = _ink_thumb(gray, side // 2)
    grid = np.arange(-DESKEW_MAX, DESKEW_MAX + 1e-6, 2.0)
    scores = list(run(lambda a: _row_sharpness(coarse, a), grid))
    starts = {0.0, *(float(grid[i]) for i in np.argsort(scores)[-2:])}
    near = [a for a0 in starts for a in np.arange(a0 - 2.0, a0 + 2.0 + 1e-6, 0.5)]
    # The 0.5 degree ranges of nearby starts overlap: each angle is scored once.
    seen = dict.fromkeys(near)
    seen.update(zip(seen, run(lambda a: _row_sharpness(fine, a), list(seen))))
    a1 = max(near, key=seen.__getitem__)
    last = np.arange(a1 - 0.5, a1 + 0.5 + 1e-6, 0.1)
    vals = list(run(lambda a: _row_sharpness(fine, a), last))
    return float(max(zip(last, vals), key=lambda av: av[1])[0])


def rotate_page(gray: np.ndarray, degrees: float) -> np.ndarray:
    """The image rotated by ``degrees`` counter-clockwise about its centre (same size)."""
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def orient_input(gray: np.ndarray, side: int = ORIENT_SIDE) -> np.ndarray:
    """[side, side] float32 ink map (1 = ink): long side scaled to ``side``, padded."""
    h, w = gray.shape
    s = side / max(h, w)
    small = cv2.resize(gray, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((side, side), np.float32)
    out[: small.shape[0], : small.shape[1]] = 1.0 - small.astype(np.float32) / 255.0
    return out


def det_input(gray: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """The image resized so its long side is ``long_side`` and padded to a multiple
    of 32 with white: (float32 [H, W] ink map, scale)."""
    h, w = gray.shape
    scale = long_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        gray = cv2.resize(gray, (nw, nh), interpolation=interp)
    ph = math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE
    pw = math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE
    out = np.zeros((ph, pw), np.float32)  # ink map: 0 = white
    out[:nh, :nw] = 1.0 - gray.astype(np.float32) / 255.0
    return out, scale


def decode_lines(out: np.ndarray, thresh: float = 0.5, min_area: int = 2) -> np.ndarray:
    """Line boxes from det.onnx's output for one image: [C, h, w] with the core
    logit, top distance and bottom distance first.  Returns [N, 5] boxes in
    detector input pixels: x0, y0, x1, y1, score."""
    prob = 1.0 / (1.0 + np.exp(-out[0]))
    mask = (prob > thresh).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return np.zeros((0, 5), np.float32)
    s = OUT_STRIDE
    flat = lab.ravel()
    keep = flat > 0
    idx = flat[keep]
    ys = (np.indices(lab.shape)[0].ravel()[keep].astype(np.float32) + 0.5) * s
    top = ys - out[1].ravel()[keep]
    bot = ys + out[2].ravel()[keep]
    sc = prob.ravel()[keep]
    cnt = np.maximum(np.bincount(idx, minlength=n).astype(np.float32), 1)
    # The mean is a robust enough summary here and far cheaper than a median.
    y0 = np.bincount(idx, weights=top, minlength=n) / cnt
    y1 = np.bincount(idx, weights=bot, minlength=n) / cnt
    score = np.bincount(idx, weights=sc, minlength=n) / cnt
    boxes = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        lh = max(1.0, y1[i] - y0[i])
        ex = min(CORE_X_SHRINK * lh, 0.5 * lh)
        x0 = stats[i, cv2.CC_STAT_LEFT] * s - ex
        x1 = (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) * s + ex
        boxes.append((x0, y0[i], x1, y1[i], score[i]))
    return np.asarray(boxes, np.float32).reshape(-1, 5)


class PageEngine:
    """det.onnx, orient.onnx and order.onnx of a model folder."""

    def __init__(self, model_dir, threads: int = 4):
        model_dir = Path(model_dir)
        meta = json.loads(need(model_dir / "meta.json").read_text(encoding="utf-8"))
        self.long_side = meta["det_long_side"]
        self.det = session(model_dir / "det.onnx", threads)
        self.orient = session(model_dir / "orient.onnx", threads)
        self.order = session(model_dir / "order.onnx", threads)
        # The skew search scores its angles on a small pool (OpenCV releases the GIL).
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None

    def orientation(self, gray: np.ndarray) -> int:
        """Quarter turns (counter-clockwise) the content is turned by: 0 when no
        turn is more likely than ORIENT_MIN; VERTICAL for vertical columns."""
        p = self.orient.run(None, {"x": orient_input(gray)[None, None]})[0][0]
        k = int(p.argmax())
        return k if p[k] > ORIENT_MIN else 0

    def upright(self, gray: np.ndarray, turn: bool = True) -> tuple[np.ndarray, int, float]:
        """The image turned upright (orient.onnx, when ``turn``) and straightened:
        (image, quarter turns undone, degrees of skew undone)."""
        a = estimate_skew(gray, pool=self.pool)
        # The orientation model reads the straightened image (a tilted page could read as turned).
        k = self.orientation(rotate_page(gray, a) if abs(a) >= DESKEW_MIN else gray) if turn else 0
        if k == VERTICAL:
            # A quarter turn counter-clockwise: columns become lines, the first (right) column on top.
            gray = np.ascontiguousarray(np.rot90(gray, 1))
        elif k:
            gray = np.ascontiguousarray(np.rot90(gray, -k))
        if k in (1, 3, VERTICAL):
            # After a quarter turn the rows were columns: measure the angle again.
            a = estimate_skew(gray, pool=self.pool)
        if abs(a) < DESKEW_MIN:
            return gray, k, 0.0
        return rotate_page(gray, a), k, a

    def _detect_at(self, gray: np.ndarray, long_side: int):
        x, s = det_input(gray, long_side)
        out = self.det.run(None, {"x": x[None, None]})[0][0]
        return decode_lines(out), s

    def detect(self, gray: np.ndarray) -> np.ndarray:
        """Text-line boxes [N, 5] (x0, y0, x1, y1, score) in image pixels."""
        boxes, s = self._detect_at(gray, self.long_side)
        if len(boxes) >= 8:
            med_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
            if med_h < ADAPTIVE_MIN_H:
                ls = int(min(ADAPTIVE_MAX_SIDE, self.long_side * ADAPTIVE_TARGET_H / max(med_h, 1.0))) // 32 * 32
                if ls > self.long_side:
                    boxes, s = self._detect_at(gray, ls)
        if len(boxes):
            boxes[:, :4] /= s
            h, w = gray.shape
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        return boxes

    def reading_order(self, boxes: np.ndarray, texts: list[str], wh: tuple[int, int]) -> list[int]:
        """Indices of ``boxes`` [N, 4] in reading order."""
        if len(boxes) < 2:
            return list(range(len(boxes)))
        feats = line_features(boxes, wh, texts)[None]
        a, cls, same = self.order.run(["a", "cls", "same"], {"feats": feats, "mask": np.ones((1, len(boxes)), bool)})
        return decode_blocks_order(a[0], same[0], boxes, cls[0])
