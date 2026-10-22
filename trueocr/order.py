"""Reading order: line features for the order model and decoding of its pairwise
order and same-block logits.

Modified from Apache-2.0 licensed code.
"""

from __future__ import annotations

import math
import unicodedata

import numpy as np

N_GEOM = 10
N_TEXT = 6
N_FEATS = N_GEOM + N_TEXT

SENTENCE_END = ".!?:;。！？：；．"

# Block classes of the order model's layout head, in output order.
BLOCK_CLASSES = [
    "Text",
    "Section-Header",
    "Page-Header",
    "Page-Footer",
    "List-Group",
    "Table",
    "Caption",
    "Footnote",
    "Table-Of-Contents",
    "Form",
]
# Lines of these blocks are read row by row.
GRID_CLASSES = {BLOCK_CLASSES.index(c) for c in ("Table", "Form", "Table-Of-Contents")}


def display_len(t: str) -> int:
    """Length with East Asian wide characters counted twice."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in t)


def line_features(boxes: np.ndarray, page_wh: tuple[float, float], texts: list[str] | None = None) -> np.ndarray:
    """[N, 4] boxes (x0, y0, x1, y1) -> [N, N_FEATS] float32 features."""
    W, H = page_wh
    b = boxes.astype(np.float32)
    n = len(b)
    if n == 0:
        return np.zeros((0, N_FEATS), np.float32)
    w = np.maximum(b[:, 2] - b[:, 0], 1e-3)
    h = np.maximum(b[:, 3] - b[:, 1], 1e-3)
    med_h = float(np.median(h))
    geom = np.stack(
        [
            b[:, 0] / W,
            b[:, 1] / H,
            b[:, 2] / W,
            b[:, 3] / H,
            (b[:, 0] + b[:, 2]) / (2 * W),
            (b[:, 1] + b[:, 3]) / (2 * H),
            w / W,
            np.log(h / med_h),
            np.log(w / h) / 4.0,
            np.full(n, med_h / H * 20.0),
        ],
        1,
    )
    txt = np.zeros((n, N_TEXT), np.float32)
    if texts is not None:
        for i, t in enumerate(texts):
            t = t.strip()
            if not t:
                continue
            # length in Latin-letter widths: a CJK character is about two wide
            txt[i, 0] = math.log1p(display_len(t)) / 5.0
            txt[i, 1] = sum(c.isdigit() for c in t) / len(t)
            txt[i, 2] = float(t[0] in "•-–▪◦*·→" or (t[:3].rstrip(".)").isdigit() and len(t) > 3))
            txt[i, 3] = float(t.endswith("-"))
            txt[i, 4] = float(t[-1] in SENTENCE_END)
            txt[i, 5] = float(t[0].isupper())
    return np.concatenate([geom, txt], 1).astype(np.float32)


def decode_order(a: np.ndarray, sweeps: int = 4) -> list[int]:
    """Order from a pairwise logit matrix [N, N] (a[i, j] > 0: i before j).

    Start from the row-sum ranking, then local search with insertion moves:
    take each item out and put it back at the position that most increases
    the summed log-probability of the pairwise relations."""
    n = a.shape[0]
    if n <= 1:
        return list(range(n))
    lp = -np.logaddexp(0.0, -a)  # log P(i before j)
    order = np.argsort(-lp.sum(1) + lp.sum(0))
    for _ in range(sweeps):
        improved = False
        for x in order.copy():
            rest = order[order != x]
            before = lp[rest, x]  # x placed after these
            after = lp[x, rest]  # x placed before these
            gains = np.concatenate(([0.0], np.cumsum(before))) + np.concatenate((np.cumsum(after[::-1])[::-1], [0.0]))
            k = int(np.argmax(gains))
            cur = int(np.nonzero(order == x)[0][0])
            if gains[k] > gains[cur] + 1e-6:
                order = np.insert(rest, k, x)
                improved = True
        if not improved:
            break
    return [int(v) for v in order]


def order_top_down(boxes: np.ndarray) -> list[int]:
    """Rows top to bottom (centres within half a line height share a row), left to right."""
    if len(boxes) == 0:
        return []
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    h = np.median(boxes[:, 3] - boxes[:, 1])
    return sorted(range(len(boxes)), key=lambda i: (round(cy[i] / max(h * 0.5, 1e-3)), boxes[i, 0]))


def order_xycut(boxes: np.ndarray) -> list[int]:
    """Recursive XY-cut: split at the widest whitespace gap on either axis.

    A horizontal cut (between rows) puts the upper part first; a vertical cut
    (between columns) puts the left part first.  Without a gap, fall back to
    top-down order."""

    def best_cut(b, axis):
        lo, hi = (b[:, 1], b[:, 3]) if axis == 1 else (b[:, 0], b[:, 2])
        o = np.argsort(lo)
        end = hi[o[0]]
        best_gap, best_k = 0.0, -1
        for k in range(1, len(o)):
            gap = lo[o[k]] - end
            if gap > best_gap:
                best_gap, best_k = gap, k
            end = max(end, hi[o[k]])
        return best_gap, best_k, o

    def rec(ids):
        if len(ids) <= 1:
            return list(ids)
        b = boxes[ids]
        gy, ky, oy = best_cut(b, 1)
        gx, kx, ox = best_cut(b, 0)
        if ky < 0 and kx < 0:
            return [ids[i] for i in order_top_down(b)]
        k, o = (ky, oy) if gy >= gx else (kx, ox)
        return rec([ids[i] for i in o[:k]]) + rec([ids[i] for i in o[k:]])

    return rec(list(range(len(boxes))))


def _cluster(logits: np.ndarray) -> list[list[int]]:
    """Average-linkage clustering of pairwise "same group" logits: merge the two
    groups with the highest mean logit between them while that mean is positive
    (probability above one half).  Unlike connected components, one wrong pair
    cannot join two groups that the other pairs keep apart.

    A merge changes only the merged group's row and column of the mean matrix,
    and each row keeps its best partner, so n lines cost about n^2 instead of n^3."""
    n = logits.shape[0]
    if n == 0:
        return []
    S = logits.astype(np.float64).copy()
    np.fill_diagonal(S, 0.0)
    size = np.ones(n)
    alive = np.ones(n, bool)
    members = [[i] for i in range(n)]
    M = S.copy()  # mean logit between groups; -inf on the diagonal and dead groups
    np.fill_diagonal(M, -np.inf)
    best = M.argmax(1)
    best_v = M[np.arange(n), best]
    for _ in range(n - 1):
        a = int(np.argmax(best_v))
        b = int(best[a])
        if best_v[a] <= 0:
            break
        a, b = min(a, b), max(a, b)
        S[a] += S[b]
        S[:, a] += S[:, b]
        S[a, a] = 0.0
        size[a] += size[b]
        alive[b] = False
        members[a] += members[b]
        row = S[a] / (size[a] * size)
        row[~alive] = -np.inf
        row[a] = -np.inf
        M[a], M[:, a] = row, row
        M[b], M[:, b] = -np.inf, -np.inf
        best_v[b] = -np.inf
        best[a] = int(row.argmax())
        best_v[a] = row[best[a]]
        # other rows: (j, a) changed and (j, b) is gone
        stale = alive & ((best == a) | (best == b))
        stale[a] = False
        for j in np.flatnonzero(stale):
            best[j] = int(M[j].argmax())
            best_v[j] = M[j, best[j]]
        up = alive & (row > best_v)
        up[a] = False
        best[up], best_v[up] = a, row[up]
    return [sorted(members[i]) for i in range(n) if alive[i]]


def decode_blocks_order(a: np.ndarray, same: np.ndarray, boxes: np.ndarray, cls: np.ndarray) -> list[int]:
    """Reading order with the layout heads: blocks first, then lines inside blocks.

    Blocks come from average-linkage clustering of the same-block logits.  Inside
    a block, lines go by XY-cut, or row by row in a table / form / table of
    contents (``cls``: per-line logits over BLOCK_CLASSES).  Blocks go by the
    mean pairwise order logit between them."""
    n = a.shape[0]
    if n <= 1:
        return list(range(n))
    blocks = _cluster(same)
    inner = []
    for b in blocks:
        bx = boxes[b, :4]
        local = order_top_down(bx) if int(cls[b].sum(0).argmax()) in GRID_CLASSES else order_xycut(bx)
        inner.append([b[k] for k in local])
    # Mean order logit between blocks p < q, antisymmetric: G a G^T with G the
    # block-membership matrix scaled by 1 / block size.
    m = len(blocks)
    G = np.zeros((m, n), np.float64)
    for k, b in enumerate(blocks):
        G[k, b] = 1.0 / len(b)
    up = np.triu(G @ a.astype(np.float64) @ G.T, 1)
    ab = (up - up.T).astype(np.float32)
    return [i for k in decode_order(ab) for i in inner[k]]
