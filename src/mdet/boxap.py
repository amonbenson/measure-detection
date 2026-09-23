"""Box average precision of measures (COCO style: 101-point interpolation, IoU 0.50 and 0.50:0.95).

Reported for comparison with the literature only (guideline 7.1). Boxes are axis-aligned in page
fractions. Note: the ground-truth vertical extent is inconsistent (docs/CONVENTIONS.md), which
lowers IoU for rows annotated with lyrics.
"""

from __future__ import annotations

import numpy as np

from .data import Row
from .evaluate import PredSystem


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a (n, 4) and b (m, 4), format x0, y0, x1, y1."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-12)


def average_precision(pages: list[tuple[list[PredSystem], list[Row]]], thresholds=np.arange(0.5, 0.96, 0.05)) -> dict:
    """pages: (predicted systems, ground-truth rows) per page."""
    dets, gts = [], []
    for k, (pred, rows) in enumerate(pages):
        for sy in pred:
            for m in sy.measures:
                xs, ys = [p[0] for p in m.quad], [p[1] for p in m.quad]
                dets.append((k, min(m.confidence, m.label_confidence), [min(xs), min(ys), max(xs), max(ys)]))
        gts.append(np.array([m.box for r in rows for m in r.measures], np.float64).reshape(-1, 4))
    n_gt = sum(len(g) for g in gts)
    dets.sort(key=lambda d: -d[1])
    aps = []
    for t in thresholds:
        used = [np.zeros(len(g), bool) for g in gts]
        tp = np.zeros(len(dets))
        for i, (k, _, box) in enumerate(dets):
            g = gts[k]
            if not len(g):
                continue
            iou = _iou(np.array([box]), g)[0]
            iou[used[k]] = -1
            j = int(np.argmax(iou))
            if iou[j] >= t:
                used[k][j] = True
                tp[i] = 1
        ctp = np.cumsum(tp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.arange(1, len(dets) + 1)
        prec = np.maximum.accumulate(prec[::-1])[::-1] if len(prec) else prec
        ap = 0.0
        for r in np.linspace(0, 1, 101):
            ok = np.nonzero(rec >= r)[0]
            ap += prec[ok[0]] if len(ok) else 0.0
        aps.append(ap / 101)
    return {"box_map50": float(aps[0]), "box_map50_95": float(np.mean(aps))}
