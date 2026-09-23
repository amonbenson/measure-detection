"""Evaluate RF-DETR box predictions with the pipeline's metrics (run in the main environment).

Boxes are in normalized-page pixels (s = 8). Systems: system-class boxes (score >= 0.5, NMS). Measures:
measure-class boxes (score >= tau, NMS), each assigned to the system with the largest vertical overlap,
sorted by x; the boundary between neighbours is the midpoint of their facing edges.
Usage (repository root): uv run python baselines/rfdetr/evaluate.py --pred runs/rfdetr/pred_test.json --split test [--tau 0.5]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from mdet.boxap import average_precision
from mdet.data import load_document, read_split
from mdet.evaluate import PredMeasure, PredSystem, evaluate_page, summarize
from mdet.geometry import PageGeometry

# RF-DETR's two-class head outputs 0 = system, 1 = measure (the COCO supercategory is dropped).
SYSTEM, MEASURE = 0, 1


def nms(boxes: list[dict], iou_thr: float) -> list[dict]:
    keep: list[dict] = []
    for b in sorted(boxes, key=lambda d: -d["score"]):
        x0, y0, x1, y1 = b["box"]
        ok = True
        for k in keep:
            a0, b0, a1, b1 = k["box"]
            iw, ih = max(0, min(x1, a1) - max(x0, a0)), max(0, min(y1, b1) - max(y0, b0))
            inter = iw * ih
            if inter / max((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter, 1e-9) > iou_thr:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def to_systems(geo: PageGeometry, page: int, dets: list[dict], tau: float) -> list[PredSystem]:
    sys_boxes = nms([d for d in dets if d["class_id"] == SYSTEM and d["score"] >= 0.5], 0.5)
    mea_boxes = nms([d for d in dets if d["class_id"] == MEASURE and d["score"] >= tau], 0.5)
    groups: list[list[dict]] = [[] for _ in sys_boxes]
    for m in mea_boxes:
        ov = [max(0, min(m["box"][3], s["box"][3]) - max(m["box"][1], s["box"][1])) for s in sys_boxes]
        if ov and max(ov) > 0:
            groups[int(np.argmax(ov))].append(m)
    out = []
    for s, ms in zip(sys_boxes, groups):
        if not ms:
            continue
        ms.sort(key=lambda d: d["box"][0])
        xs = [ms[0]["box"][0]] + [(a["box"][2] + b["box"][0]) / 2 for a, b in zip(ms, ms[1:])] + [ms[-1]["box"][2]]
        y0, y1 = s["box"][1], s["box"][3]
        yc = (y0 + y1) / 2
        (px, _) = geo.norm_to_page(np.array(xs), np.full(len(xs), yc))
        (_, (py0, py1)) = geo.norm_to_page(np.array([xs[0], xs[0]]), np.array([y0, y1]))
        measures = []
        for k, m in enumerate(ms):
            quad = [tuple(map(float, geo.norm_to_page(x, y))) for x, y in ((xs[k], y0), (xs[k + 1], y0), (xs[k + 1], y1), (xs[k], y1))]
            measures.append(PredMeasure(x0=float(px[k]), x1=float(px[k + 1]), confidence=m["score"], quad=quad))
        pxs = geo.norm_to_page(np.array([s["box"][0], s["box"][2]]), np.array([yc, yc]))[0]
        out.append(PredSystem(page=page, x0=float(pxs[0]), x1=float(pxs[1]), y0=float(py0), y1=float(py1), measures=measures))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--tau", type=float, default=0.5)
    args = ap.parse_args()
    pred = json.loads(open(args.pred, encoding="utf-8").read())
    results, pages = [], []
    for d in read_split(args.split):
        doc = load_document(d)
        for p in range(doc.num_pages):
            geo = PageGeometry.load(d, p)
            systems = to_systems(geo, p, pred.get(f"{d}_{p:03d}.png", []), args.tau)
            rows = doc.rows_on_page(p)
            results.append(evaluate_page(d, p, geo.s_original, systems, rows))
            pages.append((systems, rows))
    s = summarize(results)
    s.update(average_precision(pages))
    for k, v in s.items():
        print(f"{k:32s} {v:.4f}" if isinstance(v, float) else f"{k:32s} {v}")


if __name__ == "__main__":
    main()
