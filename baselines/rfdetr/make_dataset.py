"""COCO dataset for the RF-DETR baseline from the prepared pages (run in the main environment).

Images: normalized pages at s = 8 (dataset/derived/pages, halved). Classes: 1 = system, 2 = measure.
Boxes: enclosing boxes of the ground-truth boxes mapped into the normalized frame.
Usage (repository root): uv run python baselines/rfdetr/make_dataset.py [--out runs/rfdetr/data]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mdet.data import RUNS, load_document, read_split
from mdet.geometry import PageGeometry


def box_to_norm(geo: PageGeometry, x0: float, y0: float, x1: float, y1: float) -> list[float]:
    xs, ys = geo.page_to_norm([x0, x1, x1, x0], [y0, y0, y1, y1])
    return [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))]


def build(split: str, out: Path) -> None:
    d_out = out / {"train": "train", "val": "valid", "test": "test"}[split]
    d_out.mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    for doc_id in read_split(split):
        doc = load_document(doc_id)
        for p in range(doc.num_pages):
            geo = PageGeometry.load(doc_id, p)
            img = geo.norm8()
            name = f"{doc_id}_{p:03d}.png"
            cv2.imwrite(str(d_out / name), img)
            iid = len(images)
            images.append({"id": iid, "file_name": name, "width": int(img.shape[1]), "height": int(img.shape[0])})
            for r in doc.rows_on_page(p):
                for cat, boxes in ((1, [(r.x0, r.y0, r.x1, r.y1)]), (2, [m.box for m in r.measures])):
                    for b in boxes:
                        bb = box_to_norm(geo, *b)
                        anns.append({"id": len(anns), "image_id": iid, "category_id": cat, "bbox": bb, "area": bb[2] * bb[3], "iscrowd": 0})
    coco = {
        "images": images,
        "annotations": anns,
        "categories": [{"id": 0, "name": "notation", "supercategory": "none"}, {"id": 1, "name": "system", "supercategory": "notation"}, {"id": 2, "name": "measure", "supercategory": "notation"}],
    }
    (d_out / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")
    print(split, len(images), "images", len(anns), "boxes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RUNS / "rfdetr" / "data"))
    args = ap.parse_args()
    for s in ("train", "val", "test"):
        build(s, Path(args.out))


if __name__ == "__main__":
    main()
