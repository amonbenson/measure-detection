"""RF-DETR predictions for a COCO split folder, written as JSON boxes in normalized-page pixels.

Run inside this folder's environment:
  uv run python predict.py --model ../../runs/rfdetr/model/checkpoint_best_total.pth --images ../../runs/rfdetr/data/test --out ../../runs/rfdetr/pred_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from rfdetr import RFDETRSmall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--threshold", type=float, default=0.05)
    args = ap.parse_args()
    model = RFDETRSmall(pretrain_weights=args.model, resolution=args.resolution)
    coco = json.loads((Path(args.images) / "_annotations.coco.json").read_text(encoding="utf-8"))
    out = {}
    for im in coco["images"]:
        det = model.predict(Image.open(Path(args.images) / im["file_name"]).convert("RGB"), threshold=args.threshold)
        out[im["file_name"]] = [
            {"box": [float(v) for v in b], "class_id": int(c), "score": float(s)} for b, c, s in zip(det.xyxy, det.class_id, det.confidence)
        ]
    Path(args.out).write_text(json.dumps(out), encoding="utf-8")
    print(len(out), "images")


if __name__ == "__main__":
    main()
