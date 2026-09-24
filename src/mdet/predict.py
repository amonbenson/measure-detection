"""Full pipeline on a PDF: measures and labels in the training JSON schema, extended with quad and confidence.

Usage: uv run python -m mdet.predict score.pdf --measurenet runs/measurenet/best.pt --labelocr runs/labelocr/best.pt
       [--out result.json]
Output: { "pages": [{page, s, deskew_deg}], "measures": [{ "layout": {page, x, y, width, height},
           "value", "quad": [[x, y] x 4], "confidence", "boundary_confidence", "label_confidence" }] }
        with measures in reading order; s is the staff space in page-width units.
"""

from __future__ import annotations

import argparse
import json

from .evaluate import PredSystem
from .geometry import PageGeometry
from .infer import MeasureNetDetector
from .labelling import label_document
from .pipeline import measures_from_boundaries, systems_for_page
from .prepare import prepare_pdf
from .rectify import rectify


def predict_pdf(pdf: str, measurenet: str, labelocr: str, tau_b: float | None = None) -> dict:
    detector = MeasureNetDetector(measurenet, tau_b=tau_b)
    per_page, pages = [], []
    for info, norm16 in prepare_pdf(pdf):
        geo = PageGeometry.from_info(info, norm16)
        systems, geo, gray, _ = systems_for_page(geo)
        for ps in systems:
            bnd = detector(rectify(gray, ps.strip), ps.strip)
            ps.us = [b[0] for b in bnd]
            ps.pred.measures = measures_from_boundaries(geo, ps.strip, ps.pred, ps.us, [b[1] for b in bnd])
        per_page.append((info["page"], [ps.pred for ps in systems], systems, geo))
        pages.append({"page": info["page"], "s": info["s_render"] / info["render_width"], "deskew_deg": info["deskew_deg"]})
    label_document("", per_page, labelocr)
    return {"pages": pages, "measures": to_json([pred for _, preds, _, _ in per_page for pred in preds])}


def to_json(systems: list[PredSystem]) -> list[dict]:
    out = []
    for sy in systems:
        for m in sy.measures:
            xs = [p[0] for p in m.quad]
            ys = [p[1] for p in m.quad]
            out.append(
                {
                    "layout": {"page": sy.page, "x": min(xs), "y": min(ys), "width": max(xs) - min(xs), "height": max(ys) - min(ys)},
                    "value": m.label,
                    "quad": [[round(x, 6), round(y, 6)] for x, y in m.quad],
                    "confidence": min(m.confidence, m.label_confidence),
                    "boundary_confidence": m.confidence,
                    "label_confidence": m.label_confidence,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--measurenet", required=True)
    ap.add_argument("--labelocr", required=True)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--out")
    args = ap.parse_args()
    res = predict_pdf(args.pdf, args.measurenet, args.labelocr, args.tau)
    text = json.dumps(res, indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
