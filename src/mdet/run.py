"""Run the page pipeline on a split and evaluate it.

Usage:
  uv run python -m mdet.run --detector classical --split val
  uv run python -m mdet.run --detector measurenet --ckpt runs/measurenet/best.pt --split val [--labels runs/labelocr/best.pt]
Writes runs/eval/<name>/summary.json and pages.pkl (per-page results for reports).
"""

from __future__ import annotations

import argparse
import json
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .data import RUNS, load_document, read_split
from .evaluate import PageResult, PredSystem, evaluate_page, summarize
from .pipeline import measures_from_boundaries, page_systems
from .rectify import rectify


def predict_page(doc_id: str, page: int, detector) -> tuple[list[PredSystem], list, object]:
    systems, geo, gray, _ = page_systems(doc_id, page)
    for ps in systems:
        img = rectify(gray, ps.strip)
        bnd = detector(img, ps.strip)
        ps.pred.measures = measures_from_boundaries(geo, ps.strip, ps.pred, [b[0] for b in bnd], [b[1] for b in bnd])
    return [ps.pred for ps in systems], systems, geo


def make_detector(kind: str, ckpt: str | None, tau: float | None):
    if kind == "classical":
        from .boundaries import classical_detector

        return classical_detector() if tau is None else classical_detector(tau)
    from .infer import MeasureNetDetector

    return MeasureNetDetector(ckpt, tau_b=tau)


def run_document(args: tuple) -> list[tuple[PageResult, list[PredSystem]]]:
    doc_id, kind, ckpt, tau, labels = args
    detector = make_detector(kind, ckpt, tau)
    doc = load_document(doc_id)
    out = []
    per_page = []
    for p in range(doc.num_pages):
        pred, systems, geo = predict_page(doc_id, p, detector)
        per_page.append((p, pred, systems, geo))
    if labels:
        from .labelling import label_document

        label_document(doc_id, per_page, labels)
    for p, pred, systems, geo in per_page:
        res = evaluate_page(doc_id, p, geo.s_original, pred, doc.rows_on_page(p))
        out.append((res, pred))
    return out


def evaluate_split(split: str, kind: str, ckpt: str | None = None, tau: float | None = None, labels: str | None = None, workers: int = 6, docs: list[str] | None = None):
    docs = docs or read_split(split)
    jobs = [(d, kind, ckpt, tau, labels) for d in docs]
    results: list[PageResult] = []
    preds = {}
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            outs = list(ex.map(run_document, jobs))
    else:
        outs = [run_document(j) for j in jobs]
    for d, out in zip(docs, outs):
        for res, pred in out:
            results.append(res)
            preds[(d, res.page)] = pred
    return summarize(results), results, preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", choices=["classical", "measurenet"], default="classical")
    ap.add_argument("--ckpt")
    ap.add_argument("--labels", help="LabelOCR checkpoint; enables Stages 4 and 5")
    ap.add_argument("--split", default="val")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--name")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    name = args.name or f"{args.detector}_{args.split}"
    summary, results, preds = evaluate_split(args.split, args.detector, args.ckpt, args.tau, args.labels, args.workers)
    out = RUNS / "eval" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8", newline="\n")
    with open(out / "pages.pkl", "wb") as f:
        pickle.dump({"results": results, "preds": preds, "args": vars(args)}, f)
    for k, v in summary.items():
        print(f"{k:32s} {v:.4f}" if isinstance(v, float) else f"{k:32s} {v}")


if __name__ == "__main__":
    main()
