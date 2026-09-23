"""Golden outputs for the browser parity test (guideline 8.5).

Runs the Python pipeline (mdet.predict) on reference documents until at least --pages pages with
measures are covered, and writes runs/parity/<doc_id>.json plus runs/parity/index.json, which
packages/measure-detection/test/parity.mjs reads.
Usage: uv run python -m mdet.parity [--pages 50] [--split test]
"""

from __future__ import annotations

import argparse
import json

from .data import DOCUMENTS, RUNS, read_meta, read_split
from .predict import predict_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurenet", default=str(RUNS / "measurenet" / "best.pt"))
    ap.add_argument("--labelocr", default=str(RUNS / "labelocr" / "best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--pages", type=int, default=50)
    args = ap.parse_args()
    out = RUNS / "parity"
    out.mkdir(parents=True, exist_ok=True)
    meta = read_meta()
    index, n = [], 0
    for d in read_split(args.split):
        if n >= args.pages:
            break
        conv = meta[d]["start_convention"]
        pdf = str(DOCUMENTS / d / "score.pdf")
        g = {"doc_id": d, "convention": conv, **predict_pdf(pdf, args.measurenet, args.labelocr, conv)}
        (out / f"{d}.json").write_text(json.dumps(g, indent=1), encoding="utf-8", newline="\n")
        index.append({"pdf": pdf, "golden": str(out / f"{d}.json"), "convention": conv})
        n += len({m["layout"]["page"] for m in g["measures"]})
        print(d, len(g["measures"]), "measures", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8", newline="\n")
    print(f"{n} pages with measures in {len(index)} documents")


if __name__ == "__main__":
    main()
