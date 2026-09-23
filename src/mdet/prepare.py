"""Stages 0 and 1 for the whole dataset: render, estimate s, deskew, write normalized pages.

Output per page (dataset/derived/pages/<doc_id>/):
  <page>.png   normalized grayscale page at s = 16 px, deskewed
  <page>.json  page geometry: render size, s of the render, deskew angle, affine map render -> normalized
Usage: uv run python -m mdet.prepare [--workers 4] [--docs id ...]
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .data import DERIVED, DOCUMENTS, ROOT, all_document_ids
from .normalize import apply_normalization, estimate_skew, estimate_staff_space, normalization_matrix

RENDERER = ROOT / "tools" / "render" / "render.mjs"
PASS1_DPI = 150.0
PASS2_TARGET_S = 18.0  # render pass 2 slightly above the normalized s, then downsample
NORM_S = 16.0  # normalized page resolution; s = 8 images are 2x downscales
MIN_MODE_STRENGTH = 0.05  # below this the page has no dominant staff (e.g. a text-only page)


def render(jobs: list[dict]) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(jobs, f)
        path = f.name
    out = subprocess.run(["node", str(RENDERER), path], capture_output=True, text=True, check=True)
    Path(path).unlink()
    return [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]


def prepare_pdf(pdf: str, doc_id: str = "") -> list[tuple[dict, np.ndarray]]:
    """Stages 0 and 1 for a PDF in memory: [(page geometry info, normalized page at s = 16)]."""
    import pymupdf

    with pymupdf.open(pdf) as f:
        n = len(f)
    pdf = str(pdf)
    with tempfile.TemporaryDirectory() as tmp:
        # Pass 1: moderate resolution, estimate s.
        r1 = render([{"pdf": pdf, "page": p, "scale": PASS1_DPI / 72, "out": f"{tmp}/p1_{p}.png"} for p in range(n)])
        est1 = [estimate_staff_space(cv2.imread(r["out"], cv2.IMREAD_GRAYSCALE)) for r in r1]
        strong = [e["s"] for e in est1 if e["mode_strength"] >= MIN_MODE_STRENGTH]
        s_doc = statistics.median(strong) if strong else 8.0
        s1 = [e["s"] if e["mode_strength"] >= MIN_MODE_STRENGTH else s_doc for e in est1]
        # Pass 2: resolution with s close to PASS2_TARGET_S.
        scales = [PASS1_DPI / 72 * PASS2_TARGET_S / s for s in s1]
        r2 = render([{"pdf": pdf, "page": p, "scale": scales[p], "out": f"{tmp}/p2_{p}.png"} for p in range(n)])
        results = []
        for p, r in enumerate(r2):
            gray = cv2.imread(r["out"], cv2.IMREAD_GRAYSCALE)
            est = estimate_staff_space(gray)
            has_staff = est["mode_strength"] >= MIN_MODE_STRENGTH
            s2 = est["s"] if has_staff else PASS2_TARGET_S * s_doc / s1[p]
            angle = estimate_skew(gray, s2) if has_staff else 0.0
            m, size = normalization_matrix(gray.shape[1], gray.shape[0], s2, angle, NORM_S)
            info = {
                "doc_id": doc_id,
                "page": p,
                "render_scale": scales[p],
                "render_width": gray.shape[1],
                "render_height": gray.shape[0],
                "page_width_pt": r["pageWidthPt"],
                "page_height_pt": r["pageHeightPt"],
                "s_render": s2,
                "s_pass1": est1[p]["s"],
                "line_thickness": est["line_thickness"],
                "mode_strength": est["mode_strength"],
                "has_staff": has_staff,
                "deskew_deg": angle,
                "norm_s": NORM_S,
                "norm_matrix": m.tolist(),
                "norm_width": size[0],
                "norm_height": size[1],
            }
            results.append((info, apply_normalization(gray, m, size)))
    return results


def prepare_document(doc_id: str) -> list[dict]:
    out_dir = DERIVED / "pages" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    infos = []
    for info, norm16 in prepare_pdf(str(DOCUMENTS / doc_id / "score.pdf"), doc_id):
        p = info["page"]
        cv2.imwrite(str(out_dir / f"{p:03d}.png"), norm16)
        (out_dir / f"{p:03d}.json").write_text(json.dumps(info, indent=1), encoding="utf-8", newline="\n")
        infos.append(info)
    return infos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--docs", nargs="*")
    args = ap.parse_args()
    docs = args.docs or all_document_ids(labeled_only=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for doc_id, res in zip(docs, ex.map(prepare_document, docs)):
            s = [r["s_render"] for r in res]
            a = [r["deskew_deg"] for r in res]
            print(f"{doc_id}: {len(res)} pages, s_render {min(s):.2f}..{max(s):.2f}, deskew {min(a):+.2f}..{max(a):+.2f}", flush=True)


if __name__ == "__main__":
    main()
