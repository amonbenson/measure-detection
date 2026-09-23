"""Bootstrapping printed measure-number locations (guideline section 5.5), without external OCR.

The label of every measure is known. A text candidate is accepted as the printed label of a
measure when its glyph count equals the label length and its position relative to the measure's
left boundary agrees with the document's placement statistics:
  pass 1: nearest count-matching candidate per measure -> per-document side and offset statistics
  pass 2: accept only candidates on a dominant side within max(1 s, 3 MAD) of the median offset
A multi-measure rest (measures separated by virtual boundaries) expects its range label `a-b`.
Candidates of rows in which every measure matched are negatives (target: empty string).

Output: dataset/derived/numbers/<doc_id>.npz with crops (s = 16), texts and placement records.
Usage: uv run python -m mdet.bootstrap [--workers 6]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import DERIVED, RUNS, all_document_ids, load_document
from .evaluate import match_systems
from .geometry import load_norm16
from .pipeline import page_systems, page_x_to_strip_u
from .staves import S
from .targets import LINE_MIN_RESPONSE, SNAP_RADIUS, column_response, snap
from .rectify import rectify
from .text import crop, strip16, text_candidates

# Pass-1 search window of the candidate centre relative to the measure's left boundary, in s.
# The first measure of a system has its own window: its number is often printed under the clef.
WINDOW = {"first": (-3.0, 8.0), "other": (-3.0, 4.0)}


def position(unit: dict) -> str:
    return "first" if unit["first"] == 0 else "other"


def glyph_count(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def printed_units(row_labels: list[str], virtual: list[bool]) -> list[dict]:
    """Group measures into printed units: a single measure, or a multi-measure rest (range label)."""
    units, k = [], 0
    n = len(row_labels)
    while k < n:
        j = k
        while j + 1 < n and virtual[j + 1]:  # boundary j+1 is between measure j and j+1
            j += 1
        text = row_labels[k] if j == k else f"{row_labels[k]}-{row_labels[j]}"
        units.append({"first": k, "last": j, "text": text})
        k = j + 1
    return units


def collect_document(doc_id: str) -> dict:
    """Candidates and units per matched row (pass 1 statistics are computed by the caller)."""
    doc = load_document(doc_id)
    rows_out = []
    for p in range(doc.num_pages):
        rows = doc.rows_on_page(p)
        if not rows:
            continue
        systems, geo, gray, _ = page_systems(doc_id, p)
        g16 = load_norm16(doc_id, p)
        for i, j in match_systems([ps.pred for ps in systems], rows):
            ps, row = systems[i], rows[j]
            us = page_x_to_strip_u(geo, ps.strip, np.asarray(row.boundaries()), ps.pred)
            resp = column_response(rectify(gray, ps.strip), ps.strip)
            virtual = [False] + [snap(resp, float(u), SNAP_RADIUS)[1] < LINE_MIN_RESPONSE for u in us[1:]]
            im16 = strip16(g16, ps.strip)
            cands = text_candidates(im16, ps.strip)
            rows_out.append(
                {
                    "page": p,
                    "row": j,
                    "bounds": us.tolist(),
                    "units": printed_units([m.label for m in row.measures], virtual),
                    "cands": cands,
                    "crops": [crop(im16, c) for c in cands],
                }
            )
    return {"doc_id": doc_id, "rows": rows_out}


def _nearest(cands, u_left: float, text: str, window=(-3.0, 4.0), side: str | None = None, center: float | None = None, tol: float | None = None, used=()):
    best, best_d = None, None
    for ci, c in enumerate(cands):
        if ci in used or c.n_components != glyph_count(text):
            continue
        if side is not None and c.side != side:
            continue
        dx = (c.uc - u_left) / S
        if center is None:
            if not window[0] <= dx <= window[1]:
                continue
            d = abs(dx)
        else:
            d = abs(dx - center)
            if d > tol:
                continue
        if best_d is None or d < best_d:
            best, best_d = ci, d
    return best


def bootstrap_document(doc_id: str) -> dict:
    data = collect_document(doc_id)
    # Pass 1: placement statistics per (side, position).
    obs: dict[str, list[float]] = {}
    for r in data["rows"]:
        for un in r["units"]:
            pos = position(un)
            ci = _nearest(r["cands"], r["bounds"][un["first"]], un["text"], WINDOW[pos])
            if ci is not None:
                c = r["cands"][ci]
                obs.setdefault(f"{c.side}/{pos}", []).append((c.uc - r["bounds"][un["first"]]) / S)
    stats = {}
    for pos in ("first", "other"):
        total = sum(len(v) for k, v in obs.items() if k.endswith(pos))
        for side in ("above", "below"):
            v = obs.get(f"{side}/{pos}", [])
            if total and len(v) >= max(3, 0.2 * total):
                v = np.asarray(v)
                med = float(np.median(v))
                mad = float(np.median(np.abs(v - med)))
                stats[f"{side}/{pos}"] = {"median": med, "mad": mad, "tol": max(1.0, 3 * 1.4826 * mad), "n": len(v)}
    # Pass 2: accept, and collect negatives.
    crops, texts, records = [], [], []
    n_units = n_matched = 0
    for r in data["rows"]:
        used: set[int] = set()
        all_matched = True
        for un in r["units"]:
            n_units += 1
            pos = position(un)
            u_left = r["bounds"][un["first"]]
            best = None
            for key, st in stats.items():
                side, kpos = key.split("/")
                if kpos != pos:
                    continue
                ci = _nearest(r["cands"], u_left, un["text"], side=side, center=st["median"], tol=st["tol"], used=used)
                if ci is not None:
                    d = abs((r["cands"][ci].uc - u_left) / S - st["median"])
                    if best is None or d < best[1]:
                        best = (ci, d)
            if best is None:
                all_matched = False
                continue
            ci = best[0]
            used.add(ci)
            n_matched += 1
            c = r["cands"][ci]
            crops.append(r["crops"][ci])
            texts.append(un["text"])
            records.append({"page": r["page"], "row": r["row"], "text": un["text"], "side": c.side, "position": pos, "dx": (c.uc - u_left) / S, "first": un["first"], "last": un["last"], "positive": True})
        if all_matched:
            for ci, c in enumerate(r["cands"]):
                if ci not in used:
                    crops.append(r["crops"][ci])
                    texts.append("")
                    records.append({"page": r["page"], "row": r["row"], "text": "", "side": c.side, "dx": None, "positive": False})
    out = DERIVED / "numbers"
    out.mkdir(parents=True, exist_ok=True)
    arrays = {f"crop_{k}": c for k, c in enumerate(crops)}
    arrays["meta"] = np.frombuffer(json.dumps({"texts": texts, "records": records, "stats": stats}, default=float).encode(), dtype=np.uint8)
    np.savez_compressed(out / f"{doc_id}.npz", **arrays)
    return {"doc_id": doc_id, "units": n_units, "matched": n_matched, "negatives": texts.count(""), "stats": stats}


def load_numbers(doc_id: str) -> tuple[list[np.ndarray], list[str], list[dict], dict]:
    z = np.load(DERIVED / "numbers" / f"{doc_id}.npz")
    meta = json.loads(z["meta"].tobytes().decode())
    crops = [z[f"crop_{k}"] for k in range(len(meta["texts"]))]
    return crops, meta["texts"], meta["records"], meta["stats"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    docs = all_document_ids()
    rep = ["# Measure-number bootstrapping report", "", "| Document | Printed units | Matched | Rate | Negatives | Placement |", "|---|---|---|---|---|---|"]
    tu = tm = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(bootstrap_document, docs):
            tu += r["units"]
            tm += r["matched"]
            pl = ", ".join(f"{k}: {v['median']:+.2f} s (tol {v['tol']:.2f}, n {v['n']})" for k, v in r["stats"].items())
            rep.append(f"| `{r['doc_id']}` | {r['units']} | {r['matched']} | {r['matched'] / max(r['units'], 1):.1%} | {r['negatives']} | {pl} |")
            print(rep[-1], flush=True)
    rep += ["", f"Total: {tm} of {tu} printed units matched ({tm / max(tu, 1):.1%})."]
    (RUNS / "bootstrap").mkdir(parents=True, exist_ok=True)
    (RUNS / "bootstrap" / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
