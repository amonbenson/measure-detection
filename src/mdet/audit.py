"""Data audit (guideline section 5.2).

Writes runs/audit/report.md and overlay images: all flagged pages plus a random sample.
Usage: uv run python -m mdet.audit [--sample 50]
"""

from __future__ import annotations

import argparse
import collections
import json
import random

import numpy as np
import pymupdf
from PIL import Image, ImageDraw

from .data import RUNS, Document, all_document_ids, load_document, load_raw, parse_label, parse_value
from .labels import transition_type

TOL = 1e-4


def render_overlay(doc: Document, page: int, out, dpi: int = 100, note: str = "") -> None:
    with pymupdf.open(doc.pdf) as pdf:
        pix = pdf[page].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    im = Image.frombytes("L", (pix.width, pix.height), pix.samples).convert("RGB")
    w, h = im.size
    dr = ImageDraw.Draw(im)
    for r in doc.rows_on_page(page):
        for m in r.measures:
            dr.rectangle([m.x0 * w, m.y0 * h, m.x1 * w, m.y1 * h], outline=(220, 0, 0), width=2)
            dr.text((m.x0 * w + 3, m.y0 * h + 2), m.label, fill=(0, 0, 255))
    if note:
        dr.text((5, 5), note, fill=(255, 0, 255))
    im.save(out)


def audit(sample: int, seed: int = 0) -> None:
    out_dir = RUNS / "audit"
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    flags: list[tuple[str, int, str]] = []  # (doc_id, page, reason)
    totals = collections.Counter()
    transitions = collections.Counter()
    free_jumps: list[str] = []
    charset = collections.Counter()
    unparseable: list[str] = []
    systems_per_page: list[int] = []
    measures_per_system: list[int] = []
    order_displacements = 0
    docs: list[Document] = []

    for doc_id in all_document_ids(labeled_only=False):
        raw = load_raw(doc_id)
        if not any(e.get("layout") for e in raw):
            totals["unlabeled_documents"] += 1
            lines.append(f"- Unlabeled document `{doc_id}` ({len(raw)} entries): excluded.")
            continue
        doc = load_document(doc_id)
        docs.append(doc)
        totals["documents"] += 1
        totals["raw_entries"] += len(raw)
        totals["pages"] += doc.num_pages
        for k, v in doc.dropped.items():
            totals[f"dropped_{k}"] += v
        for e in raw:
            label, pas = parse_value(e["value"])
            if pas > 1:
                totals["pass_entries"] += 1
                if e.get("layout"):
                    flags.append((doc_id, e["layout"]["page"], f"pass {pas} entry `{e['value']}` has a box"))
        if doc.dropped["duplicate_box"]:
            pages = sorted({m.page for m in doc.measures})
            flags.append((doc_id, pages[0], f"{doc.dropped['duplicate_box']} duplicate boxes"))

        ms = doc.measures
        totals["measures"] += len(ms)
        totals["rows"] += len(doc.rows)
        # Reading order versus array order.
        idx = [m.index for m in ms]
        order_displacements += sum(1 for a, b in zip(idx, idx[1:]) if b < a)
        # Pages.
        used = collections.Counter(r.page for r in doc.rows)
        for p in range(doc.num_pages):
            systems_per_page.append(used.get(p, 0))
            if p not in used:
                totals["pages_without_measures"] += 1
                flags.append((doc_id, p, "page without measures"))
        # Bounds, row consistency, tiling, row overlap.
        for r in doc.rows:
            measures_per_system.append(len(r.measures))
            for m in r.measures:
                if min(m.box) < -TOL or max(m.box) > 1 + TOL:
                    flags.append((doc_id, r.page, f"box out of bounds `{m.label}`"))
            ys = {(round(m.y0, 4), round(m.y1, 4)) for m in r.measures}
            if len(ys) > 1:
                totals["rows_inconsistent_y"] += 1
                spread = max(abs(m.y0 - r.measures[0].y0) + abs(m.y1 - r.measures[0].y1) for m in r.measures)
                if spread > 0.01:
                    flags.append((doc_id, r.page, f"row with varying y/height (spread {spread:.3f}) at `{r.measures[0].label}`"))
            for a, b in zip(r.measures, r.measures[1:]):
                gap = b.x0 - a.x1
                if abs(gap) > TOL:
                    totals["tiling_gaps"] += 1
                    flags.append((doc_id, r.page, f"tiling gap {gap:+.4f} between `{a.label}` and `{b.label}`"))
        for p in used:
            rs = doc.rows_on_page(p)
            for a, b in zip(rs, rs[1:]):
                if b.y0 < a.y1 - TOL:
                    totals["row_overlaps"] += 1
                    flags.append((doc_id, p, f"rows overlap vertically at `{a.measures[0].label}` / `{b.measures[0].label}`"))
        # Labels.
        for m in ms:
            charset.update(m.label)
            if parse_label(m.label) is None:
                unparseable.append(f"{doc_id}:{m.label}")
        for a, b in zip(ms, ms[1:]):
            t = transition_type(a.label, b.label)
            transitions[t] += 1
            if t == "free_jump":
                free_jumps.append(f"`{doc_id}` p{b.page}: `{a.label}` -> `{b.label}`")
        transitions[f"first={ms[0].label}"] += 1

    # Report.
    rep = ["# Data audit report", ""]
    rep += ["## Totals", "", "| Quantity | Count |", "|---|---|"]
    rep += [f"| {k} | {v} |" for k, v in sorted(totals.items())]
    rep += ["", f"Reading order differs from array order at {order_displacements} places (array is a playback order).", ""]
    rep += ["## Unlabeled documents", ""] + lines + [""]
    rep += ["## Label transitions (reading order)", "", "| Type | Count |", "|---|---|"]
    rep += [f"| {k} | {v} |" for k, v in transitions.most_common()]
    rep += ["", "### Free jumps", ""] + [f"- {j}" for j in free_jumps] + [""]
    rep += ["## Charset", "", "`" + "".join(sorted(charset)) + "`", ""]
    rep += ["Unparseable labels: " + (", ".join(unparseable) if unparseable else "none"), ""]
    spp, mps = np.array(systems_per_page), np.array(measures_per_system)
    rep += ["## Per-page statistics", ""]
    rep += [f"- Systems per page: mean {spp.mean():.2f}, max {spp.max()}, histogram {dict(sorted(collections.Counter(spp.tolist()).items()))}"]
    rep += [f"- Measures per system: mean {mps.mean():.2f}, max {mps.max()}, histogram {dict(sorted(collections.Counter(mps.tolist()).items()))}", ""]
    rep += ["## Flags", "", "| Document | Page | Reason |", "|---|---|---|"]
    rep += [f"| `{d}` | {p} | {why} |" for d, p, why in flags]
    (out_dir / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8", newline="\n")

    # Overlays: every flagged page plus a random sample.
    by_id = {d.doc_id: d for d in docs}
    flagged = sorted({(d, p) for d, p, _ in flags if d in by_id})
    rng = random.Random(seed)
    all_pages = [(d.doc_id, p) for d in docs for p in sorted({r.page for r in d.rows})]
    sampled = rng.sample(all_pages, min(sample, len(all_pages)))
    for d, p in flagged:
        render_overlay(by_id[d], p, out_dir / "overlays" / f"flag_{d}_{p:03d}.png")
    for d, p in sampled:
        render_overlay(by_id[d], p, out_dir / "overlays" / f"sample_{d}_{p:03d}.png")
    print(f"report: {out_dir / 'report.md'}; {len(flags)} flags, {len(flagged)} flagged pages, {len(sampled)} sampled pages")
    print(json.dumps(totals, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()
    audit(args.sample)


if __name__ == "__main__":
    main()
