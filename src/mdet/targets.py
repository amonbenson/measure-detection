"""Training targets for the boundary model (guideline section 5.4).

For every detected system that matches a ground-truth row, write the rectified strip (s = 8) and
the ground-truth boundaries in strip columns. Targets keep the annotated positions: annotations
are precise after deskew, and snapping would move repeat barlines between their thin and thick
line. The strongest vertical line within +-1 s is used for two things only: a barline boundary
without a line (inside a multi-measure rest) is virtual and ignored in training, and the snap
offset is reported as an annotation QA signal.

Output: dataset/derived/strips/<doc_id>/<page>.npz and a summary in runs/targets/report.md.
Usage: uv run python -m mdet.targets [--workers 6]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .data import DERIVED, RUNS, all_document_ids, load_document
from .evaluate import match_systems
from .normalize import binarize
from .pipeline import page_systems, page_x_to_strip_u
from .rectify import Strip, rectify
from .staves import S

SNAP_RADIUS = 1.0 * S
LINE_MIN_RESPONSE = 0.7


def column_response(strip_img: np.ndarray, strip: Strip) -> np.ndarray:
    """Per column: mean over staves of the fraction of the staff height covered by ink.

    A barline covers every staff completely (response near 1). A stem covers one staff only.
    Neighbouring columns are combined so that lines up to 3 px wide or slightly tilted count.
    """
    bw = binarize(strip_img)
    w = bw.shape[1]
    wide = bw.copy()
    wide[:, 1:] |= bw[:, :-1]
    wide[:, :-1] |= bw[:, 1:]
    per_staff = []
    for vt, vb in strip.staff_bands:
        a, b = int(round(vt)), int(round(vb)) + 1
        per_staff.append(wide[a:b].mean(axis=0))
    return np.mean(per_staff, axis=0) if per_staff else np.zeros(w)


def snap(resp: np.ndarray, u: float, radius: float) -> tuple[float, float]:
    """Position and response of the strongest line within radius of u (centroid of the peak plateau)."""
    w = len(resp)
    a, b = max(0, int(np.floor(u - radius))), min(w, int(np.ceil(u + radius)) + 1)
    if b <= a:
        return u, 0.0
    seg = resp[a:b]
    peak = float(seg.max())
    idx = np.nonzero(seg >= peak - 0.05)[0]
    # Keep the plateau run closest to u.
    runs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
    best = min(runs, key=lambda r: abs(a + r.mean() - u))
    return float(a + best.mean()), peak


def build_page(doc_id: str, page: int, rows) -> tuple[list[dict], dict]:
    stats = {"rows": len(rows), "matched_rows": 0, "boundaries": 0, "virtual": 0, "snapped_far": 0, "snap_offsets": []}
    systems, geo, gray, _ = page_systems(doc_id, page)
    pairs = match_systems([ps.pred for ps in systems], rows)
    samples = []
    for i, j in pairs:
        ps, row = systems[i], rows[j]
        img = rectify(gray, ps.strip)
        resp = column_response(img, ps.strip)
        us = page_x_to_strip_u(geo, ps.strip, np.asarray(row.boundaries()), ps.pred)
        positions, virtual = [], []
        for k, u in enumerate(us):
            su, r = snap(resp, float(u), SNAP_RADIUS)
            is_line = r >= LINE_MIN_RESPONSE
            positions.append(float(u))
            if k == 0:
                # Start boundary: convention-dependent, not always a line. Never virtual.
                virtual.append(False)
                continue
            stats["boundaries"] += 1
            if is_line:
                virtual.append(False)
                stats["snap_offsets"].append(abs(su - u) / S)
                if abs(su - u) > 0.5 * S:
                    stats["snapped_far"] += 1
            else:
                virtual.append(True)
                stats["virtual"] += 1
        stats["matched_rows"] += 1
        samples.append(
            {
                "image": img,
                "staff_bands": np.asarray(ps.strip.staff_bands, np.float32),
                "boundaries_raw": np.asarray(us, np.float32),
                "boundaries": np.asarray(positions, np.float32),
                "virtual": np.asarray(virtual, bool),
                "labels": [m.label for m in row.measures],
                "strip": {
                    "x_start": ps.strip.x_start,
                    "width": ps.strip.width,
                    "margin": ps.strip.margin,
                    "height_nom": ps.strip.height_nom,
                    "top": ps.strip.top.tolist(),
                    "bottom": ps.strip.bottom.tolist(),
                    "staff_spacings": ps.strip.staff_spacings,
                    "system_u0": ps.strip.system_u0,
                    "system_u1": ps.strip.system_u1,
                    "shear": ps.strip.shear,
                },
            }
        )
    return samples, stats


def save_samples(path, samples: list[dict]) -> None:
    arrays = {}
    meta = []
    for k, s in enumerate(samples):
        arrays[f"image_{k}"] = s["image"]
        meta.append({key: (v.tolist() if isinstance(v, np.ndarray) else v) for key, v in s.items() if key != "image"})
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load_samples(path) -> list[dict]:
    z = np.load(path)
    meta = json.loads(z["meta"].tobytes().decode())
    out = []
    for k, m in enumerate(meta):
        m = dict(m)
        m["image"] = z[f"image_{k}"]
        for key in ("staff_bands", "boundaries_raw", "boundaries", "virtual"):
            m[key] = np.asarray(m[key])
        out.append(m)
    return out


def build_document(doc_id: str) -> dict:
    doc = load_document(doc_id)
    out_dir = DERIVED / "strips" / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    total = {"rows": 0, "matched_rows": 0, "boundaries": 0, "virtual": 0, "snapped_far": 0, "snap_offsets": []}
    for p in range(doc.num_pages):
        rows = doc.rows_on_page(p)
        samples, st = build_page(doc_id, p, rows)
        save_samples(out_dir / f"{p:03d}.npz", samples)
        for k in total:
            total[k] += st[k]
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    docs = all_document_ids()
    rep = ["# Boundary target report", "", "| Document | Rows | Matched rows | Barline boundaries | Virtual | Snapped > 0.5 s |", "|---|---|---|---|---|---|"]
    offsets = []
    with ProcessPoolExecutor(args.workers) as ex:
        for d, t in zip(docs, ex.map(build_document, docs)):
            rep.append(f"| `{d}` | {t['rows']} | {t['matched_rows']} | {t['boundaries']} | {t['virtual']} | {t['snapped_far']} |")
            offsets += t["snap_offsets"]
            print(d, {k: v for k, v in t.items() if k != "snap_offsets"}, flush=True)
    o = np.asarray(offsets)
    rep += ["", f"Snap offsets (s): mean {o.mean():.3f}, median {np.median(o):.3f}, p95 {np.percentile(o, 95):.3f}, max {o.max():.3f}"]
    (RUNS / "targets").mkdir(parents=True, exist_ok=True)
    (RUNS / "targets" / "report.md").write_text("\n".join(rep) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
