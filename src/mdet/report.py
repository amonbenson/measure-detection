"""Evaluation reports: review curve, stratified metrics and the error log (guideline 7.2, 7.3, 7.5).

Usage: uv run python -m mdet.report runs/eval/<name> [...]  (writes report.md next to pages.pkl)
"""

from __future__ import annotations

import argparse
import collections
import json
import pickle
from pathlib import Path

import numpy as np

from .data import SPLITS, read_meta
from .evaluate import PageResult, summarize
from .geometry import PageGeometry


def review_curve(results: list[PageResult], rates=(0.0, 0.01, 0.02, 0.05, 0.1, 0.2)) -> list[tuple[float, float]]:
    """Residual error after reviewing the r least confident items.

    Items are all ground-truth measures (correct = detected with the exact label; confidence is the
    detection and label confidence; missed measures have confidence 0) plus all false measures.
    Residual error = unreviewed wrong items / number of ground-truth measures.
    """
    items = []
    for r in results:
        for m in r.measure_records:
            items.append((m["confidence"], m["detected"] and m["label_ok"]))
        for f in r.false_measures:
            items.append((min(f["confidence"], 1.0), False))
    n_gt = sum(r.n_gt_measures for r in results)
    items.sort(key=lambda t: t[0])
    out = []
    for rate in rates:
        k = int(round(rate * len(items)))
        wrong = sum(1 for _, ok in items[k:] if not ok)
        out.append((rate, wrong / max(n_gt, 1)))
    return out


def strata(results: list[PageResult]) -> dict[str, dict[str, list[PageResult]]]:
    meta = read_meta()
    groups: dict[str, dict[str, list[PageResult]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in results:
        m = meta[r.doc_id]
        groups["show"][m["show"]].append(r)
        groups["score type"][m["score_type"]].append(r)
        groups["source"][m["source"]].append(r)
        groups["number position"][m["number_position"]].append(r)
        groups["alphanumeric labels"][str(m.get("has_alphanumeric_labels"))].append(r)
        groups["start convention"][str(m.get("start_convention"))].append(r)
        skew = abs(PageGeometry.load(r.doc_id, r.page).info["deskew_deg"])
        groups["skew"]["<1 deg" if skew < 1 else ("1-3 deg" if skew <= 3 else ">3 deg")].append(r)
    return groups


KEYS = [
    ("system_recall", "System R"),
    ("barline_f1@1.0s", "Barline F1 1 s"),
    ("barline_f1@0.5s", "Barline F1 0.5 s"),
    ("measure_f1", "Measure F1"),
    ("page_exact_rate", "Page exact"),
    ("end_to_end_label_accuracy", "E2E label"),
]


def fmt(v: float) -> str:
    return f"{100 * v:.2f}"


def report(run_dir: Path) -> str:
    data = pickle.loads((run_dir / "pages.pkl").read_bytes())
    results: list[PageResult] = data["results"]
    s = summarize(results)
    lines = [f"# Evaluation report: `{run_dir.name}`", "", f"Arguments: `{json.dumps(data['args'])}`", "", "## Overall", "", "| Metric | Value |", "|---|---|"]
    for k, v in s.items():
        lines.append(f"| {k} | {fmt(v) + ' %' if isinstance(v, float) and 'error' not in k else v} |")
    if data.get("tolerant"):
        st = summarize(data["tolerant"])
        lines += ["", "Rest-tolerant variant (virtual boundaries inside multi-measure rests accepted within half a sub-measure):", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {fmt(v) + ' %' if isinstance(v, float) and 'error' not in k else v} |" for k, v in st.items()]
    challenge = SPLITS / "challenge.txt"
    if challenge.exists():
        pages = {(ln.split()[0], int(ln.split()[1])) for ln in challenge.read_text().splitlines() if ln.strip()}
        sub = [r for r in results if (r.doc_id, r.page) in pages]
        if sub:
            sc = summarize(sub)
            lines += ["", f"Challenge pages ({len(sub)}): " + ", ".join(f"{h} {fmt(sc[k])} %" for k, h in KEYS)]
    lines += ["", "Boundary errors are in staff spaces s.", "", "## Review curve", "", "| Review rate | Residual error |", "|---|---|"]
    for rate, err in review_curve(results):
        lines.append(f"| {100 * rate:.0f} % | {fmt(err)} % |")
    lines += ["", "## Stratified", ""]
    for name, grp in strata(results).items():
        lines += [f"### {name}", "", "| Stratum | Pages | Measures | " + " | ".join(h for _, h in KEYS) + " |", "|---" * (3 + len(KEYS)) + "|"]
        for key, rs in sorted(grp.items()):
            ss = summarize(rs)
            lines.append(f"| {key} | {len(rs)} | {ss['n_gt_measures']} | " + " | ".join(fmt(ss[k]) for k, _ in KEYS) + " |")
        lines.append("")
    errs = collections.Counter(e["type"] for r in results for e in r.errors)
    lines += ["## Error log (automatic categories)", "", "| Category | Count |", "|---|---|"] + [f"| {k} | {v} |" for k, v in errs.most_common()]
    labs = collections.Counter((e["gt"], e["pred"]) for r in results for e in r.errors if e["type"] == "label")
    if labs:
        lines += ["", "Most frequent label confusions (ground truth -> prediction): " + ", ".join(f"`{a}`->`{b}` ({n})" for (a, b), n in labs.most_common(15))]
    worst = sorted(results, key=lambda r: r.n_correct_labels - r.n_gt_measures)[:10]
    lines += ["", "## Pages with the most errors", "", "| Document | Page | GT measures | Correct detections | Correct labels | Errors |", "|---|---|---|---|---|---|"]
    for r in worst:
        c = collections.Counter(e["type"] for e in r.errors)
        lines.append(f"| `{r.doc_id}` | {r.page} | {r.n_gt_measures} | {r.n_correct_measures} | {r.n_correct_labels} | {dict(c)} |")
    text = "\n".join(lines) + "\n"
    (run_dir / "report.md").write_text(text, encoding="utf-8", newline="\n")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    for r in args.runs:
        print(report(Path(r)))


if __name__ == "__main__":
    main()
