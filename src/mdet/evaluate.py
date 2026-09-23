"""Evaluation metrics (guideline section 7). All metrics are computed in page coordinates.

A prediction for one page is a list of PredSystem. Tolerances are in staff spaces s of the page
(s_original = s in page-width units).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import Row


@dataclass
class PredMeasure:
    x0: float  # page fraction, at the system's vertical centre
    x1: float
    label: str = ""
    confidence: float = 1.0
    label_confidence: float = 1.0
    quad: list[tuple[float, float]] = field(default_factory=list)  # 4 page-fraction corners (tl, tr, br, bl)


@dataclass
class PredSystem:
    page: int
    x0: float
    x1: float
    y0: float  # top line at the system centre, page fraction
    y1: float  # bottom line at the system centre
    measures: list[PredMeasure] = field(default_factory=list)
    confidence: float = 1.0

    def boundaries(self) -> list[float]:
        if not self.measures:
            return []
        return [self.measures[0].x0] + [m.x1 for m in self.measures]


def vertical_match_score(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    return inter / max(1e-9, min(a1 - a0, b1 - b0))


def match_systems(pred: list[PredSystem], gt: list[Row], min_score: float = 0.5) -> list[tuple[int, int]]:
    """One-to-one matching of predicted systems and ground-truth rows on one page."""
    if not pred or not gt:
        return []
    cost = np.zeros((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, r in enumerate(gt):
            hov = min(p.x1, r.x1) - max(p.x0, r.x0)
            sc = vertical_match_score(p.y0, p.y1, r.y0, r.y1) if hov > 0 else 0.0
            cost[i, j] = -sc
    ii, jj = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(ii, jj) if -cost[i, j] >= min_score]


def match_boundaries(pred: list[float], gt: list[float], tol: float) -> list[tuple[int, int, float]]:
    """One-to-one matching of boundary x positions within tol. Returns (pred idx, gt idx, |dx|)."""
    if not pred or not gt:
        return []
    d = np.abs(np.subtract.outer(np.asarray(pred), np.asarray(gt)))
    cost = np.where(d <= tol, d, 1e6)
    ii, jj = linear_sum_assignment(cost)
    return [(i, j, float(d[i, j])) for i, j in zip(ii, jj) if d[i, j] <= tol]


@dataclass
class PageResult:
    doc_id: str
    page: int
    s: float  # page-width fraction per staff space
    n_pred_systems: int = 0
    n_gt_systems: int = 0
    n_matched_systems: int = 0
    boundary: dict = field(default_factory=dict)  # tol name -> (tp, n_pred, n_gt)
    boundary_errors: list[float] = field(default_factory=list)  # in s, matched at 1 s
    n_gt_measures: int = 0
    n_pred_measures: int = 0
    n_correct_measures: int = 0
    n_correct_labels: int = 0  # correct measure and exact label
    measure_records: list[dict] = field(default_factory=list)  # per GT measure: detected, label ok, confidence
    false_measures: list[dict] = field(default_factory=list)  # predicted measures without GT match
    errors: list[dict] = field(default_factory=list)

    @property
    def exact(self) -> bool:
        return self.n_correct_measures == self.n_gt_measures == self.n_pred_measures and self.n_pred_systems == self.n_gt_systems


def evaluate_page(doc_id: str, page: int, s: float, pred: list[PredSystem], gt: list[Row], tols=(1.0, 0.5)) -> PageResult:
    res = PageResult(doc_id, page, s, n_pred_systems=len(pred), n_gt_systems=len(gt))
    res.n_gt_measures = sum(len(r.measures) for r in gt)
    res.n_pred_measures = sum(len(p.measures) for p in pred)
    for t in tols:
        res.boundary[t] = [0, sum(len(p.boundaries()) for p in pred), sum(len(r.boundaries()) for r in gt)]
    pairs = match_systems(pred, gt)
    res.n_matched_systems = len(pairs)
    matched_gt = {j for _, j in pairs}
    matched_pred = {i for i, _ in pairs}
    for j, r in enumerate(gt):
        if j not in matched_gt:
            res.errors.append({"type": "missed_system", "label": r.measures[0].label})
            for m in r.measures:
                res.measure_records.append({"detected": False, "label_ok": False, "confidence": 0.0, "gt": m.label, "pred": None})
    for i, p in enumerate(pred):
        if i not in matched_pred:
            res.errors.append({"type": "spurious_system", "y": p.y0})
            for pm in p.measures:
                res.false_measures.append({"confidence": pm.confidence, "label": pm.label})
    for i, j in pairs:
        p, r = pred[i], gt[j]
        pb, gb = p.boundaries(), r.boundaries()
        for t in tols:
            mt = match_boundaries(pb, gb, t * s)
            res.boundary[t][0] += len(mt)
            if t == 1.0:
                res.boundary_errors += [d / s for _, _, d in mt]
                g2p = {g: q for q, g, _ in mt}
                matched_p = {q for q, _, _ in mt}
                for g in range(len(gb)):
                    if g not in g2p:
                        kind = "start_edge" if g == 0 else "missed_barline"
                        res.errors.append({"type": kind, "x": gb[g], "label": r.measures[min(g, len(r.measures) - 1)].label})
                for q in range(len(pb)):
                    if q not in matched_p:
                        res.errors.append({"type": "spurious_barline", "x": pb[q]})
                used_pm = set()
                for k, m in enumerate(r.measures):
                    a, b = g2p.get(k), g2p.get(k + 1)
                    ok = a is not None and b is not None and b == a + 1
                    rec = {"detected": ok, "label_ok": False, "confidence": 0.0, "gt": m.label, "pred": None}
                    if ok:
                        pm = p.measures[a]
                        used_pm.add(a)
                        res.n_correct_measures += 1
                        rec.update(pred=pm.label, confidence=min(pm.confidence, pm.label_confidence), label_ok=pm.label == m.label)
                        if pm.label == m.label:
                            res.n_correct_labels += 1
                        else:
                            res.errors.append({"type": "label", "gt": m.label, "pred": pm.label})
                    res.measure_records.append(rec)
                for q, pm in enumerate(p.measures):
                    if q not in used_pm:
                        res.false_measures.append({"confidence": pm.confidence, "label": pm.label})
    return res


def summarize(results: list[PageResult]) -> dict:
    out: dict = {}
    ps = sum(r.n_pred_systems for r in results)
    gs = sum(r.n_gt_systems for r in results)
    ms = sum(r.n_matched_systems for r in results)
    out["system_precision"] = ms / max(ps, 1)
    out["system_recall"] = ms / max(gs, 1)
    for t in (1.0, 0.5):
        tp = sum(r.boundary[t][0] for r in results)
        npred = sum(r.boundary[t][1] for r in results)
        ngt = sum(r.boundary[t][2] for r in results)
        p, rc = tp / max(npred, 1), tp / max(ngt, 1)
        out[f"barline_precision@{t}s"] = p
        out[f"barline_recall@{t}s"] = rc
        out[f"barline_f1@{t}s"] = 2 * p * rc / max(p + rc, 1e-9)
    errs = np.array([e for r in results for e in r.boundary_errors])
    out["boundary_error_mean_s"] = float(errs.mean()) if len(errs) else float("nan")
    out["boundary_error_p95_s"] = float(np.percentile(errs, 95)) if len(errs) else float("nan")
    gm = sum(r.n_gt_measures for r in results)
    pm = sum(r.n_pred_measures for r in results)
    cm = sum(r.n_correct_measures for r in results)
    mp, mr = cm / max(pm, 1), cm / max(gm, 1)
    out["measure_precision"] = mp
    out["measure_recall"] = mr
    out["measure_f1"] = 2 * mp * mr / max(mp + mr, 1e-9)
    out["page_exact_rate"] = sum(r.exact for r in results) / max(len(results), 1)
    cl = sum(r.n_correct_labels for r in results)
    out["conditional_label_accuracy"] = cl / max(cm, 1)
    out["end_to_end_label_accuracy"] = cl / max(gm, 1)
    out["n_pages"] = len(results)
    out["n_gt_measures"] = gm
    return out
