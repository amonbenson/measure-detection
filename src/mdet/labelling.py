"""Stages 4 and 5: assign OCR readings to measures and decode the document's label sequence.

Stage 4 (guideline 4.8): candidates in the strip margins are read by LabelOCR. Each reading is
assigned to at most one measure of its system (Hungarian algorithm). The cost is the negative log
likelihood of the candidate's horizontal offset from the measure's left boundary, per side
(above / below) and position (first measure of the system / other), plus a fixed cost for leaving
a reading unassigned. The document's placement is estimated on the fly: a second assignment pass
uses offset densities fitted to the document's own first-pass assignments.
A measure whose reading is a range label (`12-15`) is split into equal measures.

Stage 5 (guideline 4.9): Viterbi decoding over all measures of the document in reading order.
Transitions follow the label grammar with probabilities counted on the training labels, so
consecutive numbering is the dominant prior; readings are evidence, not truth. Label confidence
is the posterior marginal of the chosen label (forward-backward).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.optimize import linear_sum_assignment

from .data import RUNS, load_document, parse_label, read_split
from .evaluate import PredMeasure
from .labels import TRANSITIONS, expand_range, successors, transition_type
from .staves import S
from .text import Candidate, crop, ocr_input, strip16, text_candidates

# Decoder parameters fitted on the training documents; MDET_DECODER selects another file (e.g. a fold).
PARAMS_PATH = Path(os.environ.get("MDET_DECODER", RUNS / "decoder" / "params.json"))
UNASSIGNED_COST = 6.0
NOT_A_NUMBER = 0.5  # readings whose empty-string probability exceeds this are discarded
EMISSION_FLOOR = 1e-3
BEAM = 24
# A free jump can go to any label; its type probability is shared by this many plausible targets,
# while every other transition type has one or two targets.
FREE_JUMP_TARGETS = 100


# ------------------------------------------------------------------ parameters fitted on train


def fit_params(train_docs: list[str], path: Path | None = None) -> dict:
    """Transition-type probabilities and offset densities from the training documents."""
    from .bootstrap import load_numbers

    counts = Counter()
    first = Counter()
    for d in train_docs:
        labels = [m.label for m in load_document(d).measures]
        first[labels[0]] += 1
        for a, b in zip(labels, labels[1:]):
            counts[transition_type(a, b)] += 1
    total = sum(counts.values())
    # Add-one smoothing; free jumps and resets also get a minimum probability.
    trans = {t: (counts[t] + 1) / (total + len(TRANSITIONS)) for t in TRANSITIONS}
    offsets: dict[str, list[float]] = {}
    for d in train_docs:
        _, _, records, _ = load_numbers(d)
        for r in records:
            if r["positive"] and r["first"] == r["last"]:
                offsets.setdefault(f"{r['side']}/{r['position']}", []).append(r["dx"])
    params = {
        "transitions": trans,
        "first_labels": {k: v / sum(first.values()) for k, v in first.items()},
        "offsets": {k: sorted(v) for k, v in offsets.items()},
        "train_docs": train_docs,
    }
    path = path or PARAMS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=1), encoding="utf-8", newline="\n")
    return params


@lru_cache(maxsize=1)
def load_params() -> dict:
    if not PARAMS_PATH.exists():
        return fit_params(read_split("train"))
    return json.loads(PARAMS_PATH.read_text(encoding="utf-8"))


def kde_logpdf(samples: np.ndarray, x: float, bw: float = 0.25) -> float:
    if len(samples) == 0:
        return math.log(1e-4)
    z = (x - samples) / bw
    p = np.exp(-0.5 * z * z).sum() / (len(samples) * bw * math.sqrt(2 * math.pi))
    return math.log(max(p, 1e-6))


def gauss_logpdf(mu: float, sigma: float, x: float) -> float:
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


# ------------------------------------------------------------------ Stage 4


def read_system(page16: np.ndarray, ps, reader) -> list[dict]:
    """Candidates of one system with their OCR hypotheses (non-numbers removed)."""
    im16 = strip16(page16, ps.strip)
    cands = text_candidates(im16, ps.strip)
    hyps = reader([ocr_input(crop(im16, c)) for c in cands])
    out = []
    for c, h in zip(cands, hyps):
        p_empty = sum(q for s, q in h if s == "")
        h = [(s, q) for s, q in h if s and (parse_label(s) is not None or expand_range(s) is not None)]
        if p_empty > NOT_A_NUMBER or not h:
            continue
        out.append({"cand": c, "hyps": h})
    return merge_suffixes(out)


def merge_suffixes(readings: list[dict], max_gap: float = 1.2) -> list[dict]:
    """Join a number reading and a following single-letter reading printed with a space (`30 E`).

    The candidate grouping splits them because the gap exceeds 0.6 s. Merged hypotheses are the
    concatenations of both top-3 lists, with product probabilities.
    """
    readings = sorted(readings, key=lambda r: r["cand"].u0)
    out: list[dict] = []
    for r in readings:
        prev = out[-1] if out else None
        top = r["hyps"][0][0]
        if (
            prev is not None
            and len(top) == 1
            and top.isalpha()
            and prev["hyps"][0][0].isdigit()
            and prev["cand"].side == r["cand"].side
            and 0 <= r["cand"].u0 - prev["cand"].u1 <= max_gap * S
            and min(prev["cand"].v1, r["cand"].v1) - max(prev["cand"].v0, r["cand"].v0) > 0
        ):
            a, b = prev["cand"], r["cand"]
            hyps = sorted(((x + y, qx * qy) for x, qx in prev["hyps"][:3] for y, qy in r["hyps"][:3]), key=lambda t: -t[1])[:5]
            merged = Candidate(a.u0, b.u1, min(a.v0, b.v0), max(a.v1, b.v1), a.side, a.n_components + b.n_components)
            out[-1] = {"cand": merged, "hyps": hyps}
        else:
            out.append(r)
    return out


def offset_cost(dx: float, side: str, pos: str, doc_model: dict | None) -> float:
    key = f"{side}/{pos}"
    samples = np.asarray(load_params()["offsets"].get(key, []))
    lp = kde_logpdf(samples, dx) if len(samples) else math.log(1e-4)
    if doc_model and key in doc_model:
        mu, sg = doc_model[key]
        lp = math.log(0.2 * math.exp(lp) + 0.8 * math.exp(gauss_logpdf(mu, sg, dx)) + 1e-12)
    return -lp


def assign_system(readings: list[dict], us: list[float], doc_model: dict | None) -> dict[int, dict]:
    """Hungarian assignment of readings to measures of one system. Returns measure index -> reading."""
    n_m = len(us) - 1
    if n_m <= 0 or not readings:
        return {}
    cost = np.full((len(readings), n_m + len(readings)), 1e6)
    for r, rd in enumerate(readings):
        c = rd["cand"]
        is_range = expand_range(rd["hyps"][0][0]) is not None
        for m in range(n_m):
            if is_range:
                # Range labels are centred over the multi-measure rest.
                inside = us[m] - S <= c.uc <= us[m + 1] + S
                cost[r, m] = 1.0 if inside else 1e6
            else:
                dx = (c.uc - us[m]) / S
                cost[r, m] = offset_cost(dx, c.side, "first" if m == 0 else "other", doc_model)
        cost[r, n_m + r] = UNASSIGNED_COST
    ri, ci = linear_sum_assignment(cost)
    return {int(c): {**readings[r], "cost": float(cost[r, c])} for r, c in zip(ri, ci) if c < n_m and cost[r, c] < UNASSIGNED_COST}


def fit_doc_model(assigned: list[tuple[str, str, float]]) -> dict:
    """Per (side, position): robust Gaussian of the document's first-pass offsets."""
    by: dict[str, list[float]] = {}
    for side, pos, dx in assigned:
        by.setdefault(f"{side}/{pos}", []).append(dx)
    model = {}
    for k, v in by.items():
        if len(v) >= 5:
            v = np.asarray(v)
            med = float(np.median(v))
            model[k] = (med, max(0.3, 1.4826 * float(np.median(np.abs(v - med)))))
    return model


def split_measure(pm: PredMeasure, n: int) -> list[PredMeasure]:
    """Split a measure into n equal measures (a multi-measure rest)."""
    out = []
    q = pm.quad
    for k in range(n):
        a, b = k / n, (k + 1) / n
        lerp = lambda p0, p1, t: (p0[0] + (p1[0] - p0[0]) * t, p0[1] + (p1[1] - p0[1]) * t)  # noqa: E731
        quad = [lerp(q[0], q[1], a), lerp(q[0], q[1], b), lerp(q[3], q[2], b), lerp(q[3], q[2], a)] if q else []
        out.append(PredMeasure(x0=pm.x0 + (pm.x1 - pm.x0) * a, x1=pm.x0 + (pm.x1 - pm.x0) * b, confidence=pm.confidence, quad=quad))
    return out


# ------------------------------------------------------------------ Stage 5


def decode_labels(observations: list[list[tuple[str, float]] | None]) -> list[tuple[str, float]]:
    """Viterbi + forward-backward over the pruned candidate lattice.

    observations[i]: OCR hypotheses assigned to measure i, or None. Returns (label, posterior) per measure.
    """
    params = load_params()
    lt = {t: math.log(p) for t, p in params["transitions"].items()}
    lt["free_jump"] -= math.log(FREE_JUMP_TARGETS)
    first = params["first_labels"]
    n = len(observations)
    if n == 0:
        return []

    def emission(i: int, lab: str) -> float:
        obs = observations[i]
        if not obs:
            return 0.0
        for s, q in obs:
            if s == lab:
                return math.log(max(q, EMISSION_FLOOR))
        return math.log(EMISSION_FLOOR)

    # Candidate sets and forward (max and sum) scores.
    states: list[list[str]] = []
    vit: list[np.ndarray] = []
    fwd: list[np.ndarray] = []
    back: list[np.ndarray] = []
    trans_cache: list[np.ndarray] = []
    for i in range(n):
        cand = []
        if observations[i]:
            cand += [s for s, _ in observations[i]]
        if i == 0:
            cand += sorted(first.keys())  # sorted: key order must not depend on the JSON loader
        else:
            for p in states[-1]:
                cand += [s for s, _ in successors(p)]
            cand.append("1")
        cand = list(dict.fromkeys(c for c in cand if parse_label(c) is not None))
        em = np.array([emission(i, c) for c in cand])
        if i == 0:
            prior = np.array([math.log(first.get(c, 1e-3)) for c in cand])
            v = prior + em
            f = v.copy()
            bp = np.zeros(len(cand), int)
            tm = np.zeros((0, len(cand)))
        else:
            prev = states[-1]
            tm = np.array([[lt[transition_type(p, c)] for c in cand] for p in prev])
            sv = vit[-1][:, None] + tm
            bp = sv.argmax(axis=0)
            v = sv.max(axis=0) + em
            sf = fwd[-1][:, None] + tm
            f = np.logaddexp.reduce(sf, axis=0) + em
        # Prune to the beam by Viterbi score.
        keep = np.argsort(-v, kind="stable")[:BEAM]
        keep.sort()
        states.append([cand[k] for k in keep])
        vit.append(v[keep])
        fwd.append(f[keep])
        back.append(bp[keep])
        trans_cache.append(tm[:, keep] if i > 0 else tm)
    # Backtrack.
    path = [int(np.argmax(vit[-1]))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i][path[-1]]))
    path.reverse()
    # Backward pass on the pruned lattice for posteriors.
    bwd = [np.zeros(len(s)) for s in states]
    for i in range(n - 2, -1, -1):
        em = np.array([emission(i + 1, c) for c in states[i + 1]])
        bwd[i] = np.logaddexp.reduce(trans_cache[i + 1] + (em + bwd[i + 1])[None, :], axis=1)
    z = np.logaddexp.reduce(fwd[-1])
    out = []
    for i in range(n):
        post = math.exp(float(fwd[i][path[i]] + bwd[i][path[i]] - z))
        out.append((states[i][path[i]], min(1.0, post)))
    return out


# ------------------------------------------------------------------ document driver


def label_document(doc_id: str, per_page: list, reader_ckpt) -> None:
    """Stages 4 and 5 for one document; fills PredMeasure.label and label_confidence in place.

    per_page: [(page, [PredSystem], [PageSystem], PageGeometry)], each PageSystem with `us`
    (boundary strip columns of its measures). reader_ckpt: LabelOCR checkpoint path or a reader.
    """
    from .infer import LabelReader

    reader = LabelReader(reader_ckpt) if isinstance(reader_ckpt, str) else reader_ckpt
    systems = []  # (pred system, page system, readings)
    for _, preds, pss, geo in per_page:
        page16 = geo.norm16()
        for pred, ps in zip(preds, pss):
            if pred.measures:
                systems.append((pred, ps, read_system(page16, ps, reader)))
    # Pass A: global offset densities; pass B: document-specific placement.
    assigned = [assign_system(rd, ps.us, None) for _, ps, rd in systems]
    samples = []
    for (_, ps, _), asg in zip(systems, assigned):
        for m, r in asg.items():
            if expand_range(r["hyps"][0][0]) is None:
                samples.append((r["cand"].side, "first" if m == 0 else "other", (r["cand"].uc - ps.us[m]) / S))
    doc_model = fit_doc_model(samples)
    if doc_model:
        assigned = [assign_system(rd, ps.us, doc_model) for _, ps, rd in systems]
    # Multi-measure rests: split measures whose reading is a range label.
    observations: list[list[tuple[str, float]] | None] = []
    targets: list[PredMeasure] = []
    for (pred, _, _), asg in zip(systems, assigned):
        new_measures = []
        for m, pm in enumerate(pred.measures):
            r = asg.get(m)
            rng = expand_range(r["hyps"][0][0]) if r else None
            if rng and r["hyps"][0][1] > 0.5 and len(rng) >= 2:
                # The range label names every measure of the multi-measure rest.
                parts = split_measure(pm, len(rng))
                new_measures += parts
                observations += [[(lab, r["hyps"][0][1])] for lab in rng]
                targets += parts
            else:
                new_measures.append(pm)
                obs = [(s, q) for s, q in r["hyps"] if expand_range(s) is None] if r else None
                observations.append(obs or None)
                targets.append(pm)
        pred.measures = new_measures
    for pm, (lab, conf) in zip(targets, decode_labels(observations)):
        pm.label = lab
        pm.label_confidence = conf
