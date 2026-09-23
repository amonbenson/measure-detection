"""Stage 3: boundary signals -> boundary positions (guideline section 4.6), and the classical baseline.

A boundary detector is a function (strip image, strip) -> list of (u, confidence), sorted by u,
where the first entry is the start boundary.
"""

from __future__ import annotations

import cv2
import numpy as np

from .normalize import binarize
from .rectify import Strip
from .staves import S


def peaks_1d(signal: np.ndarray, threshold: float, min_dist: float) -> list[int]:
    """Local maxima above threshold, greedy non-maximum suppression with minimum distance."""
    idx = np.nonzero(signal >= threshold)[0]
    if len(idx) == 0:
        return []
    order = idx[np.argsort(-signal[idx], kind="stable")]
    taken: list[int] = []
    blocked = np.zeros(len(signal), bool)
    r = int(np.ceil(min_dist))
    for i in order:
        if blocked[i]:
            continue
        taken.append(int(i))
        blocked[max(0, i - r) : i + r + 1] = True
    return sorted(taken)


def decode(start: np.ndarray, barline: np.ndarray, offset: np.ndarray | None, stride: int, tau_b: float, min_dist: float = 1.5 * S) -> list[tuple[float, float]]:
    """Per-column probabilities (at the given column stride) -> [(u, conf)], start boundary first.

    The start boundary is the strongest start peak. Barlines are NMS peaks above tau_b right of it.
    """

    # Column i covers strip pixels [i * stride, (i + 1) * stride); offset o in [0, 1) locates u = (i + o) * stride.
    def pos(i: int) -> float:
        if offset is None:
            return (i + 0.5) * stride - 0.5
        return (i + float(offset[i])) * stride

    si = int(np.argmax(start))
    out = [(pos(si), float(start[si]))]
    for i in peaks_1d(barline, tau_b, min_dist / stride):
        u = pos(i)
        if u > out[0][0] + min_dist:
            out.append((u, float(barline[i])))
    return out


def classical_column_signal(img: np.ndarray, strip: Strip) -> np.ndarray:
    """Barline evidence per column from geometry only.

    evidence = fraction of staves whose band is covered >= 90% by ink in this column (+-1 px),
    set to 0 where the ink between the staff lines is wider than 0.5 s (noteheads, beams, text).
    """
    bw = binarize(img)
    w = bw.shape[1]
    wide = bw.copy()
    wide[:, 1:] |= bw[:, :-1]
    wide[:, :-1] |= bw[:, 1:]
    # Ink that belongs to a horizontal run longer than 0.5 s.
    k = int(0.5 * S) + 2
    broad = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, k), np.uint8))
    covered = []
    thin = np.ones(w, bool)
    for vt, vb in strip.staff_bands:
        a, b = int(round(vt)), int(round(vb)) + 1
        covered.append(wide[a:b].mean(axis=0) >= 0.9)
        rows = [int(round(vt + (j + 0.5) * (vb - vt) / 4)) for j in range(4)]
        thin &= ~broad[rows].any(axis=0)
    ev = np.mean(covered, axis=0) if covered else np.zeros(w)
    return ev * thin


def classical_detector(tau: float = 0.75):
    """Classical baseline: start = the system's left end; barlines = peaks of the column evidence."""

    def detect(img: np.ndarray, strip: Strip) -> list[tuple[float, float]]:
        sig = classical_column_signal(img, strip)
        out = [(float(strip.system_u0), 1.0)]
        for i in peaks_1d(sig, tau, 1.5 * S):
            if i > out[0][0] + 1.5 * S:
                out.append((float(i), float(sig[i])))
        return out

    return detect
