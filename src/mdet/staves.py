"""Staff detection and system grouping on a normalized page (s = 8 px). Classical only.

Replaces the guideline's SystemNet (Model A):
1. Staff lines are found per vertical slab from the projection of a horizontal-line mask.
2. Five equidistant lines form a staff detection; detections are linked across slabs into staves.
3. Adjacent staves belong to one system if a vertical line (the system's opening line or bracket)
   connects them at their left end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .normalize import binarize

S = 8.0  # staff space of the normalized page used here


@dataclass
class Staff:
    x0: float
    x1: float
    top: np.ndarray  # polynomial coefficients (np.polyval) of the top line y(x)
    bottom: np.ndarray  # polynomial coefficients of the bottom line y(x)
    spacing: float  # distance between adjacent lines, px
    support: int = 1  # number of slab detections

    def y_top(self, x) -> np.ndarray:
        return np.polyval(self.top, x)

    def y_bottom(self, x) -> np.ndarray:
        return np.polyval(self.bottom, x)

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


@dataclass
class System:
    staves: list[Staff]
    x0: float = field(init=False)
    x1: float = field(init=False)

    def __post_init__(self) -> None:
        self.x0 = min(st.x0 for st in self.staves)
        self.x1 = max(st.x1 for st in self.staves)

    def y_top(self, x) -> np.ndarray:
        return self.staves[0].y_top(x)

    def y_bottom(self, x) -> np.ndarray:
        return self.staves[-1].y_bottom(x)

    @property
    def xc(self) -> float:
        return (self.x0 + self.x1) / 2


def _line_centers(profile: np.ndarray, thr: float, max_width: float = 0.35 * S) -> list[float]:
    """Centres of rows where the profile exceeds thr. Runs thicker than max_width (beams) are skipped."""
    on = profile >= thr
    centers = []
    i, n = 0, len(profile)
    while i < n:
        if on[i]:
            j = i
            while j + 1 < n and on[j + 1]:
                j += 1
            w = profile[i : j + 1]
            if j + 1 - i <= max_width:
                centers.append(float((np.arange(i, j + 1) * w).sum() / w.sum()))
            i = j + 1
        else:
            i += 1
    return centers


def _five_line_groups(ys: list[float]) -> list[list[float]]:
    groups, i = [], 0
    while i + 5 <= len(ys):
        g = ys[i : i + 5]
        d = np.diff(g)
        med = float(np.median(d))
        if 0.55 * S <= med <= 1.35 * S and np.abs(d - med).max() <= 0.15 * med + 0.5:
            groups.append(g)
            i += 5
        else:
            i += 1
    return groups


def detect_staves(gray: np.ndarray) -> tuple[list[Staff], np.ndarray]:
    """Returns staves sorted top to bottom, and the binarized page (1 = ink)."""
    bw = binarize(gray)
    h, w = bw.shape
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(2 * S), 1)))
    slab = int(12 * S)
    step = slab // 2
    dets = []  # (x_center, line ys)
    for x in range(0, max(1, w - slab + 1), step):
        prof = lines[:, x : x + slab].mean(axis=1)
        for g in _five_line_groups(_line_centers(prof, 0.5)):
            dets.append((x + slab / 2, g))

    # Link detections into tracks: same staff in neighbouring slabs has nearly the same top and spacing.
    tracks: list[list[tuple[float, list[float]]]] = []
    for xc, g in sorted(dets, key=lambda d: d[0]):
        sp = (g[-1] - g[0]) / 4
        best = None
        for t in tracks:
            lx, lg = t[-1]
            lsp = (lg[-1] - lg[0]) / 4
            if 0 < xc - lx <= 2.5 * step and abs(lg[0] - g[0]) < 1.0 * S and abs(lsp - sp) < 0.2 * sp:
                if best is None or abs(lg[0] - g[0]) < abs(best[-1][1][0] - g[0]):
                    best = t
        if best is None:
            tracks.append([(xc, g)])
        else:
            best.append((xc, g))

    staves = []
    for t in tracks:
        xs = np.array([d[0] for d in t])
        tops = np.array([d[1][0] for d in t])
        bots = np.array([d[1][4] for d in t])
        sp = float(np.median((bots - tops) / 4))
        deg = 0 if len(t) < 3 else (1 if len(t) < 6 else 2)
        ptop, pbot = np.polyfit(xs, tops, deg), np.polyfit(xs, bots, deg)
        # Exact horizontal extent: columns where at least 4 of the 5 lines have ink.
        cols = np.arange(w)
        cnt = np.zeros(w, np.int32)
        for k in range(5):
            y = np.polyval(ptop, cols) + k * (np.polyval(pbot, cols) - np.polyval(ptop, cols)) / 4
            yi = np.clip(np.round(y).astype(int), 1, h - 2)
            cnt += (lines[yi, cols] | lines[yi - 1, cols] | lines[yi + 1, cols]).astype(np.int32)
        present = cnt >= 4
        # Close gaps up to 3 s (clefs and dense chords hide lines), then take the run that contains the detections.
        present = cv2.morphologyEx(present.astype(np.uint8)[None], cv2.MORPH_CLOSE, np.ones((1, int(3 * S) + 1), np.uint8))[0].astype(bool)
        anchor = int(np.median(xs))
        if not present[anchor]:
            near = np.nonzero(present)[0]
            if len(near) == 0:
                continue
            anchor = int(near[np.argmin(np.abs(near - anchor))])
        x0 = anchor
        while x0 > 0 and present[x0 - 1]:
            x0 -= 1
        x1 = anchor
        while x1 < w - 1 and present[x1 + 1]:
            x1 += 1
        if x1 - x0 < 8 * S:
            continue
        staves.append(Staff(x0=float(x0), x1=float(x1 + 1), top=ptop, bottom=pbot, spacing=sp, support=len(t)))
    staves.sort(key=lambda st: float(st.y_top(st.xc)))
    return _dedupe(staves), bw


def _dedupe(staves: list[Staff]) -> list[Staff]:
    """Resolve staves that overlap vertically over a common x range.

    Two tracks of the same staff (split by a gap) are merged. A track shifted by one line (a beam
    taken as a staff line) overlaps a better-supported track and is dropped.
    """
    out: list[Staff] = []
    for st in sorted(staves, key=lambda t: -t.support):
        keep = True
        for o in out:
            xa, xb = max(o.x0, st.x0), min(o.x1, st.x1)
            xm = (xa + xb) / 2 if xb > xa else (st.xc if abs(st.xc - o.xc) < abs(o.xc - st.xc) else o.xc)
            ot, ob = float(o.y_top(xm)), float(o.y_bottom(xm))
            tt, tb = float(st.y_top(xm)), float(st.y_bottom(xm))
            if min(ob, tb) - max(ot, tt) > 0.3 * (tb - tt):
                if abs(ot - tt) < 1.0 * S:
                    o.x0, o.x1 = min(o.x0, st.x0), max(o.x1, st.x1)
                keep = False
                break
        if keep:
            out.append(st)
    out.sort(key=lambda t: float(t.y_top(t.xc)))
    return out


def connected_at_left(a: Staff, b: Staff, bw: np.ndarray, min_cover: float = 0.8) -> bool:
    """True if a vertical line joins staff a (above) and staff b (below) near their left ends."""
    h, w = bw.shape
    best = 0.0
    for x in range(int(min(a.x0, b.x0) - 2 * S), int(max(a.x0, b.x0) + 1.5 * S) + 1):
        if not 0 <= x < w:
            continue
        y0, y1 = int(round(float(a.y_bottom(x)))), int(round(float(b.y_top(x))))
        if y1 - y0 < 2:
            return True
        # Allow a line that is up to 2 px wide or slightly tilted.
        col = bw[y0:y1, max(0, x - 1) : x + 2].max(axis=1)
        best = max(best, float(col.mean()))
    return best >= min_cover


def group_systems(staves: list[Staff], bw: np.ndarray) -> list[System]:
    systems: list[list[Staff]] = []
    for st in staves:
        if systems:
            prev = systems[-1][-1]
            overlap = min(prev.x1, st.x1) - max(prev.x0, st.x0)
            if overlap > 0.5 * min(prev.x1 - prev.x0, st.x1 - st.x0) and connected_at_left(prev, st, bw):
                systems[-1].append(st)
                continue
        systems.append([st])
    return [System(staves=g) for g in systems]


def detect_systems(gray: np.ndarray) -> tuple[list[System], np.ndarray]:
    staves, bw = detect_staves(gray)
    return group_systems(staves, bw), bw
