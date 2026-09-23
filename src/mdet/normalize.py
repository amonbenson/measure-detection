"""Stage 1: staff space estimation and deskew (guideline section 4.2). Classical only."""

from __future__ import annotations

import cv2
import numpy as np


def binarize(gray: np.ndarray) -> np.ndarray:
    """Otsu threshold. Returns uint8 with 1 = ink."""
    _, bw = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def vertical_runs(bw: np.ndarray, max_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histograms of vertical black runs, white runs, and black+following-white pair lengths."""
    col = bw.T.astype(np.int8)  # iterate columns as rows for contiguous access
    pad = np.zeros((col.shape[0], 1), np.int8)
    d = np.diff(np.concatenate([pad, col, pad], axis=1), axis=1)
    black_h = np.zeros(max_len + 1, np.int64)
    white_h = np.zeros(max_len + 1, np.int64)
    pair_h = np.zeros(2 * max_len + 1, np.int64)
    rows, cols = np.nonzero(d)
    vals = d[rows, cols]
    # Transitions alternate +1 (ink starts) and -1 (ink ends) per column.
    same = rows[1:] == rows[:-1]
    lens = cols[1:] - cols[:-1]
    kind = vals[:-1]  # +1: black run follows; -1: white run follows
    black = lens[same & (kind == 1)]
    white = lens[same & (kind == -1)]
    np.add.at(black_h, np.clip(black, 0, max_len), 1)
    np.add.at(white_h, np.clip(white, 0, max_len), 1)
    # Pairs: black run immediately followed by a white run in the same column.
    idx = np.nonzero(same & (kind == 1))[0]
    idx = idx[(idx + 1 < len(lens)) & np.append(same, False)[idx + 1]]
    pairs = lens[idx] + lens[idx + 1]
    np.add.at(pair_h, np.clip(pairs, 0, 2 * max_len), 1)
    black_h[max_len] = white_h[max_len] = pair_h[2 * max_len] = 0  # clipped overflow bins
    return black_h, white_h, pair_h


def _refined_mode(h: np.ndarray, lo: int) -> float:
    h = h.astype(np.float64).copy()
    h[:lo] = 0
    k = int(np.argmax(h))
    w = h[max(k - 1, 0) : k + 2]
    x = np.arange(max(k - 1, 0), k + 2)[: len(w)]
    return float((w * x).sum() / max(w.sum(), 1e-9))


def estimate_staff_space(gray: np.ndarray, max_len: int = 120) -> dict:
    """Staff space s = mode of (black run + next white run) over vertical runs.

    The piano staff dominates the histogram, so smaller cue staves do not shift the mode.
    Returns s, line thickness and the relative strength of the mode.
    """
    bw = binarize(gray)
    black_h, white_h, pair_h = vertical_runs(bw, max_len)
    thickness = _refined_mode(black_h, 1)
    s = _refined_mode(pair_h, 3)
    strength = float(pair_h[int(round(s))] / max(pair_h[3:].sum(), 1))
    return {"s": s, "line_thickness": thickness, "mode_strength": strength}


def horizontal_line_mask(bw: np.ndarray, s: float) -> np.ndarray:
    """Keep long horizontal ink runs (staff lines, beams, ledger lines are mostly removed)."""
    k = max(3, int(round(4 * s)))
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))


def estimate_skew(gray: np.ndarray, s: float, max_deg: float = 5.0) -> float:
    """Dominant angle of staff lines in degrees (positive = counter-clockwise in image coordinates).

    Maximizes the variance of the horizontal projection of the line mask over rotation angles,
    coarse to fine. Runs on a copy scaled to s = 4 px.
    """
    f = 4.0 / s
    small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    bw = binarize(small)
    # Opening with a horizontal element suppresses text and notes but keeps slightly tilted lines.
    lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))).astype(np.float32)
    h, w = lines.shape
    c = (w / 2, h / 2)

    def score(deg: float) -> float:
        m = cv2.getRotationMatrix2D(c, deg, 1.0)
        r = cv2.warpAffine(lines, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        p = r.sum(axis=1)
        return float((p * p).sum())

    best, step = 0.0, 0.5
    grid = np.arange(-max_deg, max_deg + 1e-9, step)
    while True:
        vals = [score(a) for a in grid]
        best = float(grid[int(np.argmax(vals))])
        if step <= 0.02:
            break
        step /= 4
        grid = np.arange(best - 4 * step, best + 4 * step + 1e-9, step)
    return best


def normalization_matrix(width: int, height: int, s: float, angle_deg: float, target_s: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Affine map (2x3) from rendered page pixels to the normalized image, and the output size.

    The normalized image is scaled to s = target_s and rotated by angle_deg about the page centre.
    The output canvas contains the whole rotated page.
    """
    f = target_s / s
    m = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, f)
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], np.float64)
    pts = corners @ m.T
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    m[:, 2] -= (x0, y0)
    return m, (int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))


def apply_normalization(gray: np.ndarray, m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return cv2.warpAffine(gray, m, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
