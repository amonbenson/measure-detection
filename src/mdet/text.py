"""Text candidates in the margins of a system strip. Classical only.

Replaces the guideline's number-text head (MeasureNet head 2). A candidate is a group of connected
components in the band above the top staff line or below the bottom staff line, merged when the
horizontal gap is at most 0.6 s. Candidates are cut from the s = 16 strip for LabelOCR.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .normalize import resize_area
from .rectify import Strip, rectify
from .staves import S

S16 = 2 * S  # staff space in the s = 16 strip
MERGE_GAP = 0.6
MIN_HEIGHT, MAX_HEIGHT = 0.7, 2.6  # height of the tallest glyph, in s
MAX_WIDTH = 10.0
STAFF_CLEARANCE = 0.25  # candidates must lie this far outside the outer staff lines, in s


@dataclass
class Candidate:
    u0: float  # strip columns (s = 8 frame)
    u1: float
    v0: float
    v1: float
    side: str  # "above" or "below"
    n_components: int

    @property
    def uc(self) -> float:
        return (self.u0 + self.u1) / 2


def strip16(page16: np.ndarray, strip: Strip) -> np.ndarray:
    return rectify(page16, strip, scale=2)


def text_candidates(img16: np.ndarray, strip: Strip) -> list[Candidate]:
    h, w = img16.shape
    _, bw = cv2.threshold(img16, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    top = int((strip.margin - STAFF_CLEARANCE * S) * 2)
    bot = int((strip.margin + strip.height_nom + STAFF_CLEARANCE * S) * 2)
    out: list[Candidate] = []
    for side, (a, b) in (("above", (0, top)), ("below", (bot, h))):
        if b - a < 4:
            continue
        band = bw[a:b]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(band, connectivity=8)
        comps = []
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < 3:
                continue
            # Components touching the staff side of the band continue into the staff (stems, slurs).
            if (side == "above" and y + ch >= b - a) or (side == "below" and y == 0):
                continue
            # Components cut by the strip's outer edge are incomplete.
            if (side == "above" and y == 0) or (side == "below" and y + ch >= b - a):
                continue
            if ch > MAX_HEIGHT * S16 or cw > MAX_WIDTH * S16:
                continue
            comps.append([x, y + a, x + cw, y + a + ch])
        comps.sort()
        groups: list[list[list[int]]] = []
        for c in comps:
            for g in groups:
                gx1 = max(k[2] for k in g)
                gy0, gy1 = min(k[1] for k in g), max(k[3] for k in g)
                ov = min(gy1, c[3]) - max(gy0, c[1])
                if c[0] - gx1 <= MERGE_GAP * S16 and ov > -0.2 * S16 and (ov > 0.3 * min(gy1 - gy0, c[3] - c[1]) or c[3] - c[1] < 0.4 * S16):
                    g.append(c)
                    break
            else:
                groups.append([c])
        for g in groups:
            tallest = max(k[3] - k[1] for k in g)
            x0, y0 = min(k[0] for k in g), min(k[1] for k in g)
            x1, y1 = max(k[2] for k in g), max(k[3] for k in g)
            if not MIN_HEIGHT * S16 <= tallest <= MAX_HEIGHT * S16 or x1 - x0 > MAX_WIDTH * S16:
                continue
            n_glyphs = sum(1 for k in g if k[3] - k[1] >= 0.5 * tallest)
            out.append(Candidate(x0 / 2, x1 / 2, y0 / 2, y1 / 2, side, n_glyphs))
    return out


def crop(img16: np.ndarray, c: Candidate, pad: float = 0.5) -> np.ndarray:
    """Candidate crop from the s = 16 strip with padding (in s)."""
    p = pad * S16
    h, w = img16.shape
    x0, x1 = max(0, int(c.u0 * 2 - p)), min(w, int(np.ceil(c.u1 * 2 + p)))
    y0, y1 = max(0, int(c.v0 * 2 - p)), min(h, int(np.ceil(c.v1 * 2 + p)))
    return img16[y0:y1, x0:x1]


def ocr_input(crop_img: np.ndarray, height: int = 32, width: int = 128) -> np.ndarray:
    """Resize to the LabelOCR input: height 32, width padded to 128 (squeezed if wider). Ink = 1."""
    h, w = crop_img.shape
    nw = max(1, min(width, int(round(w * height / max(h, 1)))))
    r = resize_area(crop_img, nw, height)
    out = np.zeros((height, width), np.float32)
    out[:, :nw] = 1.0 - r.astype(np.float32) / 255.0
    return out
