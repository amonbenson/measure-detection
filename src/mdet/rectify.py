"""Stage 2: rectify a system into a straight horizontal strip (guideline section 4.4). Classical only.

Strip frame (s = 8 px): column u maps to page x = x_start + u + shear * (v - v_centre) (norm8 frame).
Row v maps linearly between the system's top contour (v = margin) and bottom contour
(v = margin + height). The shear makes vertical lines vertical when a scan is sheared (vertical
lines tilted while staff lines are level); it is estimated per system from the image.
The same mapping at scale 2 produces an s = 16 strip from the norm16 page.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .staves import S, System

MARGIN = 6 * S  # above the top line and below the bottom line
LEFT_PAD = 3 * S
RIGHT_PAD = 2 * S


@dataclass
class Strip:
    x_start: float  # norm8 x of strip column 0
    width: int
    margin: float
    height_nom: float  # nominal system height (top line to bottom line), px
    top: np.ndarray  # polynomial of the system top contour, norm8
    bottom: np.ndarray  # polynomial of the system bottom contour, norm8
    staff_bands: list[tuple[float, float]]  # (v_top, v_bottom) of each staff in strip rows
    staff_spacings: list[float]
    system_u0: float  # strip column of the system's left end (staff start)
    system_u1: float  # strip column of the system's right end
    shear: float = 0.0  # page x shift per strip row, relative to the centre row

    @property
    def v_centre(self) -> float:
        return self.margin + self.height_nom / 2

    @property
    def height(self) -> int:
        return int(round(self.height_nom + 2 * self.margin))

    def to_norm(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        x = self.x_start + u + self.shear * (v - self.v_centre)
        t, b = np.polyval(self.top, x), np.polyval(self.bottom, x)
        return x, t + (v - self.margin) * (b - t) / self.height_nom

    def from_norm(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        t, b = np.polyval(self.top, x), np.polyval(self.bottom, x)
        v = self.margin + (y - t) * self.height_nom / (b - t)
        return x - self.x_start - self.shear * (v - self.v_centre), v


def estimate_shear(image: np.ndarray, strip: Strip, max_shear: float = 0.035, step: float = 0.0025) -> float:
    """Shear that makes vertical line segments vertical in the strip.

    Maximizes the energy of the column projection of the vertical-line mask (ink runs >= 2 s tall)
    over candidate shears. Only the system rows are used.
    """
    img = rectify(image, strip)
    a, b = int(strip.margin), int(strip.margin + strip.height_nom)
    _, bw = cv2.threshold(img[a:b], 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((int(2 * S), 1), np.uint8)).astype(np.float32)
    h, w = vert.shape
    vc = strip.v_centre - a
    n = int(round(2 * max_shear / step)) + 1
    ks, scores = [], []
    for i in range(n):
        k = -max_shear + i * step
        # Sample column u + k * (v - vc) for every row: a line with that tilt becomes vertical.
        m = np.array([[1.0, k, -k * vc], [0.0, 1.0, 0.0]], np.float64)
        sheared = cv2.warpAffine(vert, m, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
        p = sheared.astype(np.float64).sum(axis=0)
        ks.append(k)
        scores.append(float((p * p).sum()))
    # Near-ties (within 0.5 %, below the measurement noise of one shear step) go to the smallest shear,
    # so that last-digit differences between implementations cannot change the result.
    best = max(scores)
    return min((k for k, sc in zip(ks, scores) if sc >= best * (1 - 5e-3)), key=abs)


def make_strip(system: System, page_width: int, image: np.ndarray | None = None) -> Strip:
    """Strip geometry for a system. If the page image is given, the shear is estimated from it."""
    xs0 = max(0.0, system.x0 - LEFT_PAD)
    xs1 = min(float(page_width), system.x1 + RIGHT_PAD)
    xc = system.xc
    top_c, bot_c = float(system.y_top(xc)), float(system.y_bottom(xc))
    h_nom = max(4.0, bot_c - top_c)
    k = h_nom / (bot_c - top_c)
    bands = [(MARGIN + (float(st.y_top(xc)) - top_c) * k, MARGIN + (float(st.y_bottom(xc)) - top_c) * k) for st in system.staves]
    strip = Strip(
        x_start=xs0,
        width=int(np.ceil(xs1 - xs0)),
        margin=MARGIN,
        height_nom=h_nom,
        top=np.asarray(system.staves[0].top, np.float64),
        bottom=np.asarray(system.staves[-1].bottom, np.float64),
        staff_bands=bands,
        staff_spacings=[st.spacing for st in system.staves],
        system_u0=system.x0 - xs0,
        system_u1=system.x1 - xs0,
    )
    if image is not None:
        strip.shear = estimate_shear(image, strip)
    return strip


def rectify(image: np.ndarray, strip: Strip, scale: int = 1) -> np.ndarray:
    """Sample the strip from a page image. scale = 1 for norm8, 2 for norm16 (output is 2x larger)."""
    w, h = strip.width * scale, strip.height * scale
    u = (np.arange(w, dtype=np.float64) + 0.5) / scale - 0.5
    v = (np.arange(h, dtype=np.float64) + 0.5) / scale - 0.5
    uu, vv = np.meshgrid(u, v)
    x, y = strip.to_norm(uu, vv)
    mx = ((x + 0.5) * scale - 0.5).astype(np.float32)
    my = ((y + 0.5) * scale - 0.5).astype(np.float32)
    return cv2.remap(image, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
