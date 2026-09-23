"""Page-level pipeline: prepared page -> systems -> strips -> measures (page coordinates).

The boundary detector and the label stages are plugged in by the caller, so the same code runs
the classical baseline and the learned pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluate import PredMeasure, PredSystem
from .geometry import PageGeometry, load_norm8
from .rectify import Strip, make_strip
from .staves import System, detect_systems


@dataclass
class PageSystem:
    system: System
    strip: Strip
    pred: PredSystem


def page_systems(doc_id: str, page: int) -> tuple[list[PageSystem], PageGeometry, np.ndarray, np.ndarray]:
    geo = PageGeometry.load(doc_id, page)
    gray = load_norm8(doc_id, page)
    if not geo.info["has_staff"]:
        return [], geo, gray, np.zeros_like(gray)
    systems, bw = detect_systems(gray)
    out = []
    for sy in systems:
        strip = make_strip(sy, gray.shape[1], gray)
        out.append(PageSystem(sy, strip, to_page_system(geo, page, sy)))
    return out, geo, gray, bw


def to_page_system(geo: PageGeometry, page: int, sy: System) -> PredSystem:
    xc = sy.xc
    yc = (float(sy.y_top(xc)) + float(sy.y_bottom(xc))) / 2
    (px0, px1), _ = geo.norm_to_page([sy.x0, sy.x1], [yc, yc])
    _, (py0, py1) = geo.norm_to_page([xc, xc], [float(sy.y_top(xc)), float(sy.y_bottom(xc))])
    return PredSystem(page=page, x0=float(px0), x1=float(px1), y0=float(py0), y1=float(py1))


def strip_u_to_page_x(geo: PageGeometry, strip: Strip, u: np.ndarray) -> np.ndarray:
    """Page x (fraction) of strip columns, measured on the system's vertical centre line."""
    v = np.full_like(np.asarray(u, np.float64), strip.margin + strip.height_nom / 2)
    x, y = strip.to_norm(u, v)
    px, _ = geo.norm_to_page(x, y)
    return px


def page_x_to_strip_u(geo: PageGeometry, strip: Strip, px: np.ndarray, pred: PredSystem) -> np.ndarray:
    """Inverse of strip_u_to_page_x (on the system's centre line, iterated once for the contour)."""
    py = np.full_like(np.asarray(px, np.float64), (pred.y0 + pred.y1) / 2)
    x, y = geo.page_to_norm(px, py)
    u, _ = strip.from_norm(x, y)
    return u


def measures_from_boundaries(geo: PageGeometry, strip: Strip, pred: PredSystem, us: list[float], confs: list[float]) -> list[PredMeasure]:
    """Build measures between consecutive boundaries (strip columns) with page-fraction quads."""
    out = []
    if len(us) < 2:
        return out
    xs = strip_u_to_page_x(geo, strip, np.asarray(us))
    vt, vb = strip.margin, strip.margin + strip.height_nom
    for k in range(len(us) - 1):
        corners = [(us[k], vt), (us[k + 1], vt), (us[k + 1], vb), (us[k], vb)]
        quad = []
        for u, v in corners:
            nx, ny = strip.to_norm(u, v)
            px, py = geo.norm_to_page(nx, ny)
            quad.append((float(px), float(py)))
        out.append(PredMeasure(x0=float(xs[k]), x1=float(xs[k + 1]), confidence=float(min(confs[k], confs[k + 1])), quad=quad))
    return out
