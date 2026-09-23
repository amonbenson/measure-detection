"""Coordinate frames of a prepared page.

- page frame: fractions of the displayed page, origin top-left (the JSON convention)
- render frame: pixels of the pass-2 render
- norm frame: pixels of the normalized, deskewed page at s = 8 (norm16 image is 2x)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .data import DERIVED

NORM8_S = 8.0


@dataclass
class PageGeometry:
    doc_id: str
    page: int
    info: dict
    a16: np.ndarray  # 2x3 render -> norm16

    @classmethod
    def load(cls, doc_id: str, page: int) -> "PageGeometry":
        info = json.loads((DERIVED / "pages" / doc_id / f"{page:03d}.json").read_text(encoding="utf-8"))
        return cls(doc_id, page, info, np.array(info["norm_matrix"], np.float64))

    @property
    def norm8_size(self) -> tuple[int, int]:
        return (self.info["norm_width"] + 1) // 2, (self.info["norm_height"] + 1) // 2

    @property
    def s_original(self) -> float:
        """Staff space in page-fraction units of the page width (for tolerances in page units)."""
        return self.info["s_render"] / self.info["render_width"]

    def page_to_norm(self, u, v) -> tuple[np.ndarray, np.ndarray]:
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        x = u * self.info["render_width"]
        y = v * self.info["render_height"]
        a = self.a16
        return (a[0, 0] * x + a[0, 1] * y + a[0, 2]) / 2, (a[1, 0] * x + a[1, 1] * y + a[1, 2]) / 2

    def norm_to_page(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        x, y = np.asarray(x, np.float64) * 2, np.asarray(y, np.float64) * 2
        inv = cv2.invertAffineTransform(self.a16)
        rx = inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]
        ry = inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]
        return rx / self.info["render_width"], ry / self.info["render_height"]


@lru_cache(maxsize=8)
def load_norm16(doc_id: str, page: int) -> np.ndarray:
    return cv2.imread(str(DERIVED / "pages" / doc_id / f"{page:03d}.png"), cv2.IMREAD_GRAYSCALE)


def load_norm8(doc_id: str, page: int) -> np.ndarray:
    g = load_norm16(doc_id, page)
    return cv2.resize(g, ((g.shape[1] + 1) // 2, (g.shape[0] + 1) // 2), interpolation=cv2.INTER_AREA)
