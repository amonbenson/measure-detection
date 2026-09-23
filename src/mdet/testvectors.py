"""Test vectors for the TypeScript port: Python results of the primitives on fixed inputs.

Writes packages/measure-detection/test/vectors.json (committed). The package's unit test
(test/unit.mjs) must reproduce every value exactly (images) or within 1e-9 (floats).
Usage: uv run python -m mdet.testvectors
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from .data import ROOT
from .geometry import halve
from .labelling import decode_labels
from .normalize import binarize, estimate_staff_space, resize_area
from .ocr import beam_search

OUT = ROOT / "packages" / "measure-detection" / "test" / "vectors.json"


def synthetic_page() -> np.ndarray:
    """A small page with two five-line staves (s = 9 px), a barline and some noise."""
    rng = np.random.default_rng(1)
    g = np.full((160, 220), 245, np.uint8)
    for top in (30, 100):
        for k in range(5):
            g[top + 9 * k : top + 9 * k + 2, 10:210] = 20
        g[top : top + 38, 120:122] = 25
    g[60:66, 40:48] = 30
    noise = rng.integers(0, 20, g.shape).astype(np.int16)
    return np.clip(g.astype(np.int16) - noise, 0, 255).astype(np.uint8)


def main() -> None:
    rng = np.random.default_rng(0)
    img = (rng.random((31, 43)) * 255).astype(np.uint8)
    img[5:20, 10:12] = 0
    page = synthetic_page()
    thr, _ = cv2.threshold(page, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = binarize(page)
    m = cv2.getRotationMatrix2D((21.5, 15.5), 3.0, 0.9)
    m[:, 2] += (1.3, -0.7)
    warped = cv2.warpAffine(img, m, (40, 30), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    f = (img > 128).astype(np.float32)
    sh = np.array([[1.0, 0.02, -0.02 * 15.0], [0.0, 1.0, 0.0]], np.float32)
    sheared = cv2.warpAffine(f, sh, (43, 31), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    mapx = (np.arange(43, dtype=np.float32)[None, :] * 0.93 + 0.37 + np.zeros((31, 1), np.float32)).astype(np.float32)
    mapy = (np.arange(31, dtype=np.float32)[:, None] * 1.07 - 0.21 + np.zeros((1, 43), np.float32)).astype(np.float32)
    remapped = cv2.remap(img, mapx, mapy, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    comps = sorted([list(map(int, stats[i])) for i in range(1, n)])
    xs, ys = [3.0, 10.0, 20.0, 31.0, 47.0, 60.0], [5.0, 4.2, 4.9, 6.1, 8.8, 11.5]
    lp = np.log(np.random.default_rng(2).dirichlet(np.ones(64) * 0.1, size=32)).astype(np.float32)
    obs = [[("1", 0.9)], None, [("3", 0.6), ("8", 0.3)], None, [("5A", 0.5), ("5", 0.4)], None, [("7", 0.2)], [("9", 0.8)]]
    vectors = {
        "img": {"width": 43, "height": 31, "data": img.flatten().tolist()},
        "page": {"width": 220, "height": 160, "data": page.flatten().tolist()},
        "otsu": float(thr),
        "binarize_sum": int(bw.sum()),
        "open_16x1": cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((1, 16), np.uint8)).flatten().tolist(),
        "open_1x16": cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((16, 1), np.uint8)).flatten().tolist(),
        "close_25x1": cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((1, 25), np.uint8)).flatten().tolist(),
        "resize_area_17x13": resize_area(img, 17, 13).flatten().tolist(),
        "resize_area_60x50": resize_area(img, 60, 50).flatten().tolist(),
        "halve": halve(img).flatten().tolist(),
        "warp_matrix": m.flatten().tolist(),
        "warp_affine": warped.flatten().tolist(),
        "warp_shear_f": sheared.flatten().tolist(),
        "remap": remapped.flatten().tolist(),
        "remap_map": {"ax": 0.93, "bx": 0.37, "ay": 1.07, "by": -0.21},
        "components": comps,
        "polyfit": {"x": xs, "y": ys, "deg2": np.polyfit(xs, ys, 2).tolist(), "deg1": np.polyfit(xs, ys, 1).tolist()},
        "staff_space": estimate_staff_space(page),
        "beam": {"logp": lp.flatten().tolist(), "result": [[s, q] for s, q in beam_search(lp)]},
        "decode": {"observations": [o if o is None else [[a, b] for a, b in o] for o in obs], "result": [[a, b] for a, b in decode_labels(obs)]},
    }
    OUT.write_text(json.dumps(vectors), encoding="utf-8", newline="\n")
    print(OUT, OUT.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
