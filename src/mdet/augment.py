"""Augmentations (guideline section 6.2). Geometric ones transform the image and the targets together."""

from __future__ import annotations

import cv2
import numpy as np

from .staves import S


def photometric(img: np.ndarray, rng: np.random.Generator, other: np.ndarray | None = None) -> np.ndarray:
    """Scan-like degradations on a grayscale uint8 image (255 = paper)."""
    x = img.astype(np.float32) / 255.0
    h, w = x.shape
    if other is not None and rng.random() < 0.2:  # bleed-through: mirrored, faded copy of another image
        o = cv2.resize(other, (w, h)).astype(np.float32) / 255.0
        a = rng.uniform(0.05, 0.15)
        x = np.minimum(x, 1 - a * (1 - o[:, ::-1]))
    if rng.random() < 0.3:  # uneven illumination (linear gradient)
        g = np.linspace(0, 1, w if rng.random() < 0.5 else h, dtype=np.float32)
        g = g[None, :] if g.shape[0] == w else g[:, None]
        if rng.random() < 0.5:
            g = 1 - g
        x = x * (1 - rng.uniform(0, 0.4) * g)
    if rng.random() < 0.8:  # contrast, brightness, gamma
        lo, hi = rng.uniform(0.0, 0.25), rng.uniform(0.8, 1.0)
        x = lo + (hi - lo) * np.clip(x, 0, 1) ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.2:  # binarization artefacts: erosion or dilation of ink by 1 px
        k = np.ones((2, 2), np.uint8)
        x = cv2.erode(x, k) if rng.random() < 0.5 else cv2.dilate(x, k)
    if rng.random() < 0.4:  # blur
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.0))
    if rng.random() < 0.4:  # gaussian noise
        x = x + rng.normal(0, rng.uniform(0.01, 0.08), x.shape).astype(np.float32)
    if rng.random() < 0.2:  # salt and pepper
        m = rng.random(x.shape)
        p = rng.uniform(0.0005, 0.005)
        x[m < p] = 0
        x[m > 1 - p] = 1
    out = np.clip(x * 255, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG
        ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(30, 96))])
        out = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return out


def geometric_strip(img: np.ndarray, bands: np.ndarray, bounds: np.ndarray, rng: np.random.Generator):
    """Scale jitter (+-15%), small rotation (+-0.5 deg) and vertical shift (+-0.5 s) of a strip.

    bounds are x positions (strip columns); bands are (v_top, v_bottom) rows. The rotation is about
    the system centre, so boundary x on the centre line is unchanged.
    """
    h, w = img.shape
    f = rng.uniform(0.85, 1.15)
    ang = rng.uniform(-0.5, 0.5)
    dy = rng.uniform(-0.5, 0.5) * S
    vc = float(bands[:, 0].min() + bands[:, 1].max()) / 2
    m = cv2.getRotationMatrix2D((w / 2, vc), ang, f)
    m[1, 2] += dy
    nw, nh = int(round(w * f)), int(round(h * f))
    m[0, 2] += (nw - w) / 2
    m[1, 2] += (nh - h) / 2
    out = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=255)
    # Map x on the centre line exactly through the affine matrix.
    nb = m[0, 0] * bounds + m[0, 1] * vc + m[0, 2]
    nbands = np.stack([m[1, 0] * (w / 2) + m[1, 1] * bands[:, 0] + m[1, 2], m[1, 0] * (w / 2) + m[1, 1] * bands[:, 1] + m[1, 2]], axis=1)
    return out, nbands, nb


def crop_strip(img: np.ndarray, bounds: np.ndarray, keep: np.ndarray, rng: np.random.Generator, min_frac: float = 0.5):
    """Random horizontal crop (partial systems). Returns image, shifted bounds and a keep mask."""
    h, w = img.shape
    cw = int(w * rng.uniform(min_frac, 1.0))
    x0 = int(rng.integers(0, w - cw + 1))
    nb = bounds - x0
    inside = (nb >= 0) & (nb < cw)
    return img[:, x0 : x0 + cw], nb, keep & inside
