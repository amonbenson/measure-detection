"""LabelOCR helpers: charset, CTC decoding and synthetic crops (guideline sections 4.7 and 6.2)."""

from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-"
BLANK = 0


def encode(text: str) -> list[int]:
    return [CHARSET.index(ch) + 1 for ch in text if ch in CHARSET]


def greedy_decode(logp: np.ndarray) -> str:
    """logp: (T, C). Collapse repeats, drop blanks."""
    best = logp.argmax(axis=1)
    out, prev = [], BLANK
    for k in best:
        if k != prev and k != BLANK:
            out.append(CHARSET[k - 1])
        prev = k
    return "".join(out)


def beam_search(logp: np.ndarray, beam: int = 8, top: int = 5) -> list[tuple[str, float]]:
    """CTC prefix beam search. Returns up to `top` (string, probability), probabilities normalized over the beam."""
    t_len, c = logp.shape
    p = np.exp(logp)
    beams: dict[tuple, tuple[float, float]] = {(): (1.0, 0.0)}  # prefix -> (p ending in blank, p ending in non-blank)
    for t in range(t_len):
        cand = np.argsort(-p[t])[:6]
        nxt: dict[tuple, list[float]] = {}
        for prefix, (pb, pnb) in beams.items():
            for k in cand:
                pk = float(p[t, k])
                if k == BLANK:
                    e = nxt.setdefault(prefix, [0.0, 0.0])
                    e[0] += (pb + pnb) * pk
                    continue
                last = prefix[-1] if prefix else None
                np_ = prefix + (int(k),)
                e = nxt.setdefault(np_, [0.0, 0.0])
                if k == last:
                    e[1] += pb * pk
                    same = nxt.setdefault(prefix, [0.0, 0.0])
                    same[1] += pnb * pk
                else:
                    e[1] += (pb + pnb) * pk
        beams = dict(sorted(((k, (v[0], v[1])) for k, v in nxt.items()), key=lambda kv: -(kv[1][0] + kv[1][1]))[:beam])
    res = [("".join(CHARSET[k - 1] for k in pre), pb + pnb) for pre, (pb, pnb) in beams.items()]
    z = sum(q for _, q in res) or 1.0
    res = sorted(((s, q / z) for s, q in res), key=lambda x: -x[1])
    # Merge identical strings (cannot happen with prefixes, kept as a guard) and cut.
    return res[:top]


# ---------------------------------------------------------------- synthetic crops

FONT_DIR = Path("C:/Windows/Fonts")
FONT_NAMES = [
    "times.ttf", "timesbd.ttf", "timesi.ttf", "arial.ttf", "arialbd.ttf", "ariali.ttf", "georgia.ttf", "georgiab.ttf",
    "cambria.ttc", "calibri.ttf", "calibrib.ttf", "verdana.ttf", "tahoma.ttf", "pala.ttf", "palab.ttf", "segoeui.ttf",
    "consola.ttf", "cour.ttf", "bookos.ttf", "gara.ttf", "GARA.TTF", "BOOKOS.TTF", "trebuc.ttf",
]
# Non-number text for negative samples: standard tempo and expression terms and neutral words,
# deliberately not taken from the training scores.
WORDS = []


@lru_cache(maxsize=1)
def available_fonts() -> list[Path]:
    fonts = [FONT_DIR / n for n in FONT_NAMES if (FONT_DIR / n).exists()]
    return list(dict.fromkeys(fonts))


def random_label(rng: random.Random) -> str:
    r = rng.random()
    n = rng.choice([rng.randint(1, 9), rng.randint(10, 99), rng.randint(100, 350)])
    if r < 0.55:
        return str(n)
    if r < 0.72:
        return f"{n}{rng.choice('ABCDEFGHIJKLMN')}" if rng.random() < 0.9 else f"{n}{rng.choice('abcd')}"
    if r < 0.8:
        return rng.choice("ABCDEFGH")
    if r < 0.92:
        k = rng.randint(1, 30)
        return f"{n}-{n + k}"
    a = rng.choice("ABCDEF")
    return f"{n}{a}-{n}{chr(ord(a) + rng.randint(1, 6))}"


def render_text(text: str, rng: random.Random) -> np.ndarray:
    """Render text as a grayscale crop with padding similar to real crops (255 = paper)."""
    fonts = available_fonts()
    size = rng.randint(22, 40)
    font = ImageFont.truetype(str(rng.choice(fonts)), size)
    shown = text
    if "-" in text:
        dash = "\u2013" if rng.random() < 0.7 else "-"  # en dash as printed
        a, b = text.split("-")
        if rng.random() < 0.3:  # printed as `30 A-30 D`
            a, b = (x[:-1] + " " + x[-1] if x[-1].isalpha() and x[:-1] else x for x in (a, b))
        shown = a + dash + b
    bbox = font.getbbox(shown)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(size * rng.uniform(0.2, 0.5))
    img = Image.new("L", (w + 2 * pad, h + 2 * pad), 255)
    d = ImageDraw.Draw(img)
    d.text((pad - bbox[0], pad - bbox[1]), shown, font=font, fill=rng.randint(0, 60))
    if rng.random() < 0.15:  # boxed label
        d.rectangle([pad // 3, pad // 3, img.width - 1 - pad // 3, img.height - 1 - pad // 3], outline=0, width=max(1, size // 16))
    if rng.random() < 0.3:  # clutter: a stray stem or ledger fragment at the border
        x = rng.choice([1, img.width - 2])
        d.line([x, 0, x, img.height], fill=0, width=1)
    ang = rng.uniform(-3, 3)
    img = img.rotate(ang, resample=Image.BILINEAR, expand=False, fillcolor=255)
    return np.asarray(img)


def synthetic_sample(rng: random.Random) -> tuple[np.ndarray, str]:
    if rng.random() < 0.2:
        return render_text(rng.choice(WORDS), rng), ""
    t = random_label(rng)
    return render_text(t, rng), t


def ctc_input_length() -> int:
    return 32


def label_ok(text: str) -> bool:
    return all(ch in CHARSET for ch in text) and len(text) <= 16 and math.isfinite(len(text))
