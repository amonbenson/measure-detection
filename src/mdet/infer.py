"""Inference wrappers for the trained networks (PyTorch; the browser runtime uses the ONNX exports)."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

from .boundaries import decode
from .models import LabelOCR, MeasureNet
from .ocr import CHARSET, beam_search
from .rectify import Strip

DEFAULT_TAU_B = 0.4


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def _load_measurenet(ckpt: str) -> MeasureNet:
    m = MeasureNet()
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"])
    return m.eval().to(_device())


@lru_cache(maxsize=4)
def _load_labelocr(ckpt: str) -> LabelOCR:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = LabelOCR(len(state.get("charset", CHARSET)) + 1)
    m.load_state_dict(state["model"])
    return m.eval().to(_device())


def padded_shape(h: int, w: int) -> tuple[int, int]:
    """Input shape bucket: height from the training buckets (else a multiple of 128), width a multiple of 128 >= 1216."""
    from .train_measurenet import HEIGHT_BUCKETS, TRAIN_WIDTH

    hb = next((b for b in HEIGHT_BUCKETS if b >= h), (h + 127) // 128 * 128)
    return hb, max(TRAIN_WIDTH, (w + 127) // 128 * 128)


def strip_input(img: np.ndarray, strip: Strip) -> np.ndarray:
    from .train_measurenet import make_input

    x = make_input(img, np.asarray(strip.staff_bands, np.float32).reshape(-1, 2))
    h, w = x.shape[1:]
    xp = np.zeros((2, *padded_shape(h, w)), np.float32)
    xp[:, :h, :w] = x
    return xp


class MeasureNetDetector:
    """Boundary detector: (strip image, strip) -> [(u, confidence)] with the start boundary first."""

    def __init__(self, ckpt: str, tau_b: float | None = None, convention: str = "system-edge"):
        from .train_measurenet import CONVENTIONS

        self.ckpt = ckpt
        self.tau_b = DEFAULT_TAU_B if tau_b is None else tau_b
        self.start_channel = CONVENTIONS.index(convention)

    def signals(self, img: np.ndarray, strip: Strip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = _load_measurenet(self.ckpt)
        with torch.no_grad():
            y = model(torch.from_numpy(strip_input(img, strip))[None].to(_device()))[0].float().cpu().numpy()
        c = math.ceil(img.shape[1] / 2)
        sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
        return sig(y[self.start_channel, :c]), sig(y[2, :c]), y[3, :c]

    def __call__(self, img: np.ndarray, strip: Strip) -> list[tuple[float, float]]:
        start, bar, off = self.signals(img, strip)
        return decode(start, bar, off, 2, self.tau_b)


class LabelReader:
    """LabelOCR with CTC beam search: crops -> top-5 [(string, probability)] per crop."""

    def __init__(self, ckpt: str):
        self.ckpt = ckpt

    def __call__(self, inputs: list[np.ndarray]) -> list[list[tuple[str, float]]]:
        if not inputs:
            return []
        model = _load_labelocr(self.ckpt)
        with torch.no_grad():
            lp = model(torch.from_numpy(np.stack(inputs))[:, None].to(_device())).float().cpu().numpy()
        return [beam_search(lp[k]) for k in range(len(inputs))]
