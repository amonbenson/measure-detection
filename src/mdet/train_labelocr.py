"""Train LabelOCR on bootstrapped number crops plus synthetic crops (guideline section 6.3).

Real and synthetic samples are mixed 1:1. Alphanumeric labels are oversampled 4x.
Usage: uv run python -m mdet.train_labelocr [--epochs 60] [--out runs/labelocr]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import photometric
from .bootstrap import load_numbers
from .data import RUNS, read_split
from .models import LabelOCR, count_parameters
from .ocr import CHARSET, encode, greedy_decode, synthetic_sample
from .text import ocr_input


def load_real(doc_ids: list[str], neg_ratio: float = 0.5, seed: int = 0) -> list[tuple[np.ndarray, str]]:
    """Positives, and negatives subsampled to neg_ratio x the positive count."""
    rng = random.Random(seed)
    pos, neg = [], []
    for d in doc_ids:
        crops, texts, _, _ = load_numbers(d)
        for c, t in zip(crops, texts):
            (pos if t else neg).append((c, t))
    rng.shuffle(neg)
    return pos + neg[: int(neg_ratio * len(pos))]


def jitter(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small scale, shift and padding changes of a crop."""
    h, w = img.shape
    f = rng.uniform(0.85, 1.15)
    img = cv2.resize(img, (max(2, int(w * f)), max(2, int(h * f * rng.uniform(0.9, 1.1)))))
    t, b, l, r = (int(rng.integers(0, max(1, h // 5))) for _ in range(4))
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=255)
    ct, cb, cl, cr = (int(rng.integers(0, max(1, img.shape[k % 2] // 12))) for k in range(4))
    return img[ct : img.shape[0] - cb, cl : img.shape[1] - cr]


class OcrDataset(Dataset):
    def __init__(self, real: list[tuple[np.ndarray, str]], train: bool, synthetic_ratio: float = 1.0, seed: int = 0):
        self.real = real
        self.train = train
        self.seed = seed
        self.epoch = 0
        if train:
            # Oversample alphanumeric and range labels.
            self.index = [i for i, (_, t) in enumerate(real) for _ in range(4 if t and not t.isdigit() else 1)]
            self.n_syn = int(synthetic_ratio * len(self.index))
        else:
            self.index = list(range(len(real)))
            self.n_syn = 0

    def __len__(self) -> int:
        return len(self.index) + self.n_syn

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        if i < len(self.index):
            img, text = self.real[self.index[i]]
        else:
            img, text = synthetic_sample(random.Random(int(rng.integers(1 << 31))))
        if self.train:
            img = photometric(jitter(img, rng), rng)
        x = ocr_input(img)
        return torch.from_numpy(x)[None], text


def collate(items):
    xs = torch.stack([x for x, _ in items])
    texts = [t for _, t in items]
    targets = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
    lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.cat(targets) if any(len(t) for t in targets) else torch.zeros(0, dtype=torch.long)
    return xs, flat, lengths, texts


@torch.no_grad()
def evaluate(model, ds: OcrDataset, device) -> dict:
    model.eval()
    dl = DataLoader(ds, batch_size=512, collate_fn=collate)
    ok = okpos = npos = n = 0
    for xs, _, _, texts in dl:
        lp = model(xs.to(device)).float().cpu().numpy()
        for k, t in enumerate(texts):
            pred = greedy_decode(lp[k])
            ok += pred == t
            n += 1
            if t:
                npos += 1
                okpos += pred == t
    return {"string_accuracy": ok / max(n, 1), "positive_accuracy": okpos / max(npos, 1), "n": n, "n_positive": npos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--train-docs", nargs="*")
    ap.add_argument("--val-docs", nargs="*")
    ap.add_argument("--out", default=str(RUNS / "labelocr"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_docs = args.train_docs or [d for sp in args.train_splits for d in read_split(sp)]
    val_docs = args.val_docs or read_split(args.val_split)
    train = OcrDataset(load_real(train_docs, seed=args.seed), train=True, seed=args.seed)
    val = OcrDataset(load_real(val_docs, neg_ratio=1e9), train=False)
    print(f"train real {len(train.real)} (+{train.n_syn} synthetic per epoch), val {len(val)}", flush=True)
    dl = DataLoader(train, batch_size=args.batch, shuffle=True, collate_fn=collate, num_workers=args.workers, persistent_workers=True, drop_last=True)
    model = LabelOCR(len(CHARSET) + 1).to(device)
    ema = copy.deepcopy(model).eval()
    print(f"parameters {count_parameters(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / (2 * len(dl))) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        train.epoch = ep
        t0, tot = time.time(), 0.0
        for xs, flat, lengths, _ in dl:
            lp = model(xs.to(device))  # (B, T, C)
            t_len = torch.full((lp.shape[0],), lp.shape[1], dtype=torch.long)
            loss = F.ctc_loss(lp.transpose(0, 1).float(), flat, t_len, lengths, blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                d = min(0.999, (1 + sched.last_epoch) / (10 + sched.last_epoch))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(dl), "time": time.time() - t0, "val": evaluate(ema, val, device)}
        if rec["val"]["string_accuracy"] > best:
            best = rec["val"]["string_accuracy"]
            torch.save({"model": ema.state_dict(), "charset": CHARSET, "epoch": ep, "val": rec["val"], "train_docs": train_docs}, out / "best.pt")
        log.append(rec)
        print(json.dumps(rec), flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8", newline="\n")
    print(f"best val string accuracy {best:.4f}")


if __name__ == "__main__":
    main()
