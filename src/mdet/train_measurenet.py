"""Train MeasureNet (boundary model) on strips from dataset/derived/strips (guideline section 6.3).

Usage: uv run python -m mdet.train_measurenet [--epochs 150] [--train-splits train] [--out runs/measurenet]
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from .augment import crop_strip, geometric_strip, photometric
from .boundaries import decode
from .data import DERIVED, RUNS, read_split
from .evaluate import match_boundaries
from .models import MeasureNet, count_parameters
from .staves import S
from .targets import load_samples

STRIDE = 2
SIGMA = 0.5 * S / STRIDE  # Gaussian sigma in output columns
# Fixed training shapes (guideline 8.2 shape buckets): few distinct shapes keep GPU memory stable.
HEIGHT_BUCKETS = (192, 320, 448, 576, 704, 832)
TRAIN_WIDTH = 1216
START, BARLINE, OFFSET = 0, 1, 2  # output channels


def load_strips(doc_ids: list[str]) -> list[dict]:
    out = []
    for d in doc_ids:
        for f in sorted((DERIVED / "strips" / d).glob("*.npz")):
            for s in load_samples(f):
                s["doc_id"] = d
                s["page"] = int(f.stem)
                out.append(s)
    return out


def make_input(img: np.ndarray, bands: np.ndarray) -> np.ndarray:
    """(2, H, W) float32: ink (1 = black) and staff-band mask."""
    h, w = img.shape
    x = np.zeros((2, h, w), np.float32)
    x[0] = 1.0 - img.astype(np.float32) / 255.0
    for vt, vb in bands:
        a, b = max(0, int(round(vt))), min(h, int(round(vb)) + 1)
        x[1, a:b] = 1.0
    return x


def make_targets(width: int, bounds: np.ndarray, virtual: np.ndarray, keep: np.ndarray, has_start: bool):
    """Heatmaps (2, C) for start and barline, validity mask (2, C), positive indices and offsets, C = ceil(width / STRIDE)."""
    c = math.ceil(width / STRIDE)
    heat = np.zeros((2, c), np.float32)
    valid = np.ones((2, c), bool)
    cols = np.arange(c, dtype=np.float32)
    pos_idx, pos_off = [], []
    for k, u in enumerate(bounds):
        if not keep[k]:
            continue
        cu = u / STRIDE
        ch = START if (k == 0 and has_start) else BARLINE
        if virtual[k]:
            valid[BARLINE] &= ~(np.abs(cols + 0.5 - cu) <= S / STRIDE)
            continue
        g = np.exp(-((cols + 0.5 - cu) ** 2) / (2 * SIGMA**2))
        heat[ch] = np.maximum(heat[ch], g)
        i = int(np.clip(np.floor(cu), 0, c - 1))
        heat[ch, i] = 1.0
        pos_idx.append(i)
        pos_off.append(cu - i)
    return heat, valid, np.asarray(pos_idx, np.int64), np.asarray(pos_off, np.float32)


class StripDataset(Dataset):
    def __init__(self, strips: list[dict], train: bool, seed: int = 0):
        self.strips = strips
        self.train = train
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.strips)

    def __getitem__(self, i: int):
        s = self.strips[i]
        img = s["image"]
        bands = np.asarray(s["staff_bands"], np.float32).reshape(-1, 2)
        bounds = np.asarray(s["boundaries"], np.float32)
        virtual = np.asarray(s["virtual"], bool)
        keep = np.ones(len(bounds), bool)
        has_start = True
        if self.train:
            rng = np.random.default_rng((self.seed, self.epoch, i))
            if rng.random() < 0.8:
                img, bands, bounds = geometric_strip(img, bands, bounds, rng)
            if rng.random() < 0.2:
                img, bounds, keep = crop_strip(img, bounds, keep, rng)
                has_start = bool(keep[0])
            if img.shape[1] > TRAIN_WIDTH:
                img, bounds, keep = crop_strip(img, bounds, keep, rng, min_frac=TRAIN_WIDTH / img.shape[1])
                img = img[:, :TRAIN_WIDTH]
                keep &= bounds < TRAIN_WIDTH
                has_start = bool(keep[0])
            if img.shape[0] > HEIGHT_BUCKETS[-1]:
                # Very tall strips: keep the centre rows (rare; a system taller than 104 s).
                cut = (img.shape[0] - HEIGHT_BUCKETS[-1]) // 2
                img = img[cut : cut + HEIGHT_BUCKETS[-1]]
                bands = np.clip(bands - cut, 0, HEIGHT_BUCKETS[-1] - 1)
            other = self.strips[int(rng.integers(len(self.strips)))]["image"]
            img = photometric(img, rng, other)
        x = make_input(img, bands)
        heat, valid, pidx, poff = make_targets(img.shape[1], bounds, virtual, keep, has_start)
        return x, heat, valid, pidx, poff


class HeightBucketSampler(Sampler):
    """Batches of strips with similar height, shuffled each epoch."""

    def __init__(self, heights: list[int], batch: int, shuffle: bool, seed: int = 0):
        self.heights = heights
        self.batch = batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        idx = list(range(len(self.heights)))
        if self.shuffle:
            rng.shuffle(idx)
            # Sort within chunks of 16 batches by height, then shuffle the batch order.
            chunk = self.batch * 16
            idx = [j for c in range(0, len(idx), chunk) for j in sorted(idx[c : c + chunk], key=lambda k: self.heights[k])]
        else:
            idx.sort(key=lambda k: self.heights[k])
        batches = [idx[b : b + self.batch] for b in range(0, len(idx), self.batch)]
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self) -> int:
        return math.ceil(len(self.heights) / self.batch)


def collate(items):
    hmax = max(x.shape[1] for x, *_ in items)
    hmax = next((b for b in HEIGHT_BUCKETS if b >= hmax), (hmax + 127) // 128 * 128)
    wmax = TRAIN_WIDTH
    c = wmax // STRIDE
    b = len(items)
    xs = torch.zeros(b, 2, hmax, wmax)
    heat = torch.zeros(b, 2, c)
    valid = torch.zeros(b, 2, c, dtype=torch.bool)
    pos_b, pos_i, pos_o = [], [], []
    for k, (x, h, vd, pidx, poff) in enumerate(items):
        xs[k, :, : x.shape[1], : x.shape[2]] = torch.from_numpy(x)
        n = h.shape[1]
        heat[k, :, :n] = torch.from_numpy(h)
        valid[k, :, :n] = torch.from_numpy(vd)
        pos_b += [k] * len(pidx)
        pos_i += pidx.tolist()
        pos_o += poff.tolist()
    return xs, heat, valid, torch.tensor(pos_b), torch.tensor(pos_i), torch.tensor(pos_o)


def focal_loss(logits: torch.Tensor, heat: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Penalty-reduced focal loss (CenterNet) over valid columns, normalized by the positive count."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float()
    neg_w = (1 - heat) ** 4
    lpos = -torch.log(p) * (1 - p) ** 2 * pos
    lneg = -torch.log(1 - p) * p**2 * neg_w * (1 - pos)
    v = valid.float()
    return ((lpos + lneg) * v).sum() / (pos * v).sum().clamp(min=1)


@torch.no_grad()
def evaluate_strips(model, strips: list[dict], device, tau: float = 0.5) -> dict:
    model.eval()
    tp = npred = ngt = 0
    for s in strips:
        x = make_input(s["image"], np.asarray(s["staff_bands"]).reshape(-1, 2))
        h, w = x.shape[1:]
        hb = next((b for b in HEIGHT_BUCKETS if b >= h), (h + 127) // 128 * 128)
        xp = np.zeros((2, hb, max(TRAIN_WIDTH, (w + 127) // 128 * 128)), np.float32)
        xp[:, :h, :w] = x
        y = model(torch.from_numpy(xp)[None].to(device))[0].float().cpu().numpy()
        c = math.ceil(w / STRIDE)
        pr = decode(1 / (1 + np.exp(-y[START, :c])), 1 / (1 + np.exp(-y[BARLINE, :c])), y[OFFSET, :c], STRIDE, tau)
        pu = [u for u, _ in pr]
        gb = np.asarray(s["boundaries"])[~np.asarray(s["virtual"], bool)]
        m = match_boundaries(pu, gb.tolist(), 1.0 * S)
        tp += len(m)
        npred += len(pu)
        ngt += len(gb)
    p, r = tp / max(npred, 1), tp / max(ngt, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--train-docs", nargs="*", help="explicit training documents (overrides --train-splits)")
    ap.add_argument("--val-docs", nargs="*")
    ap.add_argument("--out", default=str(RUNS / "measurenet"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # Without a cap, the caching allocator can grow past the card and Windows falls back to slow system memory.
        torch.cuda.set_per_process_memory_fraction(0.8)

    train_docs = args.train_docs or [d for sp in args.train_splits for d in read_split(sp)]
    val_docs = args.val_docs or read_split(args.val_split)
    train = load_strips(train_docs)
    val = load_strips(val_docs)
    print(f"train strips {len(train)}, val strips {len(val)}", flush=True)

    ds = StripDataset(train, train=True, seed=args.seed)
    sampler = HeightBucketSampler([s["image"].shape[0] for s in train], args.batch, shuffle=True, seed=args.seed)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=args.workers, persistent_workers=args.workers > 0)

    model = MeasureNet().to(device)
    ema = copy.deepcopy(model).eval()
    print(f"parameters {count_parameters(model)}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(sampler)
    warm = 3 * len(sampler)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    scaler = torch.amp.GradScaler()
    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        ds.epoch = sampler.epoch = ep
        t0, tot = time.time(), 0.0
        for xs, heat, valid, pb, pi, po in dl:
            xs, heat, valid = xs.to(device, non_blocking=True), heat.to(device), valid.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                y = model(xs)
            loss = focal_loss(y[:, :2], heat, valid)
            if len(pb):
                off = y[pb.to(device), OFFSET, pi.to(device)].float()
                loss = loss + F.l1_loss(off, po.to(device))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            with torch.no_grad():
                d = min(0.999, (1 + sched.last_epoch) / (10 + sched.last_epoch))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item()
        rec = {"epoch": ep, "loss": tot / len(sampler), "time": time.time() - t0}
        if ep % 5 == 4 or ep == args.epochs - 1:
            rec["val"] = evaluate_strips(ema, val, device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if rec["val"]["f1"] > best:
                best = rec["val"]["f1"]
                torch.save({"model": ema.state_dict(), "epoch": ep, "val": rec["val"], "train_docs": train_docs, "args": vars(args)}, out / "best.pt")
        log.append(rec)
        print(json.dumps(rec), flush=True)
    torch.save({"model": ema.state_dict(), "epoch": args.epochs - 1, "train_docs": train_docs, "args": vars(args)}, out / "last.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1), encoding="utf-8", newline="\n")
    print(f"best val f1 {best:.4f}")


if __name__ == "__main__":
    main()
