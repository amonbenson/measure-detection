"""Fine-tune RF-DETR-S (Apache-2.0) on system and measure boxes (guideline section 6.4).

Run inside this folder's environment:  uv run python train.py --data ../../runs/rfdetr/data --out ../../runs/rfdetr/model
"""

from __future__ import annotations

import argparse

from rfdetr import RFDETRSmall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resolution", type=int, default=1024)
    args = ap.parse_args()
    model = RFDETRSmall(resolution=args.resolution)
    model.train(
        dataset_dir=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=2,
        grad_accum_steps=8,
        lr=1e-4,
        num_workers=2,
        early_stopping=False,
        progress_bar=False,
    )


if __name__ == "__main__":
    main()
