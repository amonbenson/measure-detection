"""ONNX export (guideline sections 6.1 and 8.2) and PyTorch/ONNX Runtime parity check.

MeasureNet is exported with dynamic height and width (the strip size varies); LabelOCR with a
fixed 32 x 128 input and dynamic batch.
Usage: uv run python -m mdet.export [--measurenet ckpt] [--labelocr ckpt] [--out dir]
Without checkpoints, untrained stubs are exported (the day-one operator check).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .data import RUNS
from .models import LabelOCR, MeasureNet


def export_model(model: torch.nn.Module, example: torch.Tensor, path: Path, dynamic: dict) -> float:
    model.eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, str(path), input_names=["input"], output_names=["output"], dynamic_axes=dynamic, opset_version=17, dynamo=False)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(example).numpy()
    got = sess.run(None, {"input": example.numpy()})[0]
    return float(np.abs(ref - got).max())


def load_checkpoint(model: torch.nn.Module, ckpt: str | None) -> dict:
    if not ckpt:
        return {}
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    return {k: v for k, v in state.items() if k != "model" and k != "optimizer"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurenet")
    ap.add_argument("--labelocr")
    ap.add_argument("--out", default=str(RUNS / "export"))
    args = ap.parse_args()
    out = Path(args.out)
    torch.manual_seed(0)

    mn = MeasureNet()
    mn_meta = load_checkpoint(mn, args.measurenet)
    x = torch.rand(1, 2, 256, 1088)
    err = export_model(mn, x, out / "measurenet.onnx", {"input": {0: "batch", 2: "height", 3: "width"}, "output": {0: "batch", 2: "columns"}})
    # Check a second shape to confirm dynamic axes.
    sess = ort.InferenceSession(str(out / "measurenet.onnx"), providers=["CPUExecutionProvider"])
    x2 = torch.rand(1, 2, 448, 1280)
    with torch.no_grad():
        err2 = float(np.abs(mn(x2).numpy() - sess.run(None, {"input": x2.numpy()})[0]).max())
    print(f"measurenet.onnx: max abs diff {err:.2e} (256x1088), {err2:.2e} (448x1280)")

    ocr = LabelOCR(n_classes=64)
    ocr_meta = load_checkpoint(ocr, args.labelocr)
    if ocr_meta.get("charset"):
        ocr = LabelOCR(n_classes=len(ocr_meta["charset"]) + 1)
        load_checkpoint(ocr, args.labelocr)
    err = export_model(ocr, torch.rand(4, 1, 32, 128), out / "labelocr.onnx", {"input": {0: "batch"}, "output": {0: "batch"}})
    print(f"labelocr.onnx: max abs diff {err:.2e}")
    manifest = {"measurenet": mn_meta, "labelocr": ocr_meta}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
