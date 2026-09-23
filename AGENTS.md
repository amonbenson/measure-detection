# AGENTS.md

Guidance for coding agents working in this repository. Read `docs/PLAN.md` (terms, architecture) and `docs/CONVENTIONS.md` (annotation rules) before changing code.

## Language rules

- One term has one meaning. Use the terms defined in `docs/PLAN.md` section 1 (document, show, page, staff, system, row, measure, boundary, start boundary, barline boundary, virtual boundary, s, label, value, pass, range label). Do not introduce synonyms.
- Plain ASCII in code, comments and docs: no emojis, no em dashes.
- Concise comments that explain why, not what.

## Layout

| Path | Content |
|---|---|
| `src/mdet/` | Python package: data loading, classical stages, models, training, evaluation |
| `tools/render/render.mjs` | pdfjs-dist renderer used for training data (same library as the browser runtime) |
| `packages/measure-detection/` | TypeScript npm package: browser runtime (classical stages ported 1:1, ONNX models) |
| `dataset/` | Data. Only `splits/` and `meta.json` are committed; everything else is gitignored |
| `runs/` | Checkpoints, logs, evaluation outputs (gitignored) |
| `docs/` | Plan, conventions, audit report, results |

## Invariants

- Never commit PDFs, measure JSON, derived images, checkpoints or scripts that download data from the server.
- `dataset/splits/test.txt` is frozen. Add data to train and val only.
- Split by document, never by page.
- Tune thresholds on `val` only. Run `test` rarely and report it as is.
- Classical code first. Add learned components only where a classical rule fails measurably, and record the measurement in `docs/RESULTS.md`.
- The Python pipeline is the reference. Any change to a classical stage in `src/mdet/` must be mirrored in `packages/measure-detection/src/` and pass the parity test (`npm test` in the package).
- Image operations that must match between Python and TypeScript use the explicit implementations (`halve`, `resize_area`, Otsu, fixed-point bilinear warp). Do not replace them with library calls that have different rounding.
- Python `round()` and NumPy round half to even; the TypeScript port uses `roundHalfEven` in the same places.

## Commands

```
uv sync                                        # Python environment (PyTorch CUDA 12.8)
npm install                                    # Node workspace (renderer, package)
uv run python -m mdet.audit                    # data audit -> runs/audit/report.md
uv run python -m mdet.splits                   # meta.json and splits (test split is frozen)
uv run python -m mdet.prepare                  # render and normalize all pages
uv run python -m mdet.targets                  # boundary targets (strips)
uv run python -m mdet.bootstrap [--ocr ckpt]   # measure-number crops
uv run python -m mdet.train_measurenet         # boundary model
uv run python -m mdet.train_labelocr           # label reader
uv run python -m mdet.run --detector measurenet --ckpt runs/measurenet/best.pt --labels runs/labelocr/best.pt --split val
uv run python -m mdet.report runs/eval/<name>  # stratified report, review curve, error log
uv run python -m mdet.export --measurenet ... --labelocr ... --out packages/measure-detection/models
```

## GPU notes (Windows, 12 GB)

- Train one model at a time. Two concurrent trainings exceed GPU memory and Windows silently falls back to system memory (10x to 50x slower).
- Keep input shapes in a small set of buckets (`HEIGHT_BUCKETS`, `TRAIN_WIDTH`) and cap memory with `torch.cuda.set_per_process_memory_fraction`; variable shapes fragment the caching allocator.
- Process pools on Windows need a module file (not stdin) because workers are spawned.
