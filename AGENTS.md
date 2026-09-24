# AGENTS.md

Guidance for coding agents working in this repository. Read `docs/PLAN.md` (terms, architecture), `docs/CONVENTIONS.md` (annotation rules) and `docs/RESULTS.md` (section 0 is the current model) before changing code.

## Language rules

- One term has one meaning. Use the terms defined in `docs/PLAN.md` section 1 (document, show, page, staff, system, row, measure, boundary, start boundary, barline boundary, virtual boundary, s, label, value, pass, range label). Do not introduce synonyms.
- Plain ASCII in code, comments and docs: no emojis, no em dashes.
- Concise comments that explain why, not what.

## Layout

| Path | Content |
|---|---|
| `src/mdet/` | Python package: data loading, classical stages, models, training, evaluation |
| `tools/render/render.mjs` | pdfjs-dist renderer used for training data (same library as the browser runtime) |
| `packages/measure-detection/src/` | TypeScript npm package: browser runtime (classical stages ported 1:1, ONNX models) |
| `packages/measure-detection/models/` | Exported model bundle (ONNX, `decoder.json`, `manifest.json`) |
| `packages/measure-detection/example/` | Curation page and its server (`npm run example`): split books into documents, detect, correct, export training data |
| `dataset/` | Data. Only `splits/` and `meta.json` are committed; everything else is gitignored |
| `runs/` | Checkpoints, logs, evaluation outputs (gitignored). `runs/v1/` holds the first training run |
| `docs/` | Plan, conventions, results |

## Invariants

- Never commit PDFs, measure JSON, derived images, checkpoints or scripts that download data from the server.
- `source.json` holds only `show_id` and the song `number` (plus `curated` and `modified` for documents written by the curation page). Never store titles, show names, file names or other string labels in training data or curation sessions. `meta.json` uses the show id as `show`.
- One start convention for every document: the first measure of a system starts after clef, key and time signature, in the space before the first note or rest. `mdet.splits` flags rows that start at the system edge (`edge_starts` in `meta.json`); fix them before training.
- `dataset/splits/test.txt` is frozen. Add data to train and val only. The curation server enforces this.
- Split by document, never by page.
- Tune thresholds on `val` only. Run `test` rarely and report it as is.
- Classical code first. Add learned components only where a classical rule fails measurably, and record the measurement in `docs/RESULTS.md`.
- The Python pipeline is the reference. Any change to a classical stage in `src/mdet/` must be mirrored in `packages/measure-detection/src/` and pass the parity test (`npm test` in the package). A change that only moves work (workers, ordering) needs a parity run on one document plus a comparison of worker and in-thread output, not the full run.
- Image operations that must match between Python and TypeScript use the explicit implementations (`halve`, `resize_area`, Otsu, fixed-point bilinear warp). Do not replace them with library calls that have different rounding.
- Python `round()` and NumPy round half to even; the TypeScript port uses `roundHalfEven` in the same places.
- The runtime's worker code (`worker.ts`, `ops.ts`, `stages.ts` and what they import) must not import onnxruntime-web or pdf.js: workers load without a bundler's import resolution. Models and rendering stay on the main thread.
- Original documents edited in the curation page keep their first measure list as `measures.orig.json`.

## Commands

```
uv sync                                        # Python environment (PyTorch CUDA 12.8)
npm install                                    # Node workspace (renderer, package)
uv run python -m mdet.audit                    # data audit -> runs/audit/report.md
uv run python -m mdet.prepare [--docs ...]     # render and normalize pages (only new documents need it)
uv run python -m mdet.targets                  # boundary targets (strips); after any measures.json change
uv run python -m mdet.splits                   # meta.json and start check (needs strips); splits are frozen
uv run python -m mdet.splits --challenge       # challenge pages of test
uv run python -m mdet.bootstrap                # measure-number crops, round 0
uv run python -m mdet.train_labelocr --out runs/labelocr_round0
uv run python -m mdet.bootstrap --ocr runs/labelocr_round0/best.pt   # self-training round
uv run python -m mdet.train_labelocr           # label reader, round 1
uv run python -m mdet.train_measurenet         # boundary model
uv run python -m mdet.run --detector measurenet --ckpt runs/measurenet/best.pt --labels runs/labelocr/best.pt --split val --tau 0.4 --name val
uv run python -m mdet.report runs/eval/<name>  # stratified report, review curve, error log
uv run python -m mdet.export --measurenet runs/measurenet/best.pt --labelocr runs/labelocr/best.pt --out packages/measure-detection/models
uv run python -m mdet.testvectors              # after a decoder refit: test vectors for the unit test
uv run python -m mdet.parity                   # golden outputs for the browser parity test
npm test -w measure-detection                  # build, unit and parity tests (Node, WASM)
npm run example -w measure-detection           # curation page on http://localhost:8080
```

The decoder parameters (`runs/decoder/params.json`) are fitted on the train split when missing; delete the file after the train split or the bootstrap records change, and fit it once before evaluating with parallel workers.

## GPU notes (Windows, 12 GB)

- Train one model at a time. Two concurrent trainings exceed GPU memory and Windows silently falls back to system memory (10x to 50x slower).
- Keep input shapes in a small set of buckets (`HEIGHT_BUCKETS`, `TRAIN_WIDTH`) and cap memory with `torch.cuda.set_per_process_memory_fraction`; variable shapes fragment the caching allocator.
- Process pools on Windows need a module file (not stdin) because workers are spawned.

## Browser testing

- pdf.js does not render in hidden pages, and WebGPU stalls there. When the browser used for testing is hidden, test workers and models without pdf.js (for example with page images decoded by `createImageBitmap`), and run the pipeline in Node.
