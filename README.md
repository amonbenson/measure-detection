# measure-detection

Finds every measure on printed and scanned sheet-music pages (piano-conductor, piano-vocal and vocal scores) and reads its measure label (`12`, `23A`, `B`). The result is a list of measures with page, quadrilateral, axis-aligned box, label and confidence, in the same JSON schema as the training annotations.

The repository contains:

- the training and evaluation pipeline (Python, `src/mdet/`),
- the browser runtime as an npm package (`packages/measure-detection/`), which reproduces the Python pipeline with ONNX models.

Design, terms and decisions: [`docs/PLAN.md`](docs/PLAN.md). Annotation conventions found in the data: [`docs/CONVENTIONS.md`](docs/CONVENTIONS.md). Measured results: [`docs/RESULTS.md`](docs/RESULTS.md).

## How it works

Most of the work is classical image processing. Two small networks make the decisions that rules cannot make reliably.

| Stage | Method |
|---|---|
| 0 Render | pdf.js, two passes (staff space first, then a resolution with s close to 18 px) |
| 1 Normalize | Staff space s from vertical run lengths; deskew by projection energy; page scaled to s = 16 and s = 8 |
| Systems | Staff lines from slab projections; five equidistant lines form a staff; staves joined by the system's opening line form a system |
| 2 Rectify | Each system becomes a straight strip (staff contours fitted per staff, per-system scan shear corrected), 6 s margins |
| 3 Boundaries | **MeasureNet** (0.27 M parameters): per-column probabilities of start boundary and barline; peak picking |
| Text | Connected components in the strip margins, grouped into candidate strings |
| Labels | **LabelOCR** (0.77 M parameters, convolutional CTC): reads candidates, top-5 hypotheses |
| 4 Assign | Hungarian assignment of readings to measures by offset likelihood; the document's number placement is estimated on the fly |
| 5 Decode | Viterbi over the whole document: consecutive numbering is the prior, readings are evidence; multi-measure rests (`12-15`) are split |

## Using the npm package

```js
import * as pdfjs from "pdfjs-dist";
import { MeasureDetector } from "measure-detection";

const detector = await MeasureDetector.create({
  models: { measurenet: "/models/measurenet.onnx", labelocr: "/models/labelocr.onnx" },
  decoder: await (await fetch("/models/decoder.json")).json(),
});
const pdf = await pdfjs.getDocument({ url }).promise;
const measures = await detector.detect(pdf, { startConvention: "system-edge" });
// [{ layout: { page, x, y, width, height }, value: "12", quad: [[x, y], ...], confidence, ... }]
```

The model files are in `packages/measure-detection/models/` (copy them to your static assets; `manifest.json` records the dataset, split hashes, training commit and metrics of the bundle). Run `detect` in a Web Worker to keep the UI responsive. `startConvention` selects where the first measure of a system begins: at the system's opening line (`system-edge`, default) or after clef, key and time signature (`after-header`). Low `confidence` marks measures worth showing to the user for review.

## Training and evaluation

Requirements: [uv](https://docs.astral.sh/uv/), Node 22, an NVIDIA GPU with CUDA 12.8 drivers.

```
uv sync
npm install
```

Data layout (not in the repository): one folder per document.

```
dataset/documents/<doc_id>/score.pdf       the score
dataset/documents/<doc_id>/measures.json   [{ "beats", "layout": {page, x, y, width, height}, "value" }]
dataset/documents/<doc_id>/source.json     { "show", "number", "title" }
```

Pipeline:

```
uv run python -m mdet.audit                     # data audit and overlays -> runs/audit/
uv run python -m mdet.splits                    # dataset/meta.json; splits are frozen in dataset/splits/
uv run python -m mdet.prepare                   # render and normalize pages
uv run python -m mdet.targets                   # boundary targets
uv run python -m mdet.splits                    # adds the start convention to meta.json
uv run python -m mdet.splits --challenge        # rule-based challenge pages
uv run python -m mdet.bootstrap                 # measure-number crops (round 0)
uv run python -m mdet.train_labelocr            # label reader
uv run python -m mdet.bootstrap --ocr runs/labelocr/best.pt   # self-training round
uv run python -m mdet.train_labelocr            # label reader, round 1
uv run python -m mdet.train_measurenet          # boundary model
uv run python -m mdet.run --detector measurenet --ckpt runs/measurenet/best.pt --labels runs/labelocr/best.pt --split test --name test
uv run python -m mdet.report runs/eval/test     # stratified report, review curve, error log
uv run python -m mdet.export --measurenet runs/measurenet/best.pt --labelocr runs/labelocr/best.pt --out packages/measure-detection/models
uv run python -m mdet.parity                    # golden outputs for the browser parity test
npm test -w measure-detection                   # build, parity test (Node, WASM)
```

Browser demo (WebGPU or WASM, draws the detected measures, reports timings):

```
uv run python packages/measure-detection/demo/serve.py
# open http://localhost:8766/packages/measure-detection/demo/index.html?pdf=/dataset/documents/<doc_id>/score.pdf
```

Box-detector baseline (RF-DETR-S, separate environment in `baselines/rfdetr/`):

```
uv run python baselines/rfdetr/make_dataset.py
cd baselines/rfdetr && uv sync && uv run python train.py --data ../../runs/rfdetr/data --out ../../runs/rfdetr/model
```

Single PDF with the Python pipeline:

```
uv run python -m mdet.predict score.pdf --measurenet runs/measurenet/best.pt --labelocr runs/labelocr/best.pt
```

## Repository layout

| Path | Content |
|---|---|
| `src/mdet/` | Python package (data, classical stages, models, training, evaluation) |
| `tools/render/` | pdf.js renderer for training data |
| `packages/measure-detection/` | npm package (TypeScript runtime and ONNX models) |
| `dataset/splits/`, `dataset/meta.json` | Frozen splits and per-document metadata (the only committed data) |
| `baselines/rfdetr/` | RF-DETR box-detector baseline (own uv environment) |
| `scripts/` | Job queue for the leave-one-show-out folds and the baseline |
| `docs/` | Plan, conventions, audit report, results |
