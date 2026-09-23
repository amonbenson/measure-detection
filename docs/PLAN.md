# Implementation plan

This plan implements `measure-detection-guideline.md` (v1.0) on the data available on the Pocketbase server. It records the audit findings first, because they change several parts of the guideline.

## 1. Terms

Each term has exactly one meaning in this repository.

| Term | Meaning |
|---|---|
| document | One song: one PDF plus its measure list. Identified by the Pocketbase `songs` record id. |
| show | The musical a document belongs to. All documents of a show share one engraving source. |
| page | One displayed PDF page (after `/Rotate` and CropBox). |
| staff | Five horizontal staff lines. |
| system | A group of staves that is read simultaneously, connected by an opening line on the left. |
| row | The ground-truth counterpart of a system: all annotated measures on one page with the same `y`. |
| measure | One annotated box, or one predicted interval between two boundaries inside a system. |
| boundary | The x position that separates two measures, or the left or right end of a system's measure sequence. |
| start boundary | The left boundary of the first measure of a system. |
| barline boundary | Any other boundary. |
| virtual boundary | A ground-truth boundary with no visible barline (inside a multi-measure rest). |
| s | Staff space in pixels of the image being processed. |
| label | The printed measure name, for example `23`, `23A`, `A`. |
| value | The raw string in the JSON, for example `23A-2`. |
| pass | The `-k` suffix of a value. It counts playback repetitions and is never printed. |
| range label | A printed label for a multi-measure rest, for example `12-15` or `30A-30D`. |

## 2. Audit findings

Source: 3 shows, 55 documents, 4,969 JSON entries, about 470 pages. "third show" has no documents.

| Finding | Evidence | Consequence |
|---|---|---|
| Coordinates are fractions of the displayed page, origin top-left, y down | Overlays on sample pages of all sources | Loader applies `/Rotate` (show A pages are stored landscape with `/Rotate 270`) |
| Two engraving sources, three file sources | show A: 600 dpi grayscale scans, A4. show A song 15: A5, built from image strips. show B: born-digital vector, A5, no text layer | No third publisher exists, so `test_ood` is replaced by a leave-one-show-out experiment |
| Score types | show A: piano-conductor with 1-3 extra staves. show B: mostly one vocal staff, some two-staff vocal systems | PC and V are covered. PV is not present |
| Numbering style | show A: every measure, below the system. show B: every measure, above the staff, plus boxed rehearsal numbers | Both margins are searched |
| The JSON is a playback sequence, not reading order | Jumps back to earlier pages; inserted measures listed out of order | Reading order is derived from geometry (page, row y, x), never from array order |
| Pass suffix `-k` | 172 values. Pass 1 has a box. Passes 2+ have no box | Label = value without `-k`. Entries without box are dropped |
| Entries without box | 143, including one trailing "end" entry per document | Dropped |
| Duplicate boxes | One document (show B 4) labels the same 4 measures as `9A..9D` and `10..13` | Keep one box. Label is the later entry (matches the print); recorded in `docs/CONVENTIONS.md` |
| Label grammar | `N`, `NL` (`23A`, `8a`), `L` (`A`..`E`), each with optional pass | Grammar: optional integer + optional letter suffix. Charset: digits, letters, `-` for range labels |
| Multi-measure rests | show B: one printed rest with range label `12-15` is annotated as 22 equal-width boxes | Detected interval is split into N equal measures after labelling. Virtual boundaries are ignored in barline training |
| Vertical box extent is inconsistent | show A: top to bottom staff line. show B: sometimes includes lyrics, sometimes the staff only | Predictions span top to bottom staff line. System matching uses intersection over the smaller height |
| Start boundary convention differs by show | show A: system opening line, or the start-repeat barline. show B: after clef and key signature | Learned by the boundary model; not hard-coded |
| Tiling | 2 gaps above 1e-4 in 1,315 rows | Rows tile; gaps are logged |
| Unlabeled pages | 7 pages, all "turn page" arrow pages | True negatives |
| Unlabeled documents | show B 11 and 14 have zero entries | Excluded from all splits |

## 3. Architecture

The guideline proposes three networks (SystemNet, MeasureNet, LabelOCR). Following the rule "classical algorithms first, machine learning only where needed", this plan keeps two networks and replaces the rest with classical code. Each replacement is kept only if its validation metric meets the threshold named below; otherwise the guideline's learned component is built.

| Stage | Guideline | This plan | Kind |
|---|---|---|---|
| 0 Render | pdfjs-dist | pdfjs-dist in Node, same library as the browser | classical |
| 1 Normalize | run-length s estimate, Hough deskew | same | classical |
| A Systems | SystemNet (U-Net) | Staff detection by horizontal projection of long horizontal runs; staves grouped into systems by the vertical opening line connecting them at the left end. Fallback to SystemNet if val system recall < 99% | classical |
| 2 Rectify | polynomial contour fit, warp | Staff line tracking per staff, degree 2 fit, warp to straight strip with 6 s margins | classical |
| B Boundaries | MeasureNet head 1 (column head) | Same: small CNN on the strip, per-column logits for `start` and `barline` plus sub-column offset | learned (model 1) |
| B Text mask | MeasureNet head 2 | Classical text candidates: connected components of text size in the margin bands, merged when closer than 1 s | classical |
| 3 Measures | peak NMS, intervals | same | classical |
| C Read labels | LabelOCR (conv CTC) | same | learned (model 2) |
| 4 Assign | Hungarian on offset likelihood | same | classical |
| 5 Decode | Viterbi over label grammar | same, plus range labels that split a measure into N equal measures | classical |

Bootstrapping number locations (guideline 5.5) uses no external OCR engine. A candidate is matched to a measure when (a) it lies in the expected position relative to the measure's left boundary and (b) its component count is plausible for the label length. After the first training round of LabelOCR, matches are re-verified by LabelOCR reading (self-training). A random sample of 200 accepted crops is inspected visually.

## 4. Data splits

Split by document, stratified by show. Proportions 70/15/15 of the 53 labeled documents.

| Split | Content |
|---|---|
| train | about 37 documents |
| val | about 8 documents |
| test | about 8 documents, frozen after creation |
| challenge | Pages of `test` documents selected by rule: alphanumeric labels, range labels, or skew above 1 degree |
| loso | Leave-one-show-out experiment: models trained on show A train+val only, evaluated on all show B documents, and the reverse. Replaces `test_ood` |

Split files are committed (`dataset/splits/*.txt`). PDFs and JSON are never committed.

## 5. Repository layout

```
measure-detection/
  README.md  AGENTS.md  pyproject.toml  uv.lock  package.json
  docs/        PLAN.md  CONVENTIONS.md  RESULTS.md
  dataset/     (gitignored except splits/ and meta.json)
    documents/<doc_id>/score.pdf  measures.json  source.json
    splits/    train.txt  val.txt  test.txt  challenge.txt
    meta.json  per-document metadata (score type, source, numbering style)
    derived/   pages, targets, number crops (regenerated by scripts)
  tools/render/  Node renderer (pdfjs-dist) used for training data
  src/mdet/      Python package: io, audit, normalize, staves, systems, rectify,
                 targets, bootstrap, models, train, pipeline, decoder, evaluate
  packages/measure-detection/  TypeScript npm package (browser runtime)
  runs/        (gitignored) checkpoints, logs, reports, overlays
```

The script that downloads data from Pocketbase is not part of the repository. The README documents the expected `dataset/documents` format instead.

## 6. Phases

Each phase ends with a commit.

| Phase | Work | Exit criterion |
|---|---|---|
| 1 Setup | git repo, uv project (Python 3.12, PyTorch CUDA), Node workspace, `.gitignore`, README, AGENTS | `uv run python -c "import torch; torch.cuda.is_available()"` is true |
| 2 Data foundation | loader (value parsing, dedup, reading order), audit script and report, overlays, `CONVENTIONS.md`, `meta.json`, splits | Audit report committed; every flag explained |
| 3 Rendering and normalization | pdfjs-dist renderer, s estimation, deskew, normalized pages at s = 8 and s = 16 | s estimated on all pages; outliers inspected |
| 4 Classical geometry | staff detection, system grouping, rectification, boundary snapping (virtual boundary detection) | Val system recall and precision >= 99% |
| 5 Evaluation harness | all metrics of guideline 7.2 except runtime, error taxonomy log, stratified reports | Runs end-to-end on ground truth (sanity: perfect scores) |
| 6 Classical baseline | morphological barline detector with height rules | Metrics reported on val and test |
| 7 Boundary model | targets, augmentation, training on GPU, threshold tuning on val | Val barline F1 at 1 s reported and above classical baseline |
| 8 Labels | text candidates, bootstrapping, LabelOCR with synthetic font crops, assignment, decoder | End-to-end label accuracy reported |
| 9 Full evaluation | test, challenge, loso; review curve; error log; `RESULTS.md` | All metrics in `RESULTS.md` |
| 10 Browser package | ONNX export, TypeScript port of classical stages, npm package, parity test in Node (onnxruntime-web WASM), download size | Parity within 0.2 s on 50 reference pages, identical labels |
| 11 Box detector baseline | RF-DETR-S fine-tuned on the same splits | Metrics reported next to the main pipeline |

## 7. Tools

Python 3.12 via uv. PyTorch with CUDA on the local RTX 4070. NumPy, OpenCV, SciPy, Pillow, PyMuPDF (audit only). Node 22 with `pdfjs-dist` and `@napi-rs/canvas` for rendering. `onnx` and `onnxruntime` for export checks.

## 8. Changes during implementation

Recorded as they happened; results in `docs/RESULTS.md`.

| Change | Reason |
|---|---|
| Near-duplicate boxes (IoU >= 0.8) are duplicates, not only identical boxes | Repeated passages annotated twice with slightly different coordinates |
| Per-system scan shear estimated and removed in rectification | 13 % of show A barlines were tilted relative to the staff lines |
| MeasureNet has two start channels, one per start convention; the convention is a runtime setting | The start convention varies per document and is not visible in the image |
| Candidate text: components are rejected only if they intersect the staff region | Range labels printed close to the staff were lost |
| A number followed by a single letter printed with a space (`30 E`) is merged into one reading | Candidate grouping split them |
| Each measure of a split multi-measure rest gets its own label from the range as observation | The decoder otherwise left the suffix sequence after one step |
| Free-jump transition probability is divided by 100 plausible targets | A free jump was as cheap as a suffix step |
| Boundary matching is order preserving; a rest-tolerant metric variant is reported beside the strict one | Hungarian matching crossed pairs inside narrow sub-measures; annotated rest splits are arbitrary |
| OpenCV `INTER_AREA` replaced by explicit area and 2x2 resizes; TypeScript reproduces the OpenCV 5 float bilinear warp | Bit-exact parity between Python and the browser |
| Training shapes in fixed buckets and a GPU memory cap | Variable shapes made Windows fall back to system memory (20x slower) |
| FP16 export not done | FP32 bundle is 4.2 MB, below the 10 MB target |
