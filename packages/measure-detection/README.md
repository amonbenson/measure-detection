# measure-detection

![Detected measures and measure numbers on a piano-vocal score, shown in the curation page](https://raw.githubusercontent.com/amonbenson/measure-detection/main/docs/images/hero.png)

Finds every measure on printed and scanned sheet-music pages (piano-conductor, piano-vocal and vocal scores) in the browser and reads its measure label (`12`, `23A`, `B`). The result is a list of measures with page, quadrilateral, axis-aligned box, label and confidence.

Classical image processing does most of the work; two small ONNX models (MeasureNet for boundaries, LabelOCR for measure numbers, about 4 MB together) make the decisions rules cannot make reliably. Design, training and measured results: [github.com/amonbenson/measure-detection](https://github.com/amonbenson/measure-detection).

## Install

```
npm install measure-detection pdfjs-dist
```

`pdfjs-dist` is a peer dependency: you open the PDF with your own pdf.js and pass the document in.

## Usage

```js
import * as pdfjs from "pdfjs-dist";
import { MeasureDetector } from "measure-detection";

const detector = await MeasureDetector.create({
  models: { measurenet: "/models/measurenet.onnx", labelocr: "/models/labelocr.onnx" },
  decoder: await (await fetch("/models/decoder.json")).json(),
});
// wasmUrl lets pdf.js decode JBIG2 and JPEG 2000 scans; serve node_modules/pdfjs-dist/wasm/ there.
const pdf = await pdfjs.getDocument({ url, wasmUrl: "/pdfjs-wasm/" }).promise;
const measures = await detector.detect(pdf);
// [{ layout: { page, x, y, width, height }, value: "12", quad: [[x, y], ...], confidence, ... }]
```

The model files ship in `node_modules/measure-detection/models/` (also importable as `measure-detection/models/*`). Copy them to your static assets or let your bundler emit them as URLs. `models` also accepts `ArrayBuffer` or `Uint8Array`. `manifest.json` records the dataset, training commit and metrics of the bundle.

The first measure of a system starts after clef, key and time signature, in the space before the first note or rest. Low `confidence` marks measures worth showing to the user for review.

## Performance and resources

- The image processing runs in Web Workers (`measure-detection/worker.js`, started with `new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`, which Vite and webpack bundle automatically; pass `createWorker` for other setups). Pool size: CPU cores minus one, at most 4, and 1 or 2 on devices that report 2 or 4 GB of memory. `workers: 0` or an environment without workers (Node, strict CSP) processes pages on the calling thread with identical results. `detector.dispose()` stops the workers.
- Pages are streamed: at most (workers + 1) pages are in memory at once, so memory does not grow with the document length.
- The models run on WebGPU when the browser has an adapter, otherwise on WASM in onnxruntime's own worker (`wasmProxy`, default on). WASM uses several threads only if the page is cross-origin isolated (`Cross-Origin-Opener-Policy: same-origin`, `Cross-Origin-Embedder-Policy: require-corp`).
- Browsers throttle or pause pdf.js rendering in background tabs, so detection continues when the tab is visible again.

## License

MIT
