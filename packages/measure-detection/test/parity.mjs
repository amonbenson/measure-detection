// Parity test (guideline 8.5): the TypeScript pipeline must reproduce the Python reference.
//
// Golden outputs are produced by `uv run python -m mdet.parity` (repository root) and listed in
// $MDET_PARITY_DIR/index.json: [{ "pdf": "...", "golden": "..." }].
// Requirement: same number of measures per page, boundaries within 0.2 s, identical labels.
// Runs in Node with onnxruntime-web (WASM backend) and @napi-rs/canvas.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createCanvas } from "@napi-rs/canvas";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import { MeasureDetector } from "../dist/index.js";

// pdf.js decodes JBIG2 and JPEG 2000 images with WebAssembly and needs this folder (forward slashes, trailing slash).
const wasmUrl = path.join(path.dirname(fileURLToPath(import.meta.resolve("pdfjs-dist/package.json"))), "wasm").split(path.sep).join("/") + "/";
const here = path.dirname(fileURLToPath(import.meta.url));
const models = path.join(here, "..", "models");
const dir = process.env.MDET_PARITY_DIR || path.join(here, "..", "..", "..", "runs", "parity");
const TOL_S = 0.2;

if (!fs.existsSync(path.join(dir, "index.json"))) {
  console.log(`parity: no golden outputs in ${dir}; run \`uv run python -m mdet.parity\` first. Skipped.`);
  process.exit(0);
}
const index = JSON.parse(fs.readFileSync(path.join(dir, "index.json"), "utf8"));
const detector = await MeasureDetector.create({
  models: {
    measurenet: new Uint8Array(fs.readFileSync(path.join(models, "measurenet.onnx"))),
    labelocr: new Uint8Array(fs.readFileSync(path.join(models, "labelocr.onnx"))),
  },
  decoder: JSON.parse(fs.readFileSync(path.join(models, "decoder.json"), "utf8")),
  providers: ["wasm"],
});

let pages = 0, failures = 0, maxErr = 0, measures = 0;
const t0 = Date.now();
for (const item of index) {
  const golden = JSON.parse(fs.readFileSync(item.golden, "utf8"));
  const doc = await pdfjs.getDocument({ data: new Uint8Array(fs.readFileSync(item.pdf)), verbosity: 0, wasmUrl }).promise;
  const got = await detector.detect(doc, { canvas: (w, h) => createCanvas(w, h) });
  const byPage = (list) => {
    const m = new Map();
    for (const e of list) (m.get(e.layout.page) ?? m.set(e.layout.page, []).get(e.layout.page)).push(e);
    return m;
  };
  const g = byPage(golden.measures), t = byPage(got);
  for (const [page, gm] of g) {
    pages++;
    const tm = t.get(page) ?? [];
    const s = golden.pages[page].s;
    const problems = [];
    if (tm.length !== gm.length) problems.push(`measure count ${tm.length} vs ${gm.length}`);
    for (let k = 0; k < Math.min(tm.length, gm.length); k++) {
      for (const [a, b] of [
        [tm[k].quad[0][0], gm[k].quad[0][0]],
        [tm[k].quad[1][0], gm[k].quad[1][0]],
      ]) {
        const e = Math.abs(a - b) / s;
        maxErr = Math.max(maxErr, e);
        if (e > TOL_S) problems.push(`measure ${k} boundary off by ${e.toFixed(3)} s`);
      }
      if (tm[k].value !== gm[k].value) problems.push(`measure ${k} label ${tm[k].value} vs ${gm[k].value}`);
    }
    measures += gm.length;
    if (problems.length) {
      failures++;
      console.log(`FAIL ${path.basename(path.dirname(item.pdf))} page ${page}: ${problems.slice(0, 5).join("; ")}`);
    }
  }
  await doc.destroy?.();
}
const secs = (Date.now() - t0) / 1000;
console.log(`parity: ${pages - failures}/${pages} pages identical within ${TOL_S} s (${measures} measures), max boundary difference ${maxErr.toFixed(4)} s, ${secs.toFixed(1)} s total (${(secs / pages).toFixed(2)} s per page, WASM)`);
process.exit(failures ? 1 : 0);
