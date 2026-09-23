// Stage timings of the runtime on one PDF page (Node, WASM). Usage: node test/profile.mjs <pdf> [page]

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createCanvas } from "@napi-rs/canvas";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import { renderGray, preparePage, PASS1_DPI } from "../dist/render.js";
import { estimateStaffSpace, estimateSkew } from "../dist/normalize.js";
import { detectSystems } from "../dist/staves.js";
import { makeStrip, rectify } from "../dist/rectify.js";
import { loadModels, boundarySignals } from "../dist/models.js";

// pdf.js decodes JBIG2 and JPEG 2000 images with WebAssembly and needs this folder (forward slashes, trailing slash).
const wasmUrl = path.join(path.dirname(fileURLToPath(import.meta.resolve("pdfjs-dist/package.json"))), "wasm").split(path.sep).join("/") + "/";
const here = path.dirname(fileURLToPath(import.meta.url));
const [pdfPath, pageArg] = process.argv.slice(2);
const p = Number(pageArg ?? 0);
const canvas = (w, h) => createCanvas(w, h);
const doc = await pdfjs.getDocument({ data: new Uint8Array(fs.readFileSync(pdfPath)), verbosity: 0, wasmUrl }).promise;
const models = await loadModels(
  {
    measurenet: new Uint8Array(fs.readFileSync(path.join(here, "..", "models", "measurenet.onnx"))),
    labelocr: new Uint8Array(fs.readFileSync(path.join(here, "..", "models", "labelocr.onnx"))),
  },
  ["wasm"],
);
const t = () => performance.now();
let t0 = t();
const g1 = await renderGray(doc, p, PASS1_DPI / 72, canvas);
console.log(`render pass 1      ${(t() - t0).toFixed(0)} ms (${g1.width}x${g1.height})`);
t0 = t();
const s1 = estimateStaffSpace(g1).s;
console.log(`staff space pass 1 ${(t() - t0).toFixed(0)} ms`);
t0 = t();
const g2 = await renderGray(doc, p, ((PASS1_DPI / 72) * 18) / s1, canvas);
console.log(`render pass 2      ${(t() - t0).toFixed(0)} ms (${g2.width}x${g2.height})`);
t0 = t();
const s2 = estimateStaffSpace(g2).s;
console.log(`staff space pass 2 ${(t() - t0).toFixed(0)} ms`);
t0 = t();
estimateSkew(g2, s2);
console.log(`skew               ${(t() - t0).toFixed(0)} ms`);
t0 = t();
const prep = await preparePage(doc, p, s1, s1, canvas);
console.log(`preparePage total  ${(t() - t0).toFixed(0)} ms`);
t0 = t();
const { systems } = detectSystems(prep.norm8);
console.log(`systems            ${(t() - t0).toFixed(0)} ms (${systems.length})`);
let ts = 0, tr = 0, tm = 0;
for (const sy of systems) {
  let a = t();
  const strip = makeStrip(sy, prep.norm8.width, prep.norm8);
  ts += t() - a;
  a = t();
  const img = rectify(prep.norm8, strip);
  tr += t() - a;
  a = t();
  await boundarySignals(models, img, strip);
  tm += t() - a;
}
console.log(`strips + shear      ${ts.toFixed(0)} ms, rectify ${tr.toFixed(0)} ms, MeasureNet ${tm.toFixed(0)} ms`);
