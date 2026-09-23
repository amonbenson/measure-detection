// Shear scores of the first system of a page, for debugging parity. Usage: node test/dump_shear.mjs <pdf> <page>

import fs from "node:fs";
import { createCanvas } from "@napi-rs/canvas";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import { pass1, preparePage } from "../dist/render.js";
import { detectSystems, S } from "../dist/staves.js";
import { makeStrip, rectify, vCentre } from "../dist/rectify.js";
import { binarize, open, warpAffineF } from "../dist/image.js";

const [pdfPath, pageArg] = process.argv.slice(2);
const canvas = (w, h) => createCanvas(w, h);
const doc = await pdfjs.getDocument({ data: new Uint8Array(fs.readFileSync(pdfPath)), verbosity: 0 }).promise;
const { s1, sDoc } = await pass1(doc, canvas);
const p = Number(pageArg);
const prep = await preparePage(doc, p, s1[p], sDoc, canvas);
const { systems } = detectSystems(prep.norm8);
const s = makeStrip(systems[0], prep.norm8.width);
const img = rectify(prep.norm8, s);
const a = Math.trunc(s.margin), b = Math.trunc(s.margin + s.heightNom);
const band = { width: img.width, height: b - a, data: img.data.slice(a * img.width, b * img.width) };
const vert = open(binarize(band), 1, Math.trunc(2 * S));
const f = { width: vert.width, height: vert.height, data: Float32Array.from(vert.data) };
const vc = vCentre(s) - a;
const out = [];
for (let i = 0; i < 29; i++) {
  const k = -0.035 + i * 0.0025;
  const sh = warpAffineF(f, [1, k, -k * vc, 0, 1, 0], f.width, f.height, true);
  let score = 0;
  for (let x = 0; x < sh.width; x++) {
    let q = 0;
    for (let y = 0; y < sh.height; y++) q += sh.data[y * sh.width + x];
    score += q * q;
  }
  out.push([+k.toFixed(4), score]);
}
console.log(JSON.stringify({ imgSum: img.data.reduce((x, y) => x + y, 0), vertSum: vert.data.reduce((x, y) => x + y, 0), scores: out.slice(6, 14) }));
