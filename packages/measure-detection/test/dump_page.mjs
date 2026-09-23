// Intermediate values of the runtime for one page, for debugging parity. Usage: node test/dump_page.mjs <pdf> <page>

import fs from "node:fs";
import { createCanvas } from "@napi-rs/canvas";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";
import { pass1, preparePage } from "../dist/render.js";
import { detectSystems } from "../dist/staves.js";
import { makeStrip } from "../dist/rectify.js";

const [pdfPath, pageArg] = process.argv.slice(2);
const p = Number(pageArg);
const canvas = (w, h) => createCanvas(w, h);
const doc = await pdfjs.getDocument({ data: new Uint8Array(fs.readFileSync(pdfPath)), verbosity: 0 }).promise;
const { s1, sDoc } = await pass1(doc, canvas);
const prep = await preparePage(doc, p, s1[p], sDoc, canvas);
const { systems } = detectSystems(prep.norm8);
const strips = systems.map((sy) => makeStrip(sy, prep.norm8.width, prep.norm8));
console.log(JSON.stringify({
  s1: s1[p], sRender: prep.sRender, deskew: prep.deskewDeg, render: [prep.renderWidth, prep.renderHeight],
  norm16: [prep.norm16.width, prep.norm16.height],
  systems: strips.map((s) => ({ xStart: s.xStart, width: s.width, heightNom: +s.heightNom.toFixed(4), shear: s.shear, u0: s.systemU0 })),
}));
