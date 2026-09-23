#!/usr/bin/env node
// Renders PDF pages to 8-bit grayscale PNG with pdfjs-dist.
// The browser runtime uses the same library, so training and inference see the same pixels.
//
// Usage: node render.mjs <jobs.json>
// jobs.json: [{ "pdf": "path.pdf", "page": 0, "scale": 2.5, "out": "path.png" }, ...]
// "scale" is pixels per PDF point (72 dpi * scale = dpi). "page" is 0-based.
// Prints one JSON line per job: { out, width, height, pageWidthPt, pageHeightPt }.

import fs from "node:fs";
import path from "node:path";
import { createCanvas } from "@napi-rs/canvas";
import * as pdfjs from "pdfjs-dist/legacy/build/pdf.mjs";

const jobs = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const byPdf = new Map();
for (const job of jobs) {
  if (!byPdf.has(job.pdf)) byPdf.set(job.pdf, []);
  byPdf.get(job.pdf).push(job);
}

for (const [pdfPath, pdfJobs] of byPdf) {
  const data = new Uint8Array(fs.readFileSync(pdfPath));
  const task = pdfjs.getDocument({ data, verbosity: 0 });
  const doc = await task.promise;
  for (const job of pdfJobs) {
    const page = await doc.getPage(job.page + 1);
    const base = page.getViewport({ scale: 1 });
    const viewport = page.getViewport({ scale: job.scale });
    const width = Math.round(viewport.width);
    const height = Math.round(viewport.height);
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    await page.render({ canvasContext: ctx, canvas, viewport }).promise;
    const rgba = ctx.getImageData(0, 0, width, height).data;
    const gray = createCanvas(width, height);
    const gctx = gray.getContext("2d");
    const img = gctx.createImageData(width, height);
    for (let i = 0; i < rgba.length; i += 4) {
      // Rec. 601 luma, identical to the runtime conversion.
      const y = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
      img.data[i] = img.data[i + 1] = img.data[i + 2] = y;
      img.data[i + 3] = 255;
    }
    gctx.putImageData(img, 0, 0);
    fs.mkdirSync(path.dirname(job.out), { recursive: true });
    fs.writeFileSync(job.out, await gray.encode("png"));
    page.cleanup();
    console.log(JSON.stringify({ out: job.out, width, height, pageWidthPt: base.width, pageHeightPt: base.height }));
  }
  await task.destroy();
}
