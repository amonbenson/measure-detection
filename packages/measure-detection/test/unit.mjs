// Unit parity test: TypeScript primitives against Python results (test/vectors.json, from `mdet.testvectors`).

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { binarize, close, components, halve, open, polyfit, remap, resizeArea, warpAffine, warpAffineF, otsuThreshold } from "../dist/image.js";
import { estimateStaffSpace } from "../dist/normalize.js";
import { beamSearch } from "../dist/ocr.js";
import { decodeLabels } from "../dist/labelling.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const v = JSON.parse(fs.readFileSync(path.join(here, "vectors.json"), "utf8"));
const decoder = JSON.parse(fs.readFileSync(path.join(here, "..", "models", "decoder.json"), "utf8"));
const gray = (o) => ({ width: o.width, height: o.height, data: Uint8Array.from(o.data) });
const img = gray(v.img), page = gray(v.page);

let failed = 0;
function check(name, ok, detail = "") {
  console.log(`${ok ? "ok  " : "FAIL"} ${name}${detail ? ": " + detail : ""}`);
  if (!ok) failed++;
}
function sameArray(name, got, want, tol = 0) {
  let bad = 0, maxd = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got[i] - want[i]);
    maxd = Math.max(maxd, d);
    if (d > tol) bad++;
  }
  check(name, got.length === want.length && bad === 0, `${bad} of ${want.length} differ, max difference ${maxd}`);
}
const close9 = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

check("otsu threshold", otsuThreshold(page) === v.otsu, `${otsuThreshold(page)} vs ${v.otsu}`);
const bw = binarize(page);
check("binarize", bw.data.reduce((a, b) => a + b, 0) === v.binarize_sum);
sameArray("open 16x1", open(bw, 16, 1).data, v.open_16x1);
sameArray("open 1x16", open(bw, 1, 16).data, v.open_1x16);
sameArray("close 25x1", close(bw, 25, 1).data, v.close_25x1);
sameArray("resize area 17x13", resizeArea(img, 17, 13).data, v.resize_area_17x13);
sameArray("resize area 60x50", resizeArea(img, 60, 50).data, v.resize_area_60x50);
sameArray("halve", halve(img).data, v.halve);
sameArray("warp affine (8 bit)", warpAffine(img, v.warp_matrix, 40, 30, 255).data, v.warp_affine);
const f = { width: img.width, height: img.height, data: Float32Array.from(img.data, (x) => (x > 128 ? 1 : 0)) };
sameArray("warp affine (float, inverse map)", warpAffineF(f, [1, 0.02, -0.02 * 15, 0, 1, 0], 43, 31, true).data, v.warp_shear_f, 1e-5);
const r = v.remap_map;
sameArray("remap", remap(img, 43, 31, (x, y) => [Math.fround(Math.fround(x * r.ax) + r.bx), Math.fround(Math.fround(y * r.ay) + r.by)], 255).data, v.remap);
const comps = components(bw).map((c) => [c.x, c.y, c.w, c.h, c.area]).sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
check("connected components", JSON.stringify(comps) === JSON.stringify(v.components), `${comps.length} vs ${v.components.length}`);
const p2 = polyfit(v.polyfit.x, v.polyfit.y, 2), p1 = polyfit(v.polyfit.x, v.polyfit.y, 1);
check("polyfit", p2.every((c, i) => close9(c, v.polyfit.deg2[i])) && p1.every((c, i) => close9(c, v.polyfit.deg1[i])), `${p2} vs ${v.polyfit.deg2}`);
const ss = estimateStaffSpace(page);
check("staff space", close9(ss.s, v.staff_space.s) && close9(ss.lineThickness, v.staff_space.line_thickness) && close9(ss.modeStrength, v.staff_space.mode_strength), `${ss.s} vs ${v.staff_space.s}`);
const beam = beamSearch(Float32Array.from(v.beam.logp), 32, 64);
check("CTC beam search", beam.length === v.beam.result.length && beam.every(([s, q], i) => s === v.beam.result[i][0] && Math.abs(q - v.beam.result[i][1]) < 1e-6), JSON.stringify(beam.slice(0, 2)));
const dec = decodeLabels(decoder, v.decode.observations);
check("label decoder", dec.every(([s, q], i) => s === v.decode.result[i][0] && Math.abs(q - v.decode.result[i][1]) < 1e-9), JSON.stringify(dec));

console.log(failed ? `${failed} unit checks failed` : "all unit checks passed");
process.exit(failed ? 1 : 0);
