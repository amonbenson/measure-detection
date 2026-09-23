// Text candidates in the margins of a system strip (mdet/text.py).

import { binarize, components, Gray, resizeArea, roundHalfEven } from "./image.js";
import { S } from "./staves.js";
import { Strip } from "./rectify.js";

const S16 = 2 * S;
const MERGE_GAP = 0.6;
const MIN_HEIGHT = 0.7, MAX_HEIGHT = 2.6, MAX_WIDTH = 10;
const STAFF_CLEARANCE = 0.1;

export interface Candidate {
  u0: number;
  u1: number;
  v0: number;
  v1: number;
  side: "above" | "below";
  nComponents: number;
}

export const candidateUc = (c: Candidate) => (c.u0 + c.u1) / 2;

export function textCandidates(img16: Gray, strip: Strip): Candidate[] {
  const bw = binarize(img16);
  const h = bw.height;
  const staff0 = Math.trunc((strip.margin - STAFF_CLEARANCE * S) * 2);
  const staff1 = Math.trunc((strip.margin + strip.heightNom + STAFF_CLEARANCE * S) * 2);
  const bySide: Record<"above" | "below", number[][]> = { above: [], below: [] };
  for (const c of components(bw)) {
    if (c.area < 3) continue;
    if (c.y + c.h > staff0 && c.y < staff1) continue;
    if (c.y === 0 || c.y + c.h >= h) continue;
    if (c.h > MAX_HEIGHT * S16 || c.w > MAX_WIDTH * S16) continue;
    bySide[c.y + c.h <= staff0 ? "above" : "below"].push([c.x, c.y, c.x + c.w, c.y + c.h]);
  }
  const out: Candidate[] = [];
  for (const side of ["above", "below"] as const) {
    // Python sorts lists lexicographically.
    const comps = bySide[side].sort((p, q) => p[0] - q[0] || p[1] - q[1] || p[2] - q[2] || p[3] - q[3]);
    const groups: number[][][] = [];
    for (const c of comps) {
      let placed = false;
      for (const g of groups) {
        const gx1 = Math.max(...g.map((k) => k[2]));
        const gy0 = Math.min(...g.map((k) => k[1])), gy1 = Math.max(...g.map((k) => k[3]));
        const ov = Math.min(gy1, c[3]) - Math.max(gy0, c[1]);
        if (c[0] - gx1 <= MERGE_GAP * S16 && ov > -0.2 * S16 && (ov > 0.3 * Math.min(gy1 - gy0, c[3] - c[1]) || c[3] - c[1] < 0.4 * S16)) {
          g.push(c);
          placed = true;
          break;
        }
      }
      if (!placed) groups.push([c]);
    }
    for (const g of groups) {
      const tallest = Math.max(...g.map((k) => k[3] - k[1]));
      const x0 = Math.min(...g.map((k) => k[0])), y0 = Math.min(...g.map((k) => k[1]));
      const x1 = Math.max(...g.map((k) => k[2])), y1 = Math.max(...g.map((k) => k[3]));
      if (tallest < MIN_HEIGHT * S16 || tallest > MAX_HEIGHT * S16 || x1 - x0 > MAX_WIDTH * S16) continue;
      const n = g.filter((k) => k[3] - k[1] >= 0.5 * tallest).length;
      out.push({ u0: x0 / 2, u1: x1 / 2, v0: y0 / 2, v1: y1 / 2, side, nComponents: n });
    }
  }
  return out;
}

export function crop(img16: Gray, c: Candidate, pad = 0.5): Gray {
  const p = pad * S16;
  const x0 = Math.max(0, Math.trunc(c.u0 * 2 - p)), x1 = Math.min(img16.width, Math.trunc(Math.ceil(c.u1 * 2 + p)));
  const y0 = Math.max(0, Math.trunc(c.v0 * 2 - p)), y1 = Math.min(img16.height, Math.trunc(Math.ceil(c.v1 * 2 + p)));
  const out = new Uint8Array((x1 - x0) * (y1 - y0));
  for (let y = y0; y < y1; y++) out.set(img16.data.subarray(y * img16.width + x0, y * img16.width + x1), (y - y0) * (x1 - x0));
  return { width: x1 - x0, height: y1 - y0, data: out };
}

/** LabelOCR input (32 x 128, ink = 1). */
export function ocrInput(c: Gray, height = 32, width = 128): Float32Array {
  const nw = Math.max(1, Math.min(width, roundHalfEven((c.width * height) / Math.max(c.height, 1))));
  const r = resizeArea(c, nw, height);
  const out = new Float32Array(height * width);
  for (let y = 0; y < height; y++) for (let x = 0; x < nw; x++) out[y * width + x] = 1 - r.data[y * nw + x] / 255;
  return out;
}
