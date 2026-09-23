// Stage 1: staff space estimation and deskew (mdet/normalize.py).

import { Affine, binarize, FloatImage, Gray, open, resizeArea, roundHalfEven, rotationMatrix, warpAffine, warpAffineF } from "./image.js";

export interface StaffSpace {
  s: number;
  lineThickness: number;
  modeStrength: number;
}

function refinedMode(h: Float64Array, lo: number): number {
  let k = lo;
  for (let i = lo; i < h.length; i++) if (h[i] > h[k]) k = i;
  let num = 0, den = 0;
  for (let i = Math.max(k - 1, 0); i <= Math.min(k + 1, h.length - 1); i++) {
    const v = i < lo ? 0 : h[i];
    num += v * i;
    den += v;
  }
  return num / Math.max(den, 1e-9);
}

/** s = mode of (black run + following white run) over vertical runs. */
export function estimateStaffSpace(g: Gray, maxLen = 120): StaffSpace {
  const bw = binarize(g);
  const { width: w, height: h, data } = bw;
  const black = new Float64Array(maxLen + 1);
  const pair = new Float64Array(2 * maxLen + 1);
  for (let x = 0; x < w; x++) {
    // Transitions in the column padded with 0 on both ends.
    let prev = 0;
    const trans: number[] = [];
    const kinds: number[] = [];
    for (let y = 0; y <= h; y++) {
      const v = y < h ? data[y * w + x] : 0;
      if (v !== prev) {
        trans.push(y);
        kinds.push(v ? 1 : -1);
        prev = v;
      }
    }
    for (let i = 0; i + 1 < trans.length; i++) {
      const len = trans[i + 1] - trans[i];
      if (kinds[i] === 1) {
        black[Math.min(len, maxLen)]++;
        if (i + 2 < trans.length) {
          const pl = len + (trans[i + 2] - trans[i + 1]);
          pair[Math.min(pl, 2 * maxLen)]++;
        }
      }
    }
  }
  black[maxLen] = 0;
  pair[2 * maxLen] = 0;
  const thickness = refinedMode(black, 1);
  const s = refinedMode(pair, 3);
  let total = 0;
  for (let i = 3; i < pair.length; i++) total += pair[i];
  return { s, lineThickness: thickness, modeStrength: pair[roundHalfEven(s)] / Math.max(total, 1) };
}

/** Dominant staff-line angle in degrees; projection energy maximized over rotations, coarse to fine. */
export function estimateSkew(g: Gray, s: number, maxDeg = 5): number {
  const f = 4 / s;
  const small = resizeArea(g, Math.max(1, roundHalfEven(g.width * f)), Math.max(1, roundHalfEven(g.height * f)));
  const lines = open(binarize(small), 9, 1);
  const img: FloatImage = { width: lines.width, height: lines.height, data: Float32Array.from(lines.data) };
  const cx = img.width / 2, cy = img.height / 2;
  const score = (deg: number) => {
    const r = warpAffineF(img, rotationMatrix(cx, cy, deg, 1), img.width, img.height);
    let e = 0;
    for (let y = 0; y < r.height; y++) {
      let p = 0;
      for (let x = 0; x < r.width; x++) p += r.data[y * r.width + x];
      e += p * p;
    }
    return e;
  };
  let step = 0.5;
  let grid: number[] = [];
  for (let a = -maxDeg; a <= maxDeg + 1e-9; a += step) grid.push(a);
  let best = 0;
  for (;;) {
    let bestScore = -Infinity;
    for (const a of grid) {
      const v = score(a);
      if (v > bestScore) {
        bestScore = v;
        best = a;
      }
    }
    if (step <= 0.02) break;
    step /= 4;
    grid = [];
    for (let k = -4; k <= 4; k++) grid.push(best + k * step);
  }
  return best;
}

/** Affine map render -> normalized page (scale to targetS, rotate about the centre) and output size. */
export function normalizationMatrix(width: number, height: number, s: number, angleDeg: number, targetS: number): { m: Affine; width: number; height: number } {
  const m = rotationMatrix(width / 2, height / 2, angleDeg, targetS / s);
  const pts = [
    [0, 0],
    [width, 0],
    [0, height],
    [width, height],
  ].map(([x, y]) => [m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5]]);
  const x0 = Math.min(...pts.map((p) => p[0])), y0 = Math.min(...pts.map((p) => p[1]));
  const x1 = Math.max(...pts.map((p) => p[0])), y1 = Math.max(...pts.map((p) => p[1]));
  m[2] -= x0;
  m[5] -= y0;
  return { m, width: Math.ceil(x1 - x0), height: Math.ceil(y1 - y0) };
}

export function applyNormalization(g: Gray, m: Affine, width: number, height: number): Gray {
  return warpAffine(g, m, width, height, 255);
}
