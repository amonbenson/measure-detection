// Stage 2: rectify a system into a straight horizontal strip with per-system shear (mdet/rectify.py).

import { binarize, FloatImage, Gray, open, polyval, remap, warpAffineF } from "./image.js";
import { S, System, systemBottom, systemTop, systemXc, yBottom, yTop } from "./staves.js";

export const MARGIN = 6 * S;
export const LEFT_PAD = 3 * S;
export const RIGHT_PAD = 2 * S;

export interface Strip {
  xStart: number;
  width: number;
  margin: number;
  heightNom: number;
  top: number[];
  bottom: number[];
  staffBands: [number, number][];
  staffSpacings: number[];
  systemU0: number;
  systemU1: number;
  shear: number;
}

export const stripHeight = (s: Strip) => pyRound(s.heightNom + 2 * s.margin);
export const vCentre = (s: Strip) => s.margin + s.heightNom / 2;

export function stripToNorm(s: Strip, u: number, v: number): [number, number] {
  const x = s.xStart + u + s.shear * (v - vCentre(s));
  const t = polyval(s.top, x), b = polyval(s.bottom, x);
  return [x, t + ((v - s.margin) * (b - t)) / s.heightNom];
}

export function normToStrip(s: Strip, x: number, y: number): [number, number] {
  const t = polyval(s.top, x), b = polyval(s.bottom, x);
  const v = s.margin + ((y - t) * s.heightNom) / (b - t);
  return [x - s.xStart - s.shear * (v - vCentre(s)), v];
}

/** Sample the strip from a page image; scale 1 for norm8, 2 for norm16. */
export function rectify(image: Gray, s: Strip, scale = 1): Gray {
  const w = s.width * scale, h = stripHeight(s) * scale;
  return remap(image, w, h, (x, y) => {
    const u = (x + 0.5) / scale - 0.5, v = (y + 0.5) / scale - 0.5;
    const [nx, ny] = stripToNorm(s, u, v);
    return [(nx + 0.5) * scale - 0.5, (ny + 0.5) * scale - 0.5];
  });
}

/** Python int(round(x)) (half to even). */
function pyRound(x: number): number {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : r;
}

export function estimateShear(image: Gray, s: Strip, maxShear = 0.035, step = 0.0025): number {
  const img = rectify(image, s);
  const a = Math.trunc(s.margin), b = Math.trunc(s.margin + s.heightNom);
  const band: Gray = { width: img.width, height: b - a, data: img.data.slice(a * img.width, b * img.width) };
  const vert = open(binarize(band), 1, Math.trunc(2 * S));
  const f: FloatImage = { width: vert.width, height: vert.height, data: Float32Array.from(vert.data) };
  const vc = vCentre(s) - a;
  const n = Math.round((2 * maxShear) / step) + 1;
  const ks: number[] = [], scores: number[] = [];
  for (let i = 0; i < n; i++) {
    const k = -maxShear + i * step;
    const sh = warpAffineF(f, [1, k, -k * vc, 0, 1, 0], f.width, f.height, true);
    let score = 0;
    for (let x = 0; x < sh.width; x++) {
      let p = 0;
      for (let y = 0; y < sh.height; y++) p += sh.data[y * sh.width + x];
      score += p * p;
    }
    ks.push(k);
    scores.push(score);
  }
  // Near-ties (within 0.5 %, below the measurement noise of one shear step) go to the smallest shear,
  // so that last-digit differences between implementations cannot change the result.
  const best = Math.max(...scores);
  let pick = Infinity;
  ks.forEach((k, i) => {
    if (scores[i] >= best * (1 - 5e-3) && Math.abs(k) < Math.abs(pick)) pick = k;
  });
  return pick;
}

export function makeStrip(system: System, pageWidth: number, image?: Gray): Strip {
  const xs0 = Math.max(0, system.x0 - LEFT_PAD);
  const xs1 = Math.min(pageWidth, system.x1 + RIGHT_PAD);
  const xc = systemXc(system);
  const topC = systemTop(system, xc), botC = systemBottom(system, xc);
  const hNom = Math.max(4, botC - topC);
  const k = hNom / (botC - topC);
  const s: Strip = {
    xStart: xs0,
    width: Math.ceil(xs1 - xs0),
    margin: MARGIN,
    heightNom: hNom,
    top: system.staves[0].top,
    bottom: system.staves[system.staves.length - 1].bottom,
    staffBands: system.staves.map((st) => [MARGIN + (yTop(st, xc) - topC) * k, MARGIN + (yBottom(st, xc) - topC) * k] as [number, number]),
    staffSpacings: system.staves.map((st) => st.spacing),
    systemU0: system.x0 - xs0,
    systemU1: system.x1 - xs0,
    shear: 0,
  };
  if (image) s.shear = estimateShear(image, s);
  return s;
}
