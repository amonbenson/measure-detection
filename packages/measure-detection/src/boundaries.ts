// Stage 3: boundary signals -> boundary positions (mdet/boundaries.py).

import { S } from "./staves.js";

/** Local maxima above threshold with greedy non-maximum suppression (stable order for ties). */
export function peaks1d(signal: ArrayLike<number>, threshold: number, minDist: number): number[] {
  const idx: number[] = [];
  for (let i = 0; i < signal.length; i++) if (signal[i] >= threshold) idx.push(i);
  idx.sort((a, b) => signal[b] - signal[a] || a - b);
  const blocked = new Uint8Array(signal.length);
  const r = Math.ceil(minDist);
  const taken: number[] = [];
  for (const i of idx) {
    if (blocked[i]) continue;
    taken.push(i);
    for (let k = Math.max(0, i - r); k <= Math.min(signal.length - 1, i + r); k++) blocked[k] = 1;
  }
  return taken.sort((a, b) => a - b);
}

/** Start boundary = strongest start peak; barlines = NMS peaks above tauB right of it. Returns [u, conf][]. */
export function decode(start: Float32Array, barline: Float32Array, offset: Float32Array, stride: number, tauB: number, minDist = 1.5 * S): [number, number][] {
  const pos = (i: number) => (i + offset[i]) * stride;
  let si = 0;
  for (let i = 1; i < start.length; i++) if (start[i] > start[si]) si = i;
  const out: [number, number][] = [[pos(si), start[si]]];
  for (const i of peaks1d(barline, tauB, minDist / stride)) {
    const u = pos(i);
    if (u > out[0][0] + minDist) out.push([u, barline[i]]);
  }
  return out;
}
