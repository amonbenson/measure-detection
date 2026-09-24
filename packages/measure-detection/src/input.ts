// MeasureNet input tensors (mdet/infer.py strip_input). No onnxruntime import: also used in workers.

import { Gray } from "./image.js";
import { Strip } from "./rectify.js";

export const HEIGHT_BUCKETS = [192, 320, 448, 576, 704, 832];
export const TRAIN_WIDTH = 1216;

function pyRound(x: number): number {
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x);
}

export function paddedShape(h: number, w: number): [number, number] {
  const hb = HEIGHT_BUCKETS.find((b) => b >= h) ?? Math.ceil(h / 128) * 128;
  return [hb, Math.max(TRAIN_WIDTH, Math.ceil(w / 128) * 128)];
}

export interface StripInput {
  data: Float32Array;
  h: number;
  w: number;
}

/** MeasureNet input (2, H, W): ink and staff-band mask, zero padded to a shape bucket. */
export function stripInput(img: Gray, strip: Strip): StripInput {
  const [hb, wb] = paddedShape(img.height, img.width);
  const data = new Float32Array(2 * hb * wb);
  for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) data[y * wb + x] = 1 - img.data[y * img.width + x] / 255;
  for (const [vt, vb] of strip.staffBands) {
    const a = Math.max(0, pyRound(vt)), b = Math.min(img.height, pyRound(vb) + 1);
    for (let y = a; y < b; y++) data.fill(1, hb * wb + y * wb, hb * wb + y * wb + img.width);
  }
  return { data, h: hb, w: wb };
}
