// ONNX models via onnxruntime-web (WebGPU with WASM fallback) (mdet/infer.py).

import * as ort from "onnxruntime-web";
import { Gray } from "./image.js";
import { CHARSET, beamSearch } from "./ocr.js";
import { Strip } from "./rectify.js";

export const HEIGHT_BUCKETS = [192, 320, 448, 576, 704, 832];
export const TRAIN_WIDTH = 1216;

export interface ModelSources {
  measurenet: string | ArrayBuffer | Uint8Array;
  labelocr: string | ArrayBuffer | Uint8Array;
}

export interface Models {
  measurenet: ort.InferenceSession;
  labelocr: ort.InferenceSession;
  provider: string;
}

export async function loadModels(src: ModelSources, providers: string[] = ["webgpu", "wasm"]): Promise<Models> {
  let lastError: unknown;
  for (const provider of providers) {
    try {
      const opts: ort.InferenceSession.SessionOptions = { executionProviders: [provider] };
      const measurenet = await ort.InferenceSession.create(src.measurenet as any, opts);
      const labelocr = await ort.InferenceSession.create(src.labelocr as any, opts);
      return { measurenet, labelocr, provider };
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError;
}

function pyRound(x: number): number {
  return Math.abs(x % 1) === 0.5 ? 2 * Math.round(x / 2) : Math.round(x);
}

export function paddedShape(h: number, w: number): [number, number] {
  const hb = HEIGHT_BUCKETS.find((b) => b >= h) ?? Math.ceil(h / 128) * 128;
  return [hb, Math.max(TRAIN_WIDTH, Math.ceil(w / 128) * 128)];
}

/** MeasureNet input (2, H, W): ink and staff-band mask, zero padded to a shape bucket. */
export function stripInput(img: Gray, strip: Strip): { data: Float32Array; h: number; w: number } {
  const [hb, wb] = paddedShape(img.height, img.width);
  const data = new Float32Array(2 * hb * wb);
  for (let y = 0; y < img.height; y++) for (let x = 0; x < img.width; x++) data[y * wb + x] = 1 - img.data[y * img.width + x] / 255;
  for (const [vt, vb] of strip.staffBands) {
    const a = Math.max(0, pyRound(vt)), b = Math.min(img.height, pyRound(vb) + 1);
    for (let y = a; y < b; y++) data.fill(1, hb * wb + y * wb, hb * wb + y * wb + img.width);
  }
  return { data, h: hb, w: wb };
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

export interface BoundarySignals {
  startEdge: Float32Array; // start boundary, convention "system-edge"
  startHeader: Float32Array; // start boundary, convention "after-header"
  barline: Float32Array;
  offset: Float32Array;
}

export async function boundarySignals(models: Models, img: Gray, strip: Strip): Promise<BoundarySignals> {
  const { data, h, w } = stripInput(img, strip);
  const out = await models.measurenet.run({ input: new ort.Tensor("float32", data, [1, 2, h, w]) });
  const y = out.output.data as Float32Array;
  const cols = w / 2, c = Math.ceil(img.width / 2);
  return {
    startEdge: Float32Array.from(y.subarray(0, c), sigmoid),
    startHeader: Float32Array.from(y.subarray(cols, cols + c), sigmoid),
    barline: Float32Array.from(y.subarray(2 * cols, 2 * cols + c), sigmoid),
    offset: y.slice(3 * cols, 3 * cols + c),
  };
}

export async function readLabels(models: Models, inputs: Float32Array[]): Promise<[string, number][][]> {
  if (!inputs.length) return [];
  const batch = new Float32Array(inputs.length * 32 * 128);
  inputs.forEach((x, i) => batch.set(x, i * 32 * 128));
  const out = await models.labelocr.run({ input: new ort.Tensor("float32", batch, [inputs.length, 1, 32, 128]) });
  const lp = out.output.data as Float32Array;
  const T = 32, C = CHARSET.length + 1;
  return inputs.map((_, i) => beamSearch(lp.subarray(i * T * C, (i + 1) * T * C), T, C));
}
