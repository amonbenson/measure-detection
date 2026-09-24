// ONNX models via onnxruntime-web (WebGPU with WASM fallback) (mdet/infer.py).

import * as ort from "onnxruntime-web";
import { Gray } from "./image.js";
import { StripInput, stripInput } from "./input.js";
import { CHARSET, beamSearch } from "./ocr.js";
import { Strip } from "./rectify.js";
import { Signals } from "./stages.js";

export { HEIGHT_BUCKETS, TRAIN_WIDTH, paddedShape, stripInput } from "./input.js";

export interface ModelSources {
  measurenet: string | ArrayBuffer | Uint8Array;
  labelocr: string | ArrayBuffer | Uint8Array;
}

export interface Models {
  measurenet: ort.InferenceSession;
  labelocr: ort.InferenceSession;
  provider: string;
  /** Runs model calls one at a time: an InferenceSession must not run concurrently. */
  serial: <T>(fn: () => Promise<T>) => Promise<T>;
}

export function mutex(): <T>(fn: () => Promise<T>) => Promise<T> {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(fn: () => Promise<T>) => {
    const run = tail.then(fn, fn);
    tail = run.then(
      () => undefined,
      () => undefined,
    );
    return run;
  };
}

async function hasWebGpu(): Promise<boolean> {
  const gpu = typeof navigator !== "undefined" ? (navigator as any).gpu : undefined;
  if (!gpu) return false;
  try {
    return !!(await gpu.requestAdapter());
  } catch {
    return false;
  }
}

export interface LoadOptions {
  /** Execution providers in order of preference. Default: WebGPU if the browser has an adapter, else WASM. */
  providers?: string[];
  /** Run WASM inference in onnxruntime's own worker so it does not block the page (browsers only).
   *  Default: true when WASM is the first provider. */
  wasmProxy?: boolean;
}

export async function loadModels(src: ModelSources, opts: LoadOptions | string[] = {}): Promise<Models> {
  const o: LoadOptions = Array.isArray(opts) ? { providers: opts } : opts;
  const providers = o.providers ?? ((await hasWebGpu()) ? ["webgpu", "wasm"] : ["wasm"]);
  // The proxy must be set before onnxruntime initializes its WASM backend, and it is not available with WebGPU.
  const inBrowser = typeof window !== "undefined" && typeof Worker !== "undefined";
  if (providers[0] === "wasm" && inBrowser && (o.wasmProxy ?? true)) ort.env.wasm.proxy = true;
  let lastError: unknown;
  for (const provider of providers) {
    try {
      const opts: ort.InferenceSession.SessionOptions = { executionProviders: [provider] };
      const measurenet = await ort.InferenceSession.create(src.measurenet as any, opts);
      const labelocr = await ort.InferenceSession.create(src.labelocr as any, opts);
      return { measurenet, labelocr, provider, serial: mutex() };
    } catch (e) {
      lastError = e;
    }
  }
  throw lastError;
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

export type BoundarySignals = Signals;

/** MeasureNet on a prepared input; `width` is the rectified strip width (the input is padded beyond it). */
export async function measureNetSignals(models: Models, input: StripInput, width: number): Promise<BoundarySignals> {
  const { data, h, w } = input;
  const out = await models.serial(() => models.measurenet.run({ input: new ort.Tensor("float32", data, [1, 2, h, w]) }));
  const y = out.output.data as Float32Array;
  const cols = w / 2, c = Math.ceil(width / 2);
  // Bundles before the single start convention have 4 channels (start at the system edge, start after
  // the header, barline, offset); their after-header start channel is the one used now.
  const k = out.output.dims[1] === 4 ? 1 : 0;
  return {
    start: Float32Array.from(y.subarray(k * cols, k * cols + c), sigmoid),
    barline: Float32Array.from(y.subarray((k + 1) * cols, (k + 1) * cols + c), sigmoid),
    offset: y.slice((k + 2) * cols, (k + 2) * cols + c),
  };
}

export async function boundarySignals(models: Models, img: Gray, strip: Strip): Promise<BoundarySignals> {
  return measureNetSignals(models, stripInput(img, strip), img.width);
}

export async function readLabels(models: Models, inputs: Float32Array[]): Promise<[string, number][][]> {
  if (!inputs.length) return [];
  const batch = new Float32Array(inputs.length * 32 * 128);
  inputs.forEach((x, i) => batch.set(x, i * 32 * 128));
  const out = await models.serial(() => models.labelocr.run({ input: new ort.Tensor("float32", batch, [inputs.length, 1, 32, 128]) }));
  const lp = out.output.data as Float32Array;
  const T = 32, C = CHARSET.length + 1;
  return inputs.map((_, i) => beamSearch(lp.subarray(i * T * C, (i + 1) * T * C), T, C));
}
