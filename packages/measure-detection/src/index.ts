// measure-detection: find measures and measure numbers in sheet-music PDFs, client-side.

import type { PDFDocumentProxy } from "pdfjs-dist";
import { createExecutor, Executor, InlineExecutor, WorkerFailure } from "./executor.js";
import { DecoderParams } from "./labelling.js";
import { loadModels, measureNetSignals, ModelSources, Models, mutex } from "./models.js";
import { PrepareResult } from "./ops.js";
import { finishLabels, LabelSystem, Measure, PipelineOptions, readSystem } from "./pipeline.js";
import { browserCanvas, CanvasFactory, renderRgba } from "./render.js";
import { FinishedSystem, median, MIN_MODE_STRENGTH, PASS1_DPI, PASS2_TARGET_S } from "./stages.js";

export type { DecoderParams, Measure, ModelSources };
export type { CanvasFactory } from "./render.js";
export type ProgressStage = "detect" | "label";

/** One detected measure, in the training JSON schema extended with quad and confidences. */
export interface MeasureResult {
  layout: { page: number; x: number; y: number; width: number; height: number };
  value: string;
  quad: [number, number][];
  confidence: number;
  boundaryConfidence: number;
  labelConfidence: number;
}

export interface DetectorConfig {
  models: ModelSources;
  decoder: DecoderParams;
  /** Execution providers in order of preference. Default: WebGPU if the browser has an adapter, else WASM. */
  providers?: string[];
  /** Run WASM inference in onnxruntime's own worker so the page stays responsive. Default: true. */
  wasmProxy?: boolean;
  /** Web Workers for the image processing. Default: CPU cores - 1, at most 4, fewer on devices with
   *  little memory. 0 processes pages on the calling thread (also the fallback when workers cannot start). */
  workers?: number;
  /** Custom worker factory, for bundlers that need their own worker URL. It must load `measure-detection/worker.js` (`dist/worker.js`). */
  createWorker?: () => Worker;
}

/** The first measure of a system starts after clef, key and time signature (docs/CONVENTIONS.md). */
export interface DetectOptions extends PipelineOptions {
  /** Canvas factory for pdf.js rendering. Default: OffscreenCanvas. */
  canvas?: CanvasFactory;
  /** Progress per stage: "detect" (pages done, in any order), "label" (reading order and decoding of
   *  measure numbers for the whole document). */
  onProgress?: (done: number, total: number, stage: ProgressStage) => void;
  /** Called when a page is detected, with its measures before labelling (value ""). Boxes can change
   *  once more at the end: a multi-measure rest is split into its measures when its range label is read. */
  onPageMeasures?: (page: number, measures: MeasureResult[]) => void;
}

export class MeasureDetector {
  private executor: Promise<Executor> | null = null;
  private inline: Executor | null = null;
  private runs = 0;

  private constructor(
    private models: Models,
    private decoder: DecoderParams,
    private config: DetectorConfig,
  ) {}

  static async create(config: DetectorConfig): Promise<MeasureDetector> {
    const models = await loadModels(config.models, { providers: config.providers, wasmProxy: config.wasmProxy });
    return new MeasureDetector(models, config.decoder, config);
  }

  /** Execution provider in use ("webgpu" or "wasm"). */
  get provider(): string {
    return this.models.provider;
  }

  /** How pages are processed: "workers" (Web Workers) or "inline" (calling thread). Available after the first detect. */
  async processing(): Promise<{ kind: string; workers: number }> {
    const ex = await this.getExecutor();
    return { kind: ex.kind, workers: ex.kind === "workers" ? ex.size : 0 };
  }

  /** Stops the workers. The detector can still be used afterwards (they start again on demand). */
  dispose(): void {
    this.executor?.then((ex) => ex.close());
    this.executor = null;
  }

  private getExecutor(): Promise<Executor> {
    this.executor ??= createExecutor(this.config.workers, this.config.createWorker);
    return this.executor;
  }

  async detect(pdf: PDFDocumentProxy, opts: DetectOptions = {}): Promise<MeasureResult[]> {
    const canvas = opts.canvas ?? browserCanvas;
    const tauB = opts.tauB ?? 0.4;
    const run = ++this.runs;
    let ex = await this.getExecutor();
    const n = pdf.numPages;
    const pages: LabelSystem[][] = new Array(n);
    const strong = new Map<number, number>(); // page -> pass-1 staff space, pages with a clear staff
    const weak: number[] = []; // pages whose pass-1 staff space is unclear: need the document's staff space
    let done = 0;
    // One render at a time: a rendered scan page is large, and pdf.js already uses its own worker.
    const render = mutex();

    // Pass 1 (unless the staff space is given), pass 2 and all stages of one page.
    const processPage = async (p: number, sGiven: number | null): Promise<void> => {
      const key = `${run}:${p}`;
      let s = sGiven ?? strong.get(p) ?? null; // a retried page keeps its pass-1 result
      if (s === null) {
        const r1 = await render(() => renderRgba(pdf, p, PASS1_DPI / 72, canvas));
        const est = await ex.call<{ s: number; modeStrength: number }>("staffSpace", r1, [r1.data.buffer]);
        if (est.modeStrength < MIN_MODE_STRENGTH) {
          weak.push(p);
          return;
        }
        strong.set(p, est.s);
        s = est.s;
      }
      const scale = ((PASS1_DPI / 72) * PASS2_TARGET_S) / s;
      const r2 = await render(() => renderRgba(pdf, p, scale, canvas));
      try {
        const prep = await ex.call<PrepareResult>("prepare", { key, page: p, s1: s, ...r2 }, [r2.data.buffer], key);
        const signals = [];
        for (const sy of prep.systems) signals.push(await measureNetSignals(this.models, sy.input, sy.width));
        const buffers = signals.flatMap((g) => [g.start.buffer, g.barline.buffer, g.offset.buffer]);
        const finished = prep.systems.length ? await ex.call<FinishedSystem[]>("finish", { key, signals, tauB }, buffers, key) : [];
        const systems: LabelSystem[] = [];
        for (const f of finished) if (f.measures.length) systems.push({ us: f.us, measures: f.measures, readings: await readSystem(this.models, f.cands, f.ocr) });
        pages[p] = systems;
      } finally {
        ex.release(key);
      }
      opts.onPageMeasures?.(p, pages[p].flatMap((sy) => sy.measures).map(toResult));
      opts.onProgress?.(++done, n, "detect");
    };

    // A worker that stops (for example killed for memory) loses its pages; retry each once, on the
    // remaining workers or, if none is left, on this thread.
    const withRetry = async (p: number, s: number | null) => {
      try {
        await processPage(p, s);
      } catch (err) {
        if (!(err instanceof WorkerFailure)) throw err;
        if (ex.size === 0) ex = this.inline ??= new InlineExecutor();
        await processPage(p, s);
      }
    };

    try {
      const lanes = Math.max(1, ex.size) + (ex.kind === "workers" ? 1 : 0);
      await runLimited(range(n), lanes, (p) => withRetry(p, null));
      if (weak.length) {
        const sDoc = strong.size ? median([...strong.values()]) : 8;
        await runLimited(weak.sort((a, b) => a - b), lanes, (p) => withRetry(p, sDoc));
      }
    } catch (err) {
      ex.call("drop", { prefix: `${run}:` }).catch(() => undefined);
      throw err;
    }
    opts.onProgress?.(0, 1, "label");
    const measures = finishLabels(this.decoder, pages.flat());
    opts.onProgress?.(1, 1, "label");
    return measures.map(toResult);
  }
}

function range(n: number): number[] {
  return Array.from({ length: n }, (_, i) => i);
}

/** Runs fn over items with at most `limit` in flight, in item order; stops starting new items after an error. */
async function runLimited<T>(items: T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
  let next = 0;
  let failed: { err: unknown } | null = null;
  const lane = async () => {
    while (next < items.length && !failed) {
      const item = items[next++];
      try {
        await fn(item);
      } catch (err) {
        failed ??= { err };
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, lane));
  if (failed) throw (failed as { err: unknown }).err;
}

function toResult(m: Measure): MeasureResult {
  const xs = m.quad.map((q) => q[0]), ys = m.quad.map((q) => q[1]);
  const x = Math.min(...xs), y = Math.min(...ys);
  return {
    layout: { page: m.page, x, y, width: Math.max(...xs) - x, height: Math.max(...ys) - y },
    value: m.label,
    quad: m.quad,
    confidence: Math.min(m.boundaryConfidence, m.labelConfidence),
    boundaryConfidence: m.boundaryConfidence,
    labelConfidence: m.labelConfidence,
  };
}
