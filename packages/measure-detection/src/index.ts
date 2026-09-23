// measure-detection: find measures and measure numbers in sheet-music PDFs, client-side.

import type { PDFDocumentProxy } from "pdfjs-dist";
import { DecoderParams } from "./labelling.js";
import { loadModels, ModelSources, Models } from "./models.js";
import { detectPage, labelDocument, Measure, PipelineOptions, StartConvention } from "./pipeline.js";
import { browserCanvas, CanvasFactory, pass1, preparePage } from "./render.js";

export type { DecoderParams, Measure, ModelSources, StartConvention };
export type { CanvasFactory } from "./render.js";
export type ProgressStage = "scan" | "detect" | "label";

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
  /** Execution providers in order of preference. Default: WebGPU, then WASM. */
  providers?: string[];
}

export interface DetectOptions extends PipelineOptions {
  /** "system-edge" (default): the first measure of a system starts at the system's opening line.
   *  "after-header": it starts after clef, key and time signature. */
  startConvention?: StartConvention;
  /** Canvas factory for pdf.js rendering. Default: OffscreenCanvas. */
  canvas?: CanvasFactory;
  /** Progress per stage: "scan" (staff space of every page), "detect" (measures per page),
   *  "label" (reading and decoding measure numbers for the whole document). */
  onProgress?: (done: number, total: number, stage: ProgressStage) => void;
}

export class MeasureDetector {
  private constructor(private models: Models, private decoder: DecoderParams) {}

  static async create(config: DetectorConfig): Promise<MeasureDetector> {
    return new MeasureDetector(await loadModels(config.models, config.providers), config.decoder);
  }

  /** Execution provider in use ("webgpu" or "wasm"). */
  get provider(): string {
    return this.models.provider;
  }

  async detect(pdf: PDFDocumentProxy, opts: DetectOptions = {}): Promise<MeasureResult[]> {
    const canvas = opts.canvas ?? browserCanvas;
    const { s1, sDoc } = await pass1(pdf, canvas, (done) => opts.onProgress?.(done, pdf.numPages, "scan"));
    const pages = [];
    for (let p = 0; p < pdf.numPages; p++) {
      const page = await preparePage(pdf, p, s1[p], sDoc, canvas);
      pages.push({ page, systems: await detectPage(this.models, page, opts) });
      opts.onProgress?.(p + 1, pdf.numPages, "detect");
    }
    opts.onProgress?.(0, 1, "label");
    const measures = await labelDocument(this.models, this.decoder, pages);
    opts.onProgress?.(1, 1, "label");
    return measures.map(toResult);
  }
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
