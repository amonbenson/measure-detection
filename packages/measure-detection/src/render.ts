// Stages 0 and 1 for a PDF: two-pass rendering with pdfjs-dist, staff space, deskew (mdet/prepare.py).

import type { PDFDocumentProxy } from "pdfjs-dist";
import { Gray } from "./image.js";
import { estimateStaffSpace } from "./normalize.js";
import { median, MIN_MODE_STRENGTH, normalizePage, PASS1_DPI, PASS2_TARGET_S, PreparedPage, toGray } from "./stages.js";

export { MIN_MODE_STRENGTH, NORM_S, PASS1_DPI, PASS2_TARGET_S } from "./stages.js";
export type { PreparedPage } from "./stages.js";

/** Minimal canvas interface (OffscreenCanvas in browsers, @napi-rs/canvas in Node). */
export interface CanvasLike {
  width: number;
  height: number;
  getContext(type: "2d"): any;
}
export type CanvasFactory = (width: number, height: number) => CanvasLike;

export const browserCanvas: CanvasFactory = (w, h) => new OffscreenCanvas(w, h) as unknown as CanvasLike;

export interface Rgba {
  width: number;
  height: number;
  data: Uint8ClampedArray;
}

/** Render one page on a white background; returns the RGBA pixels. */
export async function renderRgba(pdf: PDFDocumentProxy, pageIndex: number, scale: number, canvas: CanvasFactory): Promise<Rgba> {
  const page = await pdf.getPage(pageIndex + 1);
  try {
    const viewport = page.getViewport({ scale });
    const width = Math.round(viewport.width), height = Math.round(viewport.height);
    const c = canvas(width, height);
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);
    await page.render({ canvasContext: ctx, canvas: c, viewport } as any).promise;
    const data: Uint8ClampedArray = ctx.getImageData(0, 0, width, height).data;
    // Release the canvas backing store now instead of at the next garbage collection.
    c.width = c.height = 0;
    return { width, height, data };
  } finally {
    page.cleanup();
  }
}

export async function renderGray(pdf: PDFDocumentProxy, pageIndex: number, scale: number, canvas: CanvasFactory): Promise<Gray> {
  const r = await renderRgba(pdf, pageIndex, scale, canvas);
  return toGray(r.data, r.width, r.height);
}

/** Pass 1 for every page: staff space at moderate resolution. */
export async function pass1(pdf: PDFDocumentProxy, canvas: CanvasFactory, onPage?: (done: number) => void): Promise<{ s1: number[]; sDoc: number }> {
  const est = [];
  for (let p = 0; p < pdf.numPages; p++) {
    est.push(estimateStaffSpace(await renderGray(pdf, p, PASS1_DPI / 72, canvas)));
    onPage?.(p + 1);
  }
  const strong = est.filter((e) => e.modeStrength >= MIN_MODE_STRENGTH).map((e) => e.s);
  const sDoc = strong.length ? median(strong) : 8;
  return { s1: est.map((e) => (e.modeStrength >= MIN_MODE_STRENGTH ? e.s : sDoc)), sDoc };
}

/** Pass 2 for one page: render with s close to 18, measure s, deskew, normalize to s = 16 and 8. */
export async function preparePage(pdf: PDFDocumentProxy, p: number, s1: number, sDoc: number, canvas: CanvasFactory): Promise<PreparedPage> {
  const scale = ((PASS1_DPI / 72) * PASS2_TARGET_S) / s1;
  const g = await renderGray(pdf, p, scale, canvas);
  return normalizePage(g, p, estimateStaffSpace(g), s1, sDoc);
}
