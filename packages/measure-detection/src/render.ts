// Stages 0 and 1 for a PDF: two-pass rendering with pdfjs-dist, staff space, deskew (mdet/prepare.py).

import type { PDFDocumentProxy } from "pdfjs-dist";
import { Affine, Gray, halve } from "./image.js";
import { applyNormalization, estimateSkew, estimateStaffSpace, normalizationMatrix } from "./normalize.js";

export const PASS1_DPI = 150;
export const PASS2_TARGET_S = 18;
export const NORM_S = 16;
export const MIN_MODE_STRENGTH = 0.05;

/** Minimal canvas interface (OffscreenCanvas in browsers, @napi-rs/canvas in Node). */
export interface CanvasLike {
  getContext(type: "2d"): any;
}
export type CanvasFactory = (width: number, height: number) => CanvasLike;

export const browserCanvas: CanvasFactory = (w, h) => new OffscreenCanvas(w, h) as unknown as CanvasLike;

export interface PreparedPage {
  page: number;
  norm16: Gray;
  norm8: Gray;
  renderWidth: number;
  renderHeight: number;
  sRender: number;
  deskewDeg: number;
  hasStaff: boolean;
  matrix: Affine; // render pixels -> norm16 pixels
}

export async function renderGray(pdf: PDFDocumentProxy, pageIndex: number, scale: number, canvas: CanvasFactory): Promise<Gray> {
  const page = await pdf.getPage(pageIndex + 1);
  const viewport = page.getViewport({ scale });
  const width = Math.round(viewport.width), height = Math.round(viewport.height);
  const c = canvas(width, height);
  const ctx = c.getContext("2d");
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  await page.render({ canvasContext: ctx, canvas: c, viewport } as any).promise;
  const rgba: Uint8ClampedArray = ctx.getImageData(0, 0, width, height).data;
  const data = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) data[j] = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
  page.cleanup();
  return { width, height, data };
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b), k = s.length >> 1;
  return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
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
  const est = estimateStaffSpace(g);
  const hasStaff = est.modeStrength >= MIN_MODE_STRENGTH;
  const s2 = hasStaff ? est.s : (PASS2_TARGET_S * sDoc) / s1;
  const angle = hasStaff ? estimateSkew(g, s2) : 0;
  const { m, width, height } = normalizationMatrix(g.width, g.height, s2, angle, NORM_S);
  const norm16 = applyNormalization(g, m, width, height);
  return { page: p, norm16, norm8: halve(norm16), renderWidth: g.width, renderHeight: g.height, sRender: s2, deskewDeg: angle, hasStaff, matrix: m };
}
