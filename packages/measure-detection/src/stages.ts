// Per-page CPU stages (mdet/prepare.py, mdet/pipeline.py). Pure functions without onnxruntime or pdf.js,
// so that they run in Web Workers as well as on the main thread with identical results.

import { decode } from "./boundaries.js";
import { Affine, Gray, halve, invertAffine } from "./image.js";
import { applyNormalization, estimateSkew, normalizationMatrix, StaffSpace } from "./normalize.js";
import { Strip, stripToNorm, rectify } from "./rectify.js";
import { crop, Candidate, ocrInput, textCandidates } from "./text.js";

export const PASS1_DPI = 150;
export const PASS2_TARGET_S = 18;
export const NORM_S = 16;
export const MIN_MODE_STRENGTH = 0.05;

export type Point = [number, number];

export interface Measure {
  page: number;
  quad: Point[]; // page fractions: top-left, top-right, bottom-right, bottom-left
  x0: number; // page x fraction of the left boundary on the system's centre line
  x1: number;
  label: string;
  boundaryConfidence: number;
  labelConfidence: number;
}

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

/** Luma of an RGBA render (canvas getImageData), rounded like the Python renderer. */
export function toGray(rgba: Uint8ClampedArray, width: number, height: number): Gray {
  const data = new Uint8Array(width * height);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j++) data[j] = Math.round(0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]);
  return { width, height, data };
}

export function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b), k = s.length >> 1;
  return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
}

/** Pass 2 after the staff space of the render is known: deskew and normalize to s = 16 and 8.
 *  sDoc is only used for pages without a staff. */
export function normalizePage(g: Gray, page: number, est: StaffSpace, s1: number, sDoc: number): PreparedPage {
  const hasStaff = est.modeStrength >= MIN_MODE_STRENGTH;
  const s2 = hasStaff ? est.s : (PASS2_TARGET_S * sDoc) / s1;
  const angle = hasStaff ? estimateSkew(g, s2) : 0;
  const { m, width, height } = normalizationMatrix(g.width, g.height, s2, angle, NORM_S);
  const norm16 = applyNormalization(g, m, width, height);
  return { page, norm16, norm8: halve(norm16), renderWidth: g.width, renderHeight: g.height, sRender: s2, deskewDeg: angle, hasStaff, matrix: m };
}

/** Page fractions of a norm8 point. */
function normToPage(p: PreparedPage, x: number, y: number): Point {
  const inv = invertAffine(p.matrix as Affine);
  const X = x * 2, Y = y * 2;
  return [(inv[0] * X + inv[1] * Y + inv[2]) / p.renderWidth, (inv[3] * X + inv[4] * Y + inv[5]) / p.renderHeight];
}

export function measuresFromBoundaries(p: PreparedPage, strip: Strip, us: number[], confs: number[]): Measure[] {
  const out: Measure[] = [];
  const vt = strip.margin, vb = strip.margin + strip.heightNom, vc = strip.margin + strip.heightNom / 2;
  const px = (u: number) => {
    const [nx, ny] = stripToNorm(strip, u, vc);
    return normToPage(p, nx, ny)[0];
  };
  const corner = (u: number, v: number) => {
    const [nx, ny] = stripToNorm(strip, u, v);
    return normToPage(p, nx, ny);
  };
  for (let k = 0; k + 1 < us.length; k++)
    out.push({
      page: p.page,
      quad: [corner(us[k], vt), corner(us[k + 1], vt), corner(us[k + 1], vb), corner(us[k], vb)],
      x0: px(us[k]),
      x1: px(us[k + 1]),
      label: "",
      boundaryConfidence: Math.min(confs[k], confs[k + 1]),
      labelConfidence: 1,
    });
  return out;
}

export interface Signals {
  start: Float32Array;
  barline: Float32Array;
  offset: Float32Array;
}

export interface FinishedSystem {
  us: number[];
  measures: Measure[];
  cands: Candidate[];
  ocr: Float32Array[]; // LabelOCR inputs, one per candidate
}

/** Stage 3 and the text part of Stage 4 for one system: boundaries -> measures, candidates and OCR inputs. */
export function finishSystem(p: PreparedPage, strip: Strip, sig: Signals, tauB: number): FinishedSystem {
  const bnd = decode(sig.start, sig.barline, sig.offset, 2, tauB);
  const us = bnd.map((b) => b[0]);
  const measures = measuresFromBoundaries(p, strip, us, bnd.map((b) => b[1]));
  if (!measures.length) return { us, measures, cands: [], ocr: [] };
  const im16 = rectify(p.norm16, strip, 2);
  const cands = textCandidates(im16, strip);
  return { us, measures, cands, ocr: cands.map((c) => ocrInput(crop(im16, c))) };
}
