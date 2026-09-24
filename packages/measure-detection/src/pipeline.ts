// Page and document pipeline (mdet/pipeline.py, mdet/predict.py, mdet/labelling.label_document).

import { decode } from "./boundaries.js";
import { Affine, invertAffine } from "./image.js";
import { assignSystem, decodeLabels, DecoderParams, filterReadings, fitDocModel, Reading } from "./labelling.js";
import { expandRange } from "./labels.js";
import { boundarySignals, Models, readLabels } from "./models.js";
import { makeStrip, rectify, Strip, stripToNorm } from "./rectify.js";
import { PreparedPage } from "./render.js";
import { detectSystems, S, System } from "./staves.js";
import { candidateUc, crop, ocrInput, textCandidates } from "./text.js";

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

interface PageSystem {
  system: System;
  strip: Strip;
  us: number[];
  measures: Measure[];
}

/** Page fractions of a norm8 point. */
function normToPage(p: PreparedPage, x: number, y: number): Point {
  const inv = invertAffine(p.matrix as Affine);
  const X = x * 2, Y = y * 2;
  return [(inv[0] * X + inv[1] * Y + inv[2]) / p.renderWidth, (inv[3] * X + inv[4] * Y + inv[5]) / p.renderHeight];
}

function measuresFromBoundaries(p: PreparedPage, strip: Strip, us: number[], confs: number[]): Measure[] {
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

export interface PipelineOptions {
  tauB?: number;
}

/** Stages 2 and 3 for one prepared page. */
export async function detectPage(models: Models, p: PreparedPage, opts: PipelineOptions = {}): Promise<PageSystem[]> {
  if (!p.hasStaff) return [];
  const tauB = opts.tauB ?? 0.4;
  const { systems } = detectSystems(p.norm8);
  const out: PageSystem[] = [];
  for (const sy of systems) {
    const strip = makeStrip(sy, p.norm8.width, p.norm8);
    const img = rectify(p.norm8, strip);
    const sig = await boundarySignals(models, img, strip);
    const bnd = decode(sig.start, sig.barline, sig.offset, 2, tauB);
    const us = bnd.map((b) => b[0]);
    out.push({ system: sy, strip, us, measures: measuresFromBoundaries(p, strip, us, bnd.map((b) => b[1])) });
  }
  return out;
}

function splitMeasure(m: Measure, n: number): Measure[] {
  const lerp = (a: Point, b: Point, t: number): Point => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  const q = m.quad;
  const out: Measure[] = [];
  for (let k = 0; k < n; k++) {
    const a = k / n, b = (k + 1) / n;
    out.push({ ...m, quad: [lerp(q[0], q[1], a), lerp(q[0], q[1], b), lerp(q[3], q[2], b), lerp(q[3], q[2], a)], x0: m.x0 + (m.x1 - m.x0) * a, x1: m.x0 + (m.x1 - m.x0) * b });
  }
  return out;
}

/** Stages 4 and 5 for a whole document; returns measures in reading order with labels. */
export async function labelDocument(models: Models, params: DecoderParams, pages: { page: PreparedPage; systems: PageSystem[] }[]): Promise<Measure[]> {
  const systems: { ps: PageSystem; readings: Reading[] }[] = [];
  for (const { page, systems: pss } of pages)
    for (const ps of pss) {
      if (!ps.measures.length) continue;
      const im16 = rectify(page.norm16, ps.strip, 2);
      const cands = textCandidates(im16, ps.strip);
      const hyps = await readLabels(models, cands.map((c) => ocrInput(crop(im16, c))));
      systems.push({ ps, readings: filterReadings(cands, hyps) });
    }
  let assigned = systems.map(({ ps, readings }) => assignSystem(params, readings, ps.us, null));
  const samples: [string, string, number][] = [];
  systems.forEach(({ ps }, i) => {
    for (const [m, r] of assigned[i]) if (expandRange(r.hyps[0][0]) === null) samples.push([r.cand.side, m === 0 ? "first" : "other", (candidateUc(r.cand) - ps.us[m]) / S]);
  });
  const doc = fitDocModel(samples);
  if (Object.keys(doc).length) assigned = systems.map(({ ps, readings }) => assignSystem(params, readings, ps.us, doc));
  const observations: ([string, number][] | null)[] = [];
  const out: Measure[] = [];
  systems.forEach(({ ps }, i) => {
    const measures: Measure[] = [];
    ps.measures.forEach((m, k) => {
      const r = assigned[i].get(k);
      const rng = r ? expandRange(r.hyps[0][0]) : null;
      if (r && rng && r.hyps[0][1] > 0.5 && rng.length >= 2) {
        const parts = splitMeasure(m, rng.length);
        measures.push(...parts);
        // The range label names every measure of the multi-measure rest.
        observations.push(...rng.map((lab) => [[lab, r.hyps[0][1]]] as [string, number][]));
      } else {
        measures.push(m);
        const obs = r ? r.hyps.filter(([s]) => expandRange(s) === null) : [];
        observations.push(obs.length ? obs : null);
      }
    });
    ps.measures = measures;
    out.push(...measures);
  });
  decodeLabels(params, observations).forEach(([lab, conf], i) => {
    out[i].label = lab;
    out[i].labelConfidence = conf;
  });
  return out;
}
