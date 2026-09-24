// Labelling part of the document pipeline (mdet/pipeline.py, mdet/predict.py, mdet/labelling.label_document).
// Readings are taken per page as soon as its text candidates exist, so page images can be freed early;
// assignment and decoding need the whole document and run at the end.

import { assignSystem, decodeLabels, DecoderParams, filterReadings, fitDocModel, Reading } from "./labelling.js";
import { expandRange } from "./labels.js";
import { Models, readLabels } from "./models.js";
import { Measure, Point } from "./stages.js";
import { S } from "./staves.js";
import { Candidate, candidateUc } from "./text.js";

export type { Measure, Point } from "./stages.js";

export interface PipelineOptions {
  tauB?: number;
}

/** One system with measures, ready for labelling. */
export interface LabelSystem {
  us: number[];
  measures: Measure[];
  readings: Reading[];
}

/** Stage 4 readings of one system's text candidates. */
export async function readSystem(models: Models, cands: Candidate[], ocr: Float32Array[]): Promise<Reading[]> {
  return filterReadings(cands, await readLabels(models, ocr));
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

/** Stages 4 (assignment) and 5 for a whole document; systems in reading order. Returns measures with labels. */
export function finishLabels(params: DecoderParams, systems: LabelSystem[]): Measure[] {
  let assigned = systems.map(({ us, readings }) => assignSystem(params, readings, us, null));
  const samples: [string, string, number][] = [];
  systems.forEach(({ us }, i) => {
    for (const [m, r] of assigned[i]) if (expandRange(r.hyps[0][0]) === null) samples.push([r.cand.side, m === 0 ? "first" : "other", (candidateUc(r.cand) - us[m]) / S]);
  });
  const doc = fitDocModel(samples);
  if (Object.keys(doc).length) assigned = systems.map(({ us, readings }) => assignSystem(params, readings, us, doc));
  const observations: ([string, number][] | null)[] = [];
  const out: Measure[] = [];
  systems.forEach((sy, i) => {
    sy.measures.forEach((m, k) => {
      const r = assigned[i].get(k);
      const rng = r ? expandRange(r.hyps[0][0]) : null;
      if (r && rng && r.hyps[0][1] > 0.5 && rng.length >= 2) {
        out.push(...splitMeasure(m, rng.length));
        // The range label names every measure of the multi-measure rest.
        observations.push(...rng.map((lab) => [[lab, r.hyps[0][1]]] as [string, number][]));
      } else {
        out.push({ ...m });
        const obs = r ? r.hyps.filter(([s]) => expandRange(s) === null) : [];
        observations.push(obs.length ? obs : null);
      }
    });
  });
  decodeLabels(params, observations).forEach(([lab, conf], i) => {
    out[i].label = lab;
    out[i].labelConfidence = conf;
  });
  return out;
}
