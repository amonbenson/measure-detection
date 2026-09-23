// Stages 4 and 5: assign OCR readings to measures and decode the label sequence (mdet/labelling.py).

import { expandRange, parseLabel, successors, TRANSITIONS, transitionType } from "./labels.js";
import { S } from "./staves.js";
import { Candidate, candidateUc } from "./text.js";

export interface DecoderParams {
  transitions: Record<string, number>;
  first_labels: Record<string, number>;
  offsets: Record<string, number[]>;
}

export interface Reading {
  cand: Candidate;
  hyps: [string, number][];
}

const UNASSIGNED_COST = 6.0;
export const NOT_A_NUMBER = 0.5;
const EMISSION_FLOOR = 1e-3;
const BEAM = 24;
// A free jump can go to any label; its type probability is shared by this many plausible targets.
const FREE_JUMP_TARGETS = 100;

/** Keep readings that are numbers (drop "" and unparseable hypotheses), as mdet.labelling.read_system. */
export function filterReadings(cands: Candidate[], hyps: [string, number][][]): Reading[] {
  const out: Reading[] = [];
  cands.forEach((c, i) => {
    const pEmpty = hyps[i].filter(([s]) => s === "").reduce((a, [, q]) => a + q, 0);
    const h = hyps[i].filter(([s]) => s && (parseLabel(s) !== null || expandRange(s) !== null));
    if (pEmpty > NOT_A_NUMBER || h.length === 0) return;
    out.push({ cand: c, hyps: h });
  });
  return mergeSuffixes(out);
}

/** Join a number reading and a following single-letter reading printed with a space (`30 E`). */
export function mergeSuffixes(readings: Reading[], maxGap = 1.2): Reading[] {
  const sorted = readings.map((r, i) => [r, i] as const).sort((a, b) => a[0].cand.u0 - b[0].cand.u0 || a[1] - b[1]).map((p) => p[0]);
  const out: Reading[] = [];
  for (const r of sorted) {
    const prev = out[out.length - 1];
    const top = r.hyps[0][0];
    if (
      prev &&
      top.length === 1 &&
      /^[A-Za-z]$/.test(top) &&
      /^\d+$/.test(prev.hyps[0][0]) &&
      prev.cand.side === r.cand.side &&
      r.cand.u0 - prev.cand.u1 >= 0 &&
      r.cand.u0 - prev.cand.u1 <= maxGap * S &&
      Math.min(prev.cand.v1, r.cand.v1) - Math.max(prev.cand.v0, r.cand.v0) > 0
    ) {
      const hyps: [string, number][] = [];
      for (const [x, qx] of prev.hyps.slice(0, 3)) for (const [y, qy] of r.hyps.slice(0, 3)) hyps.push([x + y, qx * qy]);
      hyps.sort((a, b) => b[1] - a[1]);
      const a = prev.cand, b = r.cand;
      out[out.length - 1] = {
        cand: { u0: a.u0, u1: b.u1, v0: Math.min(a.v0, b.v0), v1: Math.max(a.v1, b.v1), side: a.side, nComponents: a.nComponents + b.nComponents },
        hyps: hyps.slice(0, 5),
      };
    } else out.push(r);
  }
  return out;
}

function kdeLogpdf(samples: number[], x: number, bw = 0.25): number {
  if (!samples.length) return Math.log(1e-4);
  let p = 0;
  for (const s of samples) {
    const z = (x - s) / bw;
    p += Math.exp(-0.5 * z * z);
  }
  p /= samples.length * bw * Math.sqrt(2 * Math.PI);
  return Math.log(Math.max(p, 1e-6));
}

const gaussLogpdf = (mu: number, sg: number, x: number) => -0.5 * ((x - mu) / sg) ** 2 - Math.log(sg * Math.sqrt(2 * Math.PI));

type DocModel = Record<string, [number, number]>;

function offsetCost(params: DecoderParams, dx: number, side: string, pos: string, doc: DocModel | null): number {
  const key = `${side}/${pos}`;
  const samples = params.offsets[key] ?? [];
  let lp = samples.length ? kdeLogpdf(samples, dx) : Math.log(1e-4);
  if (doc && doc[key]) {
    const [mu, sg] = doc[key];
    lp = Math.log(0.2 * Math.exp(lp) + 0.8 * Math.exp(gaussLogpdf(mu, sg, dx)) + 1e-12);
  }
  return -lp;
}

/** Minimum-cost assignment for a rectangular cost matrix (rows <= cols). Returns column per row. */
export function hungarian(cost: number[][]): number[] {
  const n = cost.length, m = cost[0]?.length ?? 0;
  const INF = Infinity;
  const u = new Array(n + 1).fill(0), v = new Array(m + 1).fill(0), p = new Array(m + 1).fill(0), way = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(m + 1).fill(INF), used = new Array(m + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= m; j++)
        if (!used[j]) {
          const cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j] < delta) {
            delta = minv[j];
            j1 = j;
          }
        }
      for (let j = 0; j <= m; j++)
        if (used[j]) {
          u[p[j]] += delta;
          v[j] -= delta;
        } else minv[j] -= delta;
      j0 = j1;
    } while (p[j0] !== 0);
    do {
      const j1 = way[j0];
      p[j0] = p[j1];
      j0 = j1;
    } while (j0);
  }
  const res = new Array(n).fill(-1);
  for (let j = 1; j <= m; j++) if (p[j]) res[p[j] - 1] = j - 1;
  return res;
}

export function assignSystem(params: DecoderParams, readings: Reading[], us: number[], doc: DocModel | null): Map<number, Reading> {
  const nM = us.length - 1;
  const out = new Map<number, Reading>();
  if (nM <= 0 || !readings.length) return out;
  const cost = readings.map((rd, r) => {
    const row = new Array(nM + readings.length).fill(1e6);
    const uc = candidateUc(rd.cand);
    const isRange = expandRange(rd.hyps[0][0]) !== null;
    for (let m = 0; m < nM; m++) {
      if (isRange) row[m] = us[m] - S <= uc && uc <= us[m + 1] + S ? 1 : 1e6;
      else row[m] = offsetCost(params, (uc - us[m]) / S, rd.cand.side, m === 0 ? "first" : "other", doc);
    }
    row[nM + r] = UNASSIGNED_COST;
    return row;
  });
  const cols = hungarian(cost);
  cols.forEach((c, r) => {
    if (c >= 0 && c < nM && cost[r][c] < UNASSIGNED_COST) out.set(c, readings[r]);
  });
  return out;
}

export function fitDocModel(samples: [string, string, number][]): DocModel {
  const by: Record<string, number[]> = {};
  for (const [side, pos, dx] of samples) (by[`${side}/${pos}`] ??= []).push(dx);
  const model: DocModel = {};
  const median = (v: number[]) => {
    const s = [...v].sort((a, b) => a - b), k = s.length >> 1;
    return s.length % 2 ? s[k] : (s[k - 1] + s[k]) / 2;
  };
  for (const [k, v] of Object.entries(by)) {
    if (v.length < 5) continue;
    const med = median(v);
    model[k] = [med, Math.max(0.3, 1.4826 * median(v.map((x) => Math.abs(x - med))))];
  }
  return model;
}

const logaddexp = (a: number, b: number) => (a === -Infinity ? b : b === -Infinity ? a : Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b))));

/** Viterbi + forward-backward over the pruned candidate lattice. Returns [label, posterior] per measure. */
export function decodeLabels(params: DecoderParams, observations: ([string, number][] | null)[]): [string, number][] {
  const lt: Record<string, number> = {};
  for (const t of TRANSITIONS) lt[t] = Math.log(params.transitions[t]);
  lt["free_jump"] -= Math.log(FREE_JUMP_TARGETS);
  const first = params.first_labels;
  const n = observations.length;
  if (!n) return [];
  const emission = (i: number, lab: string) => {
    const obs = observations[i];
    if (!obs || !obs.length) return 0;
    for (const [s, q] of obs) if (s === lab) return Math.log(Math.max(q, EMISSION_FLOOR));
    return Math.log(EMISSION_FLOOR);
  };
  const states: string[][] = [], vit: number[][] = [], fwd: number[][] = [], back: number[][] = [], trans: number[][][] = [];
  for (let i = 0; i < n; i++) {
    let cand: string[] = [];
    if (observations[i]) cand.push(...observations[i]!.map(([s]) => s));
    if (i === 0) cand.push(...Object.keys(first));
    else {
      for (const p of states[i - 1]) cand.push(...successors(p));
      cand.push("1");
    }
    cand = [...new Set(cand.filter((c) => parseLabel(c) !== null))];
    const em = cand.map((c) => emission(i, c));
    let v: number[], f: number[], bp: number[], tm: number[][] = [];
    if (i === 0) {
      v = cand.map((c, k) => Math.log(first[c] ?? 1e-3) + em[k]);
      f = [...v];
      bp = cand.map(() => 0);
    } else {
      const prev = states[i - 1];
      tm = prev.map((p) => cand.map((c) => lt[transitionType(p, c)]));
      v = [];
      f = [];
      bp = [];
      for (let c = 0; c < cand.length; c++) {
        let best = -Infinity, arg = 0, sum = -Infinity;
        for (let p = 0; p < prev.length; p++) {
          const sv = vit[i - 1][p] + tm[p][c];
          if (sv > best) {
            best = sv;
            arg = p;
          }
          sum = logaddexp(sum, fwd[i - 1][p] + tm[p][c]);
        }
        v.push(best + em[c]);
        f.push(sum + em[c]);
        bp.push(arg);
      }
    }
    // Prune to the beam by Viterbi score (stable for ties, as numpy argsort of -v then sorted indices).
    const keep = v.map((x, k) => [x, k] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, BEAM).map((p) => p[1]).sort((a, b) => a - b);
    states.push(keep.map((k) => cand[k]));
    vit.push(keep.map((k) => v[k]));
    fwd.push(keep.map((k) => f[k]));
    back.push(keep.map((k) => bp[k]));
    trans.push(i > 0 ? tm.map((row) => keep.map((k) => row[k])) : []);
  }
  const path = [vit[n - 1].indexOf(Math.max(...vit[n - 1]))];
  for (let i = n - 1; i > 0; i--) path.push(back[i][path[path.length - 1]]);
  path.reverse();
  const bwd = states.map((s) => s.map(() => 0));
  for (let i = n - 2; i >= 0; i--) {
    const em = states[i + 1].map((c) => emission(i + 1, c));
    bwd[i] = states[i].map((_, p) => states[i + 1].reduce((acc, _c, c) => logaddexp(acc, trans[i + 1][p][c] + em[c] + bwd[i + 1][c]), -Infinity));
  }
  const z = fwd[n - 1].reduce((a, b) => logaddexp(a, b), -Infinity);
  return path.map((k, i) => [states[i][k], Math.min(1, Math.exp(fwd[i][k] + bwd[i][k] - z))]);
}
