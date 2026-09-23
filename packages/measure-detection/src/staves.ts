// Staff detection and system grouping on the normalized page at s = 8 (mdet/staves.py).

import { Binary, binarize, close, Gray, open, polyfit, polyval, roundHalfEven } from "./image.js";

export const S = 8;

export interface Staff {
  x0: number;
  x1: number;
  top: number[];
  bottom: number[];
  spacing: number;
  support: number;
}

export interface System {
  staves: Staff[];
  x0: number;
  x1: number;
}

export const yTop = (st: Staff, x: number) => polyval(st.top, x);
export const yBottom = (st: Staff, x: number) => polyval(st.bottom, x);
export const staffXc = (st: Staff) => (st.x0 + st.x1) / 2;
export const systemXc = (sy: System) => (sy.x0 + sy.x1) / 2;
export const systemTop = (sy: System, x: number) => yTop(sy.staves[0], x);
export const systemBottom = (sy: System, x: number) => yBottom(sy.staves[sy.staves.length - 1], x);

function lineCenters(profile: Float64Array, thr: number, maxWidth = 0.35 * S): number[] {
  const out: number[] = [];
  let i = 0;
  const n = profile.length;
  while (i < n) {
    if (profile[i] >= thr) {
      let j = i;
      while (j + 1 < n && profile[j + 1] >= thr) j++;
      if (j + 1 - i <= maxWidth) {
        let num = 0, den = 0;
        for (let k = i; k <= j; k++) {
          num += k * profile[k];
          den += profile[k];
        }
        out.push(num / den);
      }
      i = j + 1;
    } else i++;
  }
  return out;
}

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function fiveLineGroups(ys: number[]): number[][] {
  const groups: number[][] = [];
  let i = 0;
  while (i + 5 <= ys.length) {
    const g = ys.slice(i, i + 5);
    const d = [1, 2, 3, 4].map((k) => g[k] - g[k - 1]);
    const med = median(d);
    if (med >= 0.55 * S && med <= 1.35 * S && Math.max(...d.map((v) => Math.abs(v - med))) <= 0.15 * med + 0.5) {
      groups.push(g);
      i += 5;
    } else i++;
  }
  return groups;
}

export function detectStaves(g: Gray): { staves: Staff[]; bw: Binary } {
  const bw = binarize(g);
  const { width: w, height: h } = bw;
  const lines = open(bw, 2 * S, 1);
  const slab = 12 * S;
  const step = slab >> 1;
  const dets: [number, number[]][] = [];
  for (let x = 0; x < Math.max(1, w - slab + 1); x += step) {
    const prof = new Float64Array(h);
    const x1 = Math.min(w, x + slab);
    for (let y = 0; y < h; y++) {
      let c = 0;
      for (let k = x; k < x1; k++) c += lines.data[y * w + k];
      prof[y] = c / (x1 - x);
    }
    for (const grp of fiveLineGroups(lineCenters(prof, 0.5))) dets.push([x + slab / 2, grp]);
  }
  dets.sort((a, b) => a[0] - b[0]);
  const tracks: [number, number[]][][] = [];
  for (const [xc, grp] of dets) {
    const sp = (grp[4] - grp[0]) / 4;
    let best: [number, number[]][] | null = null;
    for (const t of tracks) {
      const [lx, lg] = t[t.length - 1];
      const lsp = (lg[4] - lg[0]) / 4;
      if (xc - lx > 0 && xc - lx <= 2.5 * step && Math.abs(lg[0] - grp[0]) < S && Math.abs(lsp - sp) < 0.2 * sp) {
        if (best === null || Math.abs(lg[0] - grp[0]) < Math.abs(best[best.length - 1][1][0] - grp[0])) best = t;
      }
    }
    if (best === null) tracks.push([[xc, grp]]);
    else best.push([xc, grp]);
  }
  const staves: Staff[] = [];
  for (const t of tracks) {
    const xs = t.map((d) => d[0]);
    const tops = t.map((d) => d[1][0]);
    const bots = t.map((d) => d[1][4]);
    const sp = median(t.map((d) => (d[1][4] - d[1][0]) / 4));
    const deg = t.length < 3 ? 0 : t.length < 6 ? 1 : 2;
    const ptop = polyfit(xs, tops, deg), pbot = polyfit(xs, bots, deg);
    const present = new Uint8Array(w);
    for (let x = 0; x < w; x++) {
      const yt = polyval(ptop, x), yb = polyval(pbot, x);
      let cnt = 0;
      for (let k = 0; k < 5; k++) {
        const yi = Math.min(h - 2, Math.max(1, roundHalfEven(yt + (k * (yb - yt)) / 4)));
        const o = yi * w + x;
        cnt += lines.data[o] | lines.data[o - w] | lines.data[o + w];
      }
      present[x] = cnt >= 4 ? 1 : 0;
    }
    const closed = close({ width: w, height: 1, data: present }, 3 * S + 1, 1).data;
    let anchor = Math.floor(median(xs));
    if (!closed[anchor]) {
      let bestD = Infinity, found = -1;
      for (let x = 0; x < w; x++) if (closed[x] && Math.abs(x - anchor) < bestD) {
        bestD = Math.abs(x - anchor);
        found = x;
      }
      if (found < 0) continue;
      anchor = found;
    }
    let x0 = anchor;
    while (x0 > 0 && closed[x0 - 1]) x0--;
    let x1 = anchor;
    while (x1 < w - 1 && closed[x1 + 1]) x1++;
    if (x1 - x0 < 8 * S) continue;
    staves.push({ x0, x1: x1 + 1, top: ptop, bottom: pbot, spacing: sp, support: t.length });
  }
  staves.sort((a, b) => yTop(a, staffXc(a)) - yTop(b, staffXc(b)));
  return { staves: dedupe(staves), bw };
}

function dedupe(staves: Staff[]): Staff[] {
  const out: Staff[] = [];
  // Stable sort by support, descending (Python sorted(key=-support)).
  const order = staves.map((s, i) => [s, i] as const).sort((a, b) => b[0].support - a[0].support || a[1] - b[1]).map((p) => p[0]);
  for (const st of order) {
    let keep = true;
    for (const o of out) {
      const xa = Math.max(o.x0, st.x0), xb = Math.min(o.x1, st.x1);
      const xm = xb > xa ? (xa + xb) / 2 : staffXc(o);
      const ot = yTop(o, xm), ob = yBottom(o, xm), tt = yTop(st, xm), tb = yBottom(st, xm);
      if (Math.min(ob, tb) - Math.max(ot, tt) > 0.3 * (tb - tt)) {
        if (Math.abs(ot - tt) < S) {
          o.x0 = Math.min(o.x0, st.x0);
          o.x1 = Math.max(o.x1, st.x1);
        }
        keep = false;
        break;
      }
    }
    if (keep) out.push(st);
  }
  out.sort((a, b) => yTop(a, staffXc(a)) - yTop(b, staffXc(b)));
  return out;
}

function connectedAtLeft(a: Staff, b: Staff, bw: Binary, minCover = 0.8): boolean {
  const { width: w, data } = bw;
  let best = 0;
  const from = Math.trunc(Math.min(a.x0, b.x0) - 2 * S), to = Math.trunc(Math.max(a.x0, b.x0) + 1.5 * S);
  for (let x = from; x <= to; x++) {
    if (x < 0 || x >= w) continue;
    const y0 = roundHalfEven(yBottom(a, x)), y1 = roundHalfEven(yTop(b, x));
    if (y1 - y0 < 2) return true;
    let c = 0;
    for (let y = y0; y < y1; y++) {
      let v = 0;
      for (let k = Math.max(0, x - 1); k <= Math.min(w - 1, x + 1); k++) v |= data[y * w + k];
      c += v;
    }
    best = Math.max(best, c / (y1 - y0));
  }
  return best >= minCover;
}

export function groupSystems(staves: Staff[], bw: Binary): System[] {
  const groups: Staff[][] = [];
  for (const st of staves) {
    if (groups.length) {
      const g = groups[groups.length - 1];
      const prev = g[g.length - 1];
      const overlap = Math.min(prev.x1, st.x1) - Math.max(prev.x0, st.x0);
      if (overlap > 0.5 * Math.min(prev.x1 - prev.x0, st.x1 - st.x0) && connectedAtLeft(prev, st, bw)) {
        g.push(st);
        continue;
      }
    }
    groups.push([st]);
  }
  return groups.map((g) => ({ staves: g, x0: Math.min(...g.map((s) => s.x0)), x1: Math.max(...g.map((s) => s.x1)) }));
}

export function detectSystems(g: Gray): { systems: System[]; bw: Binary } {
  const { staves, bw } = detectStaves(g);
  return { systems: groupSystems(staves, bw), bw };
}
