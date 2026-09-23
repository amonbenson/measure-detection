// LabelOCR decoding: CTC prefix beam search (mdet/ocr.py).

export const CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-";
const BLANK = 0;

/** logp: row-major (T, C) log-probabilities. Returns up to `top` [string, probability]. */
export function beamSearch(logp: Float32Array, T: number, C: number, beam = 8, top = 5): [string, number][] {
  let beams = new Map<string, [number[], number, number]>([["", [[], 1, 0]]]);
  for (let t = 0; t < T; t++) {
    const p = new Float64Array(C);
    for (let c = 0; c < C; c++) p[c] = Math.exp(logp[t * C + c]);
    const cand = Array.from({ length: C }, (_, i) => i).sort((a, b) => p[b] - p[a] || a - b).slice(0, 6);
    const next = new Map<string, [number[], number, number]>();
    const get = (pre: number[]) => {
      const key = pre.join(",");
      let e = next.get(key);
      if (!e) {
        e = [pre, 0, 0];
        next.set(key, e);
      }
      return e;
    };
    for (const [, [prefix, pb, pnb]] of beams) {
      for (const k of cand) {
        const pk = p[k];
        if (k === BLANK) {
          get(prefix)[1] += (pb + pnb) * pk;
          continue;
        }
        const last = prefix.length ? prefix[prefix.length - 1] : null;
        const e = get([...prefix, k]);
        if (k === last) {
          e[2] += pb * pk;
          get(prefix)[2] += pnb * pk;
        } else e[2] += (pb + pnb) * pk;
      }
    }
    beams = new Map([...next.entries()].sort((a, b) => b[1][1] + b[1][2] - (a[1][1] + a[1][2])).slice(0, beam));
  }
  const res = [...beams.values()].map(([pre, pb, pnb]) => [pre.map((k) => CHARSET[k - 1]).join(""), pb + pnb] as [string, number]);
  const z = res.reduce((a, r) => a + r[1], 0) || 1;
  return res.map(([s, q]) => [s, q / z] as [string, number]).sort((a, b) => b[1] - a[1]).slice(0, top);
}
