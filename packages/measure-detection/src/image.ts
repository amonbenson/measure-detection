// Image primitives. Each function reproduces the OpenCV / NumPy operation used by the Python
// training pipeline (src/mdet), so that browser results match the evaluated pipeline.

export interface Gray {
  width: number;
  height: number;
  data: Uint8Array; // 0 = black, 255 = paper
}

/** Binary image: 1 = ink. */
export interface Binary {
  width: number;
  height: number;
  data: Uint8Array;
}

export interface FloatImage {
  width: number;
  height: number;
  data: Float32Array;
}

export type Affine = [number, number, number, number, number, number]; // row-major 2x3

export function gray(width: number, height: number, fill = 255): Gray {
  return { width, height, data: new Uint8Array(width * height).fill(fill) };
}

/** Round half to even (Python round, NumPy round, OpenCV cvRound). */
export function roundHalfEven(x: number): number {
  const r = Math.round(x);
  if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2);
  return r;
}

/** Otsu threshold (cv2.THRESH_OTSU), then THRESH_BINARY_INV: ink = value <= threshold. */
export function otsuThreshold(g: Gray): number {
  const hist = new Float64Array(256);
  for (let i = 0; i < g.data.length; i++) hist[g.data[i]]++;
  const scale = 1 / g.data.length;
  let mu = 0;
  for (let i = 0; i < 256; i++) mu += i * hist[i] * scale;
  let q1 = 0, mu1 = 0, maxSigma = 0, maxVal = 0;
  const eps = 1.1920928955078125e-7;
  for (let i = 0; i < 256; i++) {
    const p = hist[i] * scale;
    mu1 *= q1;
    q1 += p;
    const q2 = 1 - q1;
    if (Math.min(q1, q2) < eps || Math.max(q1, q2) > 1 - eps) continue;
    mu1 = (mu1 + i * p) / q1;
    const mu2 = (mu - q1 * mu1) / q2;
    const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
    if (sigma > maxSigma) {
      maxSigma = sigma;
      maxVal = i;
    }
  }
  return maxVal;
}

export function binarize(g: Gray): Binary {
  const t = otsuThreshold(g);
  const out = new Uint8Array(g.data.length);
  for (let i = 0; i < out.length; i++) out[i] = g.data[i] > t ? 0 : 1;
  return { width: g.width, height: g.height, data: out };
}

/** Sliding min/max along rows (horizontal) with window [x - anchor, x - anchor + k - 1]; outside pixels are ignored. */
function slide(src: Uint8Array, w: number, h: number, k: number, horizontal: boolean, isMin: boolean): Uint8Array {
  const out = new Uint8Array(src.length);
  const anchor = Math.floor(k / 2);
  const n = horizontal ? w : h;
  const lines = horizontal ? h : w;
  const stride = horizontal ? 1 : w;
  const lineStride = horizontal ? w : 1;
  for (let l = 0; l < lines; l++) {
    const base = l * lineStride;
    // Binary data: count ink in the window.
    let count = 0;
    let lo = -anchor, hi = lo + k - 1; // window for position 0
    for (let j = Math.max(0, lo); j <= Math.min(n - 1, hi); j++) count += src[base + j * stride];
    for (let x = 0; x < n; x++) {
      const a = Math.max(0, x - anchor), b = Math.min(n - 1, x - anchor + k - 1);
      const size = b - a + 1;
      out[base + x * stride] = isMin ? (count === size ? 1 : 0) : count > 0 ? 1 : 0;
      // advance window
      const drop = x - anchor, add = x - anchor + k;
      if (drop >= 0 && drop < n) count -= src[base + drop * stride];
      if (add >= 0 && add < n) count += src[base + add * stride];
    }
  }
  return out;
}

export function erode(b: Binary, kw: number, kh: number): Binary {
  let d = b.data;
  if (kw > 1) d = slide(d, b.width, b.height, kw, true, true);
  if (kh > 1) d = slide(d, b.width, b.height, kh, false, true);
  return { width: b.width, height: b.height, data: d };
}

export function dilate(b: Binary, kw: number, kh: number): Binary {
  let d = b.data;
  if (kw > 1) d = slide(d, b.width, b.height, kw, true, false);
  if (kh > 1) d = slide(d, b.width, b.height, kh, false, false);
  return { width: b.width, height: b.height, data: d };
}

/** cv2.morphologyEx(MORPH_OPEN) with a kw x kh rectangle. */
export function open(b: Binary, kw: number, kh: number): Binary {
  return dilate(erode(b, kw, kh), kw, kh);
}

/** cv2.morphologyEx(MORPH_CLOSE) with a kw x kh rectangle. */
export function close(b: Binary, kw: number, kh: number): Binary {
  return erode(dilate(b, kw, kh), kw, kh);
}

/** Area-coverage weights (dst x src, sparse) as in mdet.normalize._area_weights. */
function areaWeights(src: number, dst: number): { start: number[]; w: number[][] } {
  const scale = src / dst;
  const start: number[] = [];
  const w: number[][] = [];
  for (let d = 0; d < dst; d++) {
    const a = d * scale, b = (d + 1) * scale;
    const i0 = Math.floor(a), i1 = Math.min(src, Math.ceil(b));
    const row: number[] = [];
    let sum = 0;
    for (let i = i0; i < i1; i++) {
      const v = Math.min(b, i + 1) - Math.max(a, i);
      row.push(v);
      sum += v;
    }
    start.push(i0);
    w.push(row.map((v) => v / sum));
  }
  return { start, w };
}

/** Exact area-coverage resize (mdet.normalize.resize_area). */
export function resizeArea(g: Gray, width: number, height: number): Gray {
  const wy = areaWeights(g.height, height);
  const wx = areaWeights(g.width, width);
  // Rows first (y), then columns (x), in float64 like the NumPy matrix product.
  const tmp = new Float64Array(height * g.width);
  for (let y = 0; y < height; y++) {
    const ws = wy.w[y], s0 = wy.start[y];
    for (let k = 0; k < ws.length; k++) {
      const src = (s0 + k) * g.width, wk = ws[k];
      for (let x = 0; x < g.width; x++) tmp[y * g.width + x] += wk * g.data[src + x];
    }
  }
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const ws = wx.w[x], s0 = wx.start[x];
      let v = 0;
      for (let k = 0; k < ws.length; k++) v += ws[k] * tmp[y * g.width + s0 + k];
      out[y * width + x] = Math.min(255, Math.max(0, Math.floor(v + 0.5)));
    }
  }
  return { width, height, data: out };
}

/** 2x2 block average with rounding, odd sizes padded with paper (mdet.geometry.halve). */
export function halve(g: Gray): Gray {
  const w = (g.width + 1) >> 1, h = (g.height + 1) >> 1;
  const out = new Uint8Array(w * h);
  const at = (x: number, y: number) => (x < g.width && y < g.height ? g.data[y * g.width + x] : 255);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) out[y * w + x] = (at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1) + 2) >> 2;
  return { width: w, height: h, data: out };
}

export function rotationMatrix(cx: number, cy: number, angleDeg: number, scale: number): Affine {
  const a = (angleDeg * Math.PI) / 180;
  const alpha = scale * Math.cos(a), beta = scale * Math.sin(a);
  return [alpha, beta, (1 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1 - alpha) * cy];
}

export function invertAffine(m: Affine): Affine {
  const [a, b, c, d, e, f] = m;
  let det = a * e - b * d;
  det = det !== 0 ? 1 / det : 0;
  const A11 = e * det, A22 = a * det, A12 = -b * det, A21 = -d * det;
  return [A11, A12, -A11 * c - A12 * f, A21, A22, -A21 * c - A22 * f];
}

// OpenCV 5 samples bilinearly at float coordinates (no 1/32 quantization): source coordinates from
// the inverse matrix, float32 interpolation, round half to even for 8-bit output, constant border.
const f32 = Math.fround;

function bilinear(get: (x: number, y: number) => number, w: number, h: number, x: number, y: number, border: number): number | null {
  const X = f32(x), Y = f32(y);
  const sx = Math.floor(X), sy = Math.floor(Y);
  if (sx >= w || sx < -1 || sy >= h || sy < -1) return null;
  const fx = f32(X - sx), fy = f32(Y - sy);
  const px = (xx: number, yy: number) => (xx >= 0 && xx < w && yy >= 0 && yy < h ? get(xx, yy) : border);
  const v00 = px(sx, sy), v01 = px(sx + 1, sy), v10 = px(sx, sy + 1), v11 = px(sx + 1, sy + 1);
  const a = f32(v00 + f32(f32(v01 - v00) * fx));
  const b = f32(v10 + f32(f32(v11 - v10) * fx));
  return f32(a + f32(f32(b - a) * fy));
}

function sample8(g: Gray, x: number, y: number, border: number): number {
  const v = bilinear((xx, yy) => g.data[yy * g.width + xx], g.width, g.height, x, y, border);
  return v === null ? border : Math.min(255, Math.max(0, roundHalfEven(v)));
}

/** cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT) of an 8-bit image. m maps source to destination. */
export function warpAffine(g: Gray, m: Affine, width: number, height: number, border = 255): Gray {
  const inv = invertAffine(m);
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) out[y * width + x] = sample8(g, inv[0] * x + inv[1] * y + inv[2], inv[3] * x + inv[4] * y + inv[5], border);
  return { width, height, data: out };
}

/** cv2.warpAffine of a float image with border 0. If inverse is true, m already maps destination to source.
 *  Inlined version of sampleF (same float32 operation order) because skew and shear estimation call it often. */
export function warpAffineF(g: FloatImage, m: Affine, width: number, height: number, inverse = false): FloatImage {
  const inv = inverse ? m : invertAffine(m);
  const out = new Float32Array(width * height);
  const w = g.width, h = g.height, d = g.data;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const X = f32(inv[0] * x + inv[1] * y + inv[2]), Y = f32(inv[3] * x + inv[4] * y + inv[5]);
      const sx = Math.floor(X), sy = Math.floor(Y);
      if (sx >= w || sx < -1 || sy >= h || sy < -1) continue;
      const fx = f32(X - sx), fy = f32(Y - sy);
      const inX0 = sx >= 0, inX1 = sx + 1 < w, inY0 = sy >= 0, inY1 = sy + 1 < h;
      const r0 = sy * w, r1 = r0 + w;
      const v00 = inX0 && inY0 ? d[r0 + sx] : 0, v01 = inX1 && inY0 ? d[r0 + sx + 1] : 0;
      const v10 = inX0 && inY1 ? d[r1 + sx] : 0, v11 = inX1 && inY1 ? d[r1 + sx + 1] : 0;
      const a = f32(v00 + f32(f32(v01 - v00) * fx));
      const b = f32(v10 + f32(f32(v11 - v10) * fx));
      out[y * width + x] = f32(a + f32(f32(b - a) * fy));
    }
  return { width, height, data: out };
}

/** cv2.remap(INTER_LINEAR, BORDER_CONSTANT) with float32 maps given as a function of (x, y). */
export function remap(g: Gray, width: number, height: number, map: (x: number, y: number) => [number, number], border = 255): Gray {
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const [sx, sy] = map(x, y);
      out[y * width + x] = sample8(g, sx, sy, border);
    }
  return { width, height, data: out };
}

export interface Component {
  x: number;
  y: number;
  w: number;
  h: number;
  area: number;
}

/** 8-connected components with bounding boxes (cv2.connectedComponentsWithStats, background excluded). */
export function components(b: Binary): Component[] {
  const { width: w, height: h, data } = b;
  const parent = new Int32Array(w * h).fill(-1);
  const find = (i: number): number => {
    while (parent[i] !== i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const union = (a: number, c: number) => {
    const ra = find(a), rc = find(c);
    if (ra !== rc) parent[Math.max(ra, rc)] = Math.min(ra, rc);
  };
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!data[i]) continue;
      parent[i] = i;
      if (x > 0 && data[i - 1]) union(i, i - 1);
      if (y > 0) {
        if (data[i - w]) union(i, i - w);
        if (x > 0 && data[i - w - 1]) union(i, i - w - 1);
        if (x < w - 1 && data[i - w + 1]) union(i, i - w + 1);
      }
    }
  const stats = new Map<number, Component>();
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!data[i]) continue;
      const r = find(i);
      const s = stats.get(r);
      if (!s) stats.set(r, { x, y, w: 1, h: 1, area: 1 });
      else {
        const x1 = Math.max(s.x + s.w, x + 1), y1 = Math.max(s.y + s.h, y + 1);
        s.x = Math.min(s.x, x);
        s.y = Math.min(s.y, y);
        s.w = x1 - s.x;
        s.h = y1 - s.y;
        s.area++;
      }
    }
  return [...stats.values()];
}

/** Least-squares polynomial fit (np.polyfit), coefficients highest power first. */
export function polyfit(xs: number[], ys: number[], deg: number): number[] {
  const n = deg + 1;
  // Scale x for conditioning (as numpy does implicitly via lstsq on scaled columns).
  const xm = xs.reduce((a, b) => a + b, 0) / xs.length;
  const xsd = Math.max(1e-9, Math.sqrt(xs.reduce((a, b) => a + (b - xm) ** 2, 0) / xs.length)) || 1;
  const t = xs.map((x) => (x - xm) / xsd);
  const ata = Array.from({ length: n }, () => new Array(n).fill(0));
  const aty = new Array(n).fill(0);
  for (let i = 0; i < t.length; i++) {
    const pw = Array.from({ length: n }, (_, k) => t[i] ** (deg - k));
    for (let r = 0; r < n; r++) {
      aty[r] += pw[r] * ys[i];
      for (let c = 0; c < n; c++) ata[r][c] += pw[r] * pw[c];
    }
  }
  const cT = solve(ata, aty); // coefficients in t
  // Expand p(t) with t = (x - xm) / xsd into powers of x.
  let poly = [0];
  for (let k = 0; k < n; k++) {
    // poly = poly * t + cT[k], with t = (x - xm)/xsd as polynomial [1/xsd, -xm/xsd]
    const next = new Array(poly.length + 1).fill(0);
    for (let i = 0; i < poly.length; i++) {
      next[i] += poly[i] / xsd;
      next[i + 1] += (-poly[i] * xm) / xsd;
    }
    next[next.length - 1] += cT[k];
    poly = next;
  }
  return poly.slice(poly.length - n);
}

function solve(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    [m[c], m[p]] = [m[p], m[c]];
    const d = m[c][c] || 1e-300;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = m[r][c] / d;
      for (let k = c; k <= n; k++) m[r][k] -= f * m[c][k];
    }
  }
  return m.map((row, i) => row[n] / (row[i] || 1e-300));
}

export function polyval(p: number[], x: number): number {
  let v = 0;
  for (const c of p) v = v * x + c;
  return v;
}
