// CPU operations of the page pipeline, shared by the Web Worker (worker.ts) and the in-thread fallback.
// A page goes through "prepare" (normalize, systems, MeasureNet inputs) and "finish" (boundaries,
// measures, text candidates, OCR inputs); its images stay here between the two and are freed after "finish".

import { StripInput, stripInput } from "./input.js";
import { estimateStaffSpace } from "./normalize.js";
import { makeStrip, rectify, Strip } from "./rectify.js";
import { FinishedSystem, finishSystem, MIN_MODE_STRENGTH, normalizePage, PreparedPage, Signals, toGray } from "./stages.js";
import { detectSystems } from "./staves.js";

export interface PrepareResult {
  hasStaff: boolean;
  systems: { input: StripInput; width: number }[]; // MeasureNet input and rectified strip width per system
}

export type OpState = Map<string, { prep: PreparedPage; strips: Strip[] }>;
export type OpResult = { result: unknown; transfer: Transferable[] };

const EMPTY = { width: 0, height: 0, data: new Uint8Array(0) };

export function runOp(state: OpState, op: string, a: any): OpResult {
  switch (op) {
    case "ping":
      return { result: "pong", transfer: [] };
    case "staffSpace": {
      const est = estimateStaffSpace(toGray(a.data, a.width, a.height));
      return { result: { s: est.s, modeStrength: est.modeStrength }, transfer: [] };
    }
    case "prepare": {
      const g = toGray(a.data, a.width, a.height);
      const est = estimateStaffSpace(g);
      // Without a staff the page has no systems; its normalization is not needed.
      if (est.modeStrength < MIN_MODE_STRENGTH) return { result: { hasStaff: false, systems: [] } satisfies PrepareResult, transfer: [] };
      const prep = normalizePage(g, a.page, est, a.s1, NaN);
      const strips: Strip[] = [];
      const systems: PrepareResult["systems"] = [];
      for (const sy of detectSystems(prep.norm8).systems) {
        const strip = makeStrip(sy, prep.norm8.width, prep.norm8);
        const img = rectify(prep.norm8, strip);
        strips.push(strip);
        systems.push({ input: stripInput(img, strip), width: img.width });
      }
      // "finish" needs norm16 (text) and the page geometry, not norm8.
      state.set(a.key, { prep: { ...prep, norm8: EMPTY }, strips });
      return { result: { hasStaff: true, systems } satisfies PrepareResult, transfer: systems.map((s) => s.input.data.buffer) };
    }
    case "finish": {
      const st = state.get(a.key);
      if (!st) throw new Error(`finish: unknown page ${a.key}`);
      state.delete(a.key);
      const systems: FinishedSystem[] = st.strips.map((strip, i) => finishSystem(st.prep, strip, a.signals[i] as Signals, a.tauB));
      return { result: systems, transfer: systems.flatMap((s) => s.ocr.map((x) => x.buffer)) };
    }
    case "drop":
      for (const k of [...state.keys()]) if (k.startsWith(a.prefix)) state.delete(k);
      return { result: null, transfer: [] };
    default:
      throw new Error(`unknown operation ${op}`);
  }
}
