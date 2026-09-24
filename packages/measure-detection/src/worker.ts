// Web Worker entry: runs CPU operations of the page pipeline (ops.ts) off the main thread.
// Imports only the package's own modules, so it works unbundled and with bundlers.

import { OpState, runOp } from "./ops.js";

const state: OpState = new Map();

self.onmessage = (ev: MessageEvent) => {
  const { id, op, args } = ev.data;
  try {
    const { result, transfer } = runOp(state, op, args);
    (self as unknown as Worker).postMessage({ id, result }, transfer);
  } catch (err) {
    (self as unknown as Worker).postMessage({ id, error: err instanceof Error ? err.stack || err.message : String(err) });
  }
};
