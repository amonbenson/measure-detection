// Runs the CPU operations of ops.ts in a pool of Web Workers, or on the calling thread when workers are
// unavailable (Node, old browsers, a Content Security Policy that forbids workers).

import { OpState, runOp } from "./ops.js";

/** A worker stopped (crashed, killed for memory, or failed to load). The page can be retried elsewhere. */
export class WorkerFailure extends Error {
  name = "WorkerFailure";
}

export interface Executor {
  readonly kind: "workers" | "inline";
  /** Number of pages that can be processed in parallel. */
  readonly size: number;
  /** Operations with the same affinity key run on the same worker (a page's "prepare" and "finish"). */
  call<T>(op: string, args: unknown, transfer?: Transferable[], affinity?: string): Promise<T>;
  release(affinity: string): void;
  close(): void;
}

const nextTask = () => new Promise<void>((r) => setTimeout(r, 0));

/** Same operations on the calling thread; yields to the event loop before each one so a page can repaint. */
export class InlineExecutor implements Executor {
  readonly kind = "inline";
  readonly size = 1;
  private state: OpState = new Map();

  async call<T>(op: string, args: unknown): Promise<T> {
    await nextTask();
    return runOp(this.state, op, args).result as T;
  }
  release(): void {}
  close(): void {
    this.state.clear();
  }
}

interface Slot {
  worker: Worker;
  pending: Map<number, { resolve: (v: any) => void; reject: (e: Error) => void }>;
  dead: boolean;
}

class WorkerPool implements Executor {
  readonly kind = "workers";
  private nextId = 1;
  private sticky = new Map<string, Slot>();

  constructor(private slots: Slot[]) {
    for (const slot of slots) {
      slot.worker.onmessage = (ev: MessageEvent) => {
        const { id, result, error } = ev.data;
        const p = slot.pending.get(id);
        if (!p) return;
        slot.pending.delete(id);
        if (error) p.reject(new Error(error));
        else p.resolve(result);
      };
      slot.worker.onerror = (ev: ErrorEvent) => {
        ev.preventDefault();
        this.fail(slot, ev.message || "worker error");
      };
      slot.worker.onmessageerror = () => this.fail(slot, "message could not be deserialized");
    }
  }

  get size(): number {
    return this.slots.filter((s) => !s.dead).length;
  }

  fail(slot: Slot, reason: string): void {
    if (slot.dead) return;
    slot.dead = true;
    for (const p of slot.pending.values()) p.reject(new WorkerFailure(`measure-detection worker stopped: ${reason}`));
    slot.pending.clear();
    slot.worker.terminate();
  }

  send<T>(slot: Slot, op: string, args: unknown, transfer: Transferable[] = []): Promise<T> {
    if (slot.dead) return Promise.reject(new WorkerFailure("measure-detection worker stopped"));
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      slot.pending.set(id, { resolve, reject });
      try {
        slot.worker.postMessage({ id, op, args }, transfer);
      } catch (err) {
        slot.pending.delete(id);
        reject(err instanceof Error ? err : new Error(String(err)));
      }
    });
  }

  call<T>(op: string, args: unknown, transfer: Transferable[] = [], affinity?: string): Promise<T> {
    let slot = affinity ? this.sticky.get(affinity) : undefined;
    if (!slot) {
      const live = this.slots.filter((s) => !s.dead);
      if (!live.length) return Promise.reject(new WorkerFailure("no measure-detection worker is running"));
      slot = live.reduce((a, b) => (b.pending.size < a.pending.size ? b : a));
      if (affinity) this.sticky.set(affinity, slot);
    }
    return this.send<T>(slot, op, args, transfer);
  }

  release(affinity: string): void {
    this.sticky.delete(affinity);
  }

  close(): void {
    for (const slot of this.slots) this.fail(slot, "closed");
  }
}

/** Default pool size: one core stays free for the page and pdf.js; devices with little memory get fewer workers.
 *  navigator.deviceMemory exists only in Chromium browsers (rounded, in GB). */
export function defaultWorkers(): number {
  const nav: any = typeof navigator !== "undefined" ? navigator : {};
  const cores = typeof nav.hardwareConcurrency === "number" && nav.hardwareConcurrency > 0 ? nav.hardwareConcurrency : 2;
  let n = Math.min(4, Math.max(1, cores - 1));
  if (typeof nav.deviceMemory === "number") n = Math.min(n, nav.deviceMemory <= 2 ? 1 : nav.deviceMemory <= 4 ? 2 : 4);
  return n;
}

// Loading a worker module over a slow connection on a slow device can take a while.
const START_TIMEOUT_MS = 20000;

// Browser page or browser worker. Node can have a global Worker from polyfills, but not these.
const inBrowser = () => typeof Worker !== "undefined" && (typeof document !== "undefined" || typeof (globalThis as any).WorkerGlobalScope !== "undefined");

export async function createExecutor(workers: number | undefined, factory?: () => Worker): Promise<Executor> {
  const n = Math.floor(workers ?? defaultWorkers());
  if (n <= 0 || (!factory && !inBrowser())) return new InlineExecutor();
  // The literal `new Worker(new URL(...), { type: "module" })` form lets bundlers (Vite, webpack) emit the worker.
  const make = factory ?? (() => new Worker(new URL("./worker.js", import.meta.url), { type: "module" }));
  const slots: Slot[] = [];
  for (let i = 0; i < n; i++) {
    try {
      slots.push({ worker: make(), pending: new Map(), dead: false });
    } catch {
      break; // module workers unsupported or forbidden
    }
  }
  if (!slots.length) return new InlineExecutor();
  const pool = new WorkerPool(slots);
  await Promise.all(
    slots.map((slot) => {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const timeout = new Promise<never>((_, reject) => (timer = setTimeout(() => reject(new Error("timeout")), START_TIMEOUT_MS)));
      return Promise.race([pool.send(slot, "ping", {}), timeout])
        .catch((err) => pool.fail(slot, `did not start (${err instanceof Error ? err.message : err})`))
        .finally(() => clearTimeout(timer));
    }),
  );
  if (pool.size === 0) {
    pool.close();
    return new InlineExecutor();
  }
  return pool;
}
