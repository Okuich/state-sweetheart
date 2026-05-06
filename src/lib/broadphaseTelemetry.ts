/**
 * broadphaseTelemetry.ts
 * ──────────────────────────────────────────────────────────────────────────
 * GPU timing + counter harness for comparing two broad-phase strategies —
 * a uniform spatial hash and an LBVH (Linear BVH built from Morton codes).
 * The harness measures four phases per strategy:
 *
 *      build_struct   →   build_query   →   pair_emit   →   readback
 *
 * For each phase it records:
 *   • wall-clock elapsed (performance.now)
 *   • GPU-side elapsed via a `timestamp-query` set when the
 *     `"timestamp-query"` feature is available; otherwise it reuses the
 *     wall clock and flags `gpuTimingAvailable=false`.
 *   • per-strategy COUNTERS sourced from the strategy's own `Counters`
 *     callback: candidate pairs visited, pairs emitted, pairs deduped,
 *     pairs dropped due to saturation. These come from the strategy
 *     because only it knows what "candidate" means (cell visits for
 *     spatial hash, node-tests for LBVH).
 *
 * The harness then derives a unified rate panel:
 *   pairsEmittedPerMs    = emitted / wall_ms
 *   candidatesPerMs      = candidates / wall_ms
 *   emissionRatio        = emitted / candidates           (selectivity)
 *   dedupRatio           = unique / emitted               (1.0 = no dups)
 *   saturationRatio      = dropped / (emitted + dropped)  (0.0 = healthy)
 *
 * A built-in `RingTelemetry` keeps the last `windowSize` samples per
 * strategy so callers can plot smoothed curves without re-allocating.
 *
 * Sandbox note
 * ------------
 * The Lovable preview browser typically lacks both WebGPU and the
 * `timestamp-query` feature. The harness still produces wall-clock
 * timings so unit tests and overlays work end-to-end; consult
 * `gpuTimingAvailable` to decide whether to display GPU columns.
 */

// ── Public types ─────────────────────────────────────────────────────────

export type BroadphaseStrategy = "spatial-hash" | "lbvh";

export type Phase = "build_struct" | "build_query" | "pair_emit" | "readback";

/** Per-phase, per-call timing record. */
export interface PhaseTiming {
  phase: Phase;
  /** Wall-clock duration in milliseconds (always present). */
  wallMs: number;
  /** GPU duration in milliseconds (NaN when no timestamp-query). */
  gpuMs: number;
}

/** Strategy-supplied counters for one tick. */
export interface BroadphaseCounters {
  /** Number of bodies in the scene this tick. */
  N: number;
  /** Total candidate inspections (cell visits / node-tests / etc). */
  candidates: number;
  /** Pairs the strategy emitted before dedup. */
  emitted: number;
  /** Unique pairs after dedup (or === emitted if dedup not applied). */
  unique: number;
  /** Pairs dropped because maxPairs was reached. */
  dropped: number;
}

/** Derived rates. All times are in milliseconds. */
export interface DerivedRates {
  totalWallMs: number;
  totalGpuMs: number;
  pairsEmittedPerMs: number;
  candidatesPerMs: number;
  /** emitted / candidates — closer to 1 means the strategy is precise. */
  emissionRatio: number;
  /** unique / emitted — 1.0 means no duplicates were generated. */
  dedupRatio: number;
  /** dropped / (emitted + dropped) — 0.0 means no saturation pressure. */
  saturationRatio: number;
}

/** A single strategy×tick sample assembled by `recordSample`. */
export interface StrategySample {
  strategy: BroadphaseStrategy;
  /** Monotonically-increasing tick id supplied by the caller. */
  tick: number;
  phases: PhaseTiming[];
  counters: BroadphaseCounters;
  rates: DerivedRates;
  /** Whether GPU timestamps were measured (else gpuMs is wall-mirrored NaN). */
  gpuTimingAvailable: boolean;
}

// ── Phase timer ──────────────────────────────────────────────────────────

/**
 * Bracket a phase to time it. Always returns wall-clock elapsed and a
 * stable `phase` tag. If a `GPUTimestampPair` is supplied it also reads
 * back the GPU-side elapsed nanoseconds; otherwise gpuMs is NaN.
 *
 * Usage:
 *   const t = startPhase("build_struct");
 *   // ... encode + submit ...
 *   const timing = await endPhase(t, { device, beginIdx, endIdx, querySet, resolveBuf });
 */
export interface PhaseHandle {
  phase: Phase;
  startWall: number;
}

export function startPhase(phase: Phase): PhaseHandle {
  return { phase, startWall: nowMs() };
}

export interface GPUTimestampReadback {
  /** Elapsed nanoseconds between the phase's begin/end timestamps. */
  elapsedNs: number;
}

export interface EndPhaseOptions {
  /** If provided, contributes to `gpuMs`. Otherwise gpuMs is NaN. */
  gpu?: GPUTimestampReadback;
}

export function endPhase(handle: PhaseHandle, opts: EndPhaseOptions = {}): PhaseTiming {
  const wallMs = nowMs() - handle.startWall;
  const gpuMs = opts.gpu ? opts.gpu.elapsedNs / 1_000_000 : Number.NaN;
  return { phase: handle.phase, wallMs, gpuMs };
}

// ── Sample assembly ──────────────────────────────────────────────────────

/**
 * Turn a list of phase timings + counters into a fully-derived sample.
 * The derivations are pure: no IO, no allocations beyond the result.
 */
export function recordSample(
  strategy: BroadphaseStrategy,
  tick: number,
  phases: PhaseTiming[],
  counters: BroadphaseCounters,
): StrategySample {
  let totalWallMs = 0;
  let totalGpuMs = 0;
  let anyGpu = false;
  for (const p of phases) {
    totalWallMs += p.wallMs;
    if (Number.isFinite(p.gpuMs)) {
      totalGpuMs += p.gpuMs;
      anyGpu = true;
    }
  }
  if (!anyGpu) totalGpuMs = Number.NaN;

  const denom = (x: number, y: number, fallback = 0) =>
    y > 0 ? x / y : fallback;

  const rates: DerivedRates = {
    totalWallMs,
    totalGpuMs,
    pairsEmittedPerMs: denom(counters.emitted, totalWallMs),
    candidatesPerMs:   denom(counters.candidates, totalWallMs),
    emissionRatio:     denom(counters.emitted, counters.candidates),
    dedupRatio:        denom(counters.unique, counters.emitted, 1),
    saturationRatio:   denom(counters.dropped, counters.emitted + counters.dropped),
  };

  return { strategy, tick, phases, counters, rates, gpuTimingAvailable: anyGpu };
}

// ── Ring buffer for plotting / overlays ──────────────────────────────────

export interface RingTelemetryOptions {
  windowSize?: number; // default 240 (~4s @ 60fps)
}

/**
 * Fixed-capacity per-strategy sample buffer with O(1) push and an
 * `aggregate()` summary used by HUD overlays.
 */
export class RingTelemetry {
  readonly windowSize: number;
  private bufs: Record<BroadphaseStrategy, StrategySample[]>;
  private heads: Record<BroadphaseStrategy, number>;
  private counts: Record<BroadphaseStrategy, number>;

  constructor(opts: RingTelemetryOptions = {}) {
    this.windowSize = Math.max(1, opts.windowSize ?? 240);
    this.bufs   = { "spatial-hash": new Array(this.windowSize), lbvh: new Array(this.windowSize) };
    this.heads  = { "spatial-hash": 0, lbvh: 0 };
    this.counts = { "spatial-hash": 0, lbvh: 0 };
  }

  push(sample: StrategySample): void {
    const s = sample.strategy;
    this.bufs[s][this.heads[s]] = sample;
    this.heads[s] = (this.heads[s] + 1) % this.windowSize;
    if (this.counts[s] < this.windowSize) this.counts[s]++;
  }

  /** Most recently inserted sample for `strategy`, or undefined. */
  latest(strategy: BroadphaseStrategy): StrategySample | undefined {
    if (this.counts[strategy] === 0) return undefined;
    const idx = (this.heads[strategy] - 1 + this.windowSize) % this.windowSize;
    return this.bufs[strategy][idx];
  }

  /** Snapshot all live samples for `strategy` in chronological order. */
  toArray(strategy: BroadphaseStrategy): StrategySample[] {
    const out: StrategySample[] = [];
    const c = this.counts[strategy];
    if (c === 0) return out;
    const start = c < this.windowSize ? 0 : this.heads[strategy];
    for (let i = 0; i < c; i++) {
      out.push(this.bufs[strategy][(start + i) % this.windowSize]);
    }
    return out;
  }

  /** Mean of derived rates over the live window. NaN-safe. */
  aggregate(strategy: BroadphaseStrategy): DerivedRates | undefined {
    const arr = this.toArray(strategy);
    if (arr.length === 0) return undefined;
    const acc: DerivedRates = {
      totalWallMs: 0, totalGpuMs: 0,
      pairsEmittedPerMs: 0, candidatesPerMs: 0,
      emissionRatio: 0, dedupRatio: 0, saturationRatio: 0,
    };
    let gpuN = 0;
    for (const s of arr) {
      acc.totalWallMs       += s.rates.totalWallMs;
      if (Number.isFinite(s.rates.totalGpuMs)) { acc.totalGpuMs += s.rates.totalGpuMs; gpuN++; }
      acc.pairsEmittedPerMs += s.rates.pairsEmittedPerMs;
      acc.candidatesPerMs   += s.rates.candidatesPerMs;
      acc.emissionRatio     += s.rates.emissionRatio;
      acc.dedupRatio        += s.rates.dedupRatio;
      acc.saturationRatio   += s.rates.saturationRatio;
    }
    const n = arr.length;
    return {
      totalWallMs:       acc.totalWallMs / n,
      totalGpuMs:        gpuN > 0 ? acc.totalGpuMs / gpuN : Number.NaN,
      pairsEmittedPerMs: acc.pairsEmittedPerMs / n,
      candidatesPerMs:   acc.candidatesPerMs / n,
      emissionRatio:     acc.emissionRatio / n,
      dedupRatio:        acc.dedupRatio / n,
      saturationRatio:   acc.saturationRatio / n,
    };
  }

  reset(): void {
    this.heads  = { "spatial-hash": 0, lbvh: 0 };
    this.counts = { "spatial-hash": 0, lbvh: 0 };
  }
}

// ── A/B comparison helper ────────────────────────────────────────────────

/** Per-pair (spatial-hash vs lbvh) snapshot for HUD/A-B overlays. */
export interface ComparisonRow {
  metric: string;
  hash: number;
  lbvh: number;
  /** lbvh / hash; NaN when hash is 0. Lower-is-better metrics flag inverted. */
  ratio: number;
  /** Strategy that "wins" this metric per the rule below; "tie" within 1 %. */
  winner: BroadphaseStrategy | "tie";
  /** Whether smaller numbers are better for this metric. */
  lowerIsBetter: boolean;
}

const COMPARE_SPEC: Array<{ key: keyof DerivedRates; label: string; lowerIsBetter: boolean }> = [
  { key: "totalWallMs",       label: "wall ms",            lowerIsBetter: true  },
  { key: "totalGpuMs",        label: "gpu ms",             lowerIsBetter: true  },
  { key: "pairsEmittedPerMs", label: "pairs/ms",           lowerIsBetter: false },
  { key: "candidatesPerMs",   label: "candidates/ms",      lowerIsBetter: false },
  { key: "emissionRatio",     label: "selectivity",        lowerIsBetter: false },
  { key: "dedupRatio",        label: "dedup ratio",        lowerIsBetter: false },
  { key: "saturationRatio",   label: "saturation",         lowerIsBetter: true  },
];

export function compare(hash: DerivedRates, lbvh: DerivedRates): ComparisonRow[] {
  return COMPARE_SPEC.map(({ key, label, lowerIsBetter }) => {
    const h = hash[key], l = lbvh[key];
    const ratio = h !== 0 ? l / h : Number.NaN;
    let winner: BroadphaseStrategy | "tie";
    if (!Number.isFinite(h) || !Number.isFinite(l)) {
      winner = Number.isFinite(h) ? "spatial-hash" : (Number.isFinite(l) ? "lbvh" : "tie");
    } else if (Math.abs(h - l) <= 0.01 * Math.max(Math.abs(h), Math.abs(l), 1e-12)) {
      winner = "tie";
    } else if (lowerIsBetter) {
      winner = h < l ? "spatial-hash" : "lbvh";
    } else {
      winner = h > l ? "spatial-hash" : "lbvh";
    }
    return { metric: label, hash: h, lbvh: l, ratio, winner, lowerIsBetter };
  });
}

// ── WebGPU timestamp helpers ─────────────────────────────────────────────

/** Returns true iff the supplied device exposes the timestamp feature. */
export function deviceHasTimestamp(device: GPUDevice | undefined | null): boolean {
  if (!device) return false;
  // GPUSupportedFeatures is set-like.
  try { return (device.features as unknown as Set<string>).has("timestamp-query"); }
  catch { return false; }
}

export interface TimestampPair {
  querySet: GPUQuerySet;
  resolveBuf: GPUBuffer;
  readBuf: GPUBuffer;
}

/**
 * Allocate a 2-slot timestamp query set + the two buffers needed to
 * resolve and read it back. Returns null when the feature is missing.
 */
export function createTimestampPair(device: GPUDevice): TimestampPair | null {
  if (!deviceHasTimestamp(device)) return null;
  const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
  const resolveBuf = device.createBuffer({
    size: 16, // 2 × u64
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
  });
  const readBuf = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  return { querySet, resolveBuf, readBuf };
}

/**
 * Convert a resolved timestamp buffer (already copied into `readBuf`) to
 * an elapsed-nanoseconds reading. Caller is responsible for calling
 * `mapAsync(GPUMapMode.READ)` on `pair.readBuf` first.
 */
export function readTimestampNs(pair: TimestampPair): GPUTimestampReadback {
  const view = new BigUint64Array(pair.readBuf.getMappedRange());
  const begin = view[0], end = view[1];
  const elapsed = end >= begin ? end - begin : 0n;
  pair.readBuf.unmap();
  return { elapsedNs: Number(elapsed) };
}

// ── Internals ────────────────────────────────────────────────────────────

function nowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}
