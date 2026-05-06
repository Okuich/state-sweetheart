/**
 * determinism.ts
 * ──────────────────────────────────────────────────────────────────────────
 * Determinism harness for the MPI orchestrator + simulation kernels.
 *
 * Why this is non-trivial
 * -----------------------
 * Floating-point addition is NOT associative:
 *     (a + b) + c  !=  a + (b + c)        for IEEE-754 doubles.
 * So if rank-contributions to a global sum are reduced in different orders
 * across runs (e.g. because OS scheduling races the order Promise.all
 * resolves), the bit pattern of the result differs even though the math is
 * "the same". A determinism harness therefore needs three things:
 *
 *   (1) A FIXED reduction tree — always combine ranks in canonical
 *       (rank-ascending) order, regardless of arrival order. We provide
 *       `deterministicReduce` which sorts contributions by rank index
 *       before folding them.
 *
 *   (2) A bit-stable HASH that's sensitive to every bit of every float —
 *       FNV-1a 64 over the raw IEEE-754 bytes. Two identical Float64Arrays
 *       hash identically; flipping ANY mantissa bit changes the digest.
 *
 *   (3) A REPLAY harness: run the same kernel twice with the same seed,
 *       hash per-timestep state, and assert digests match step-for-step.
 *       The first divergent step is reported with its step index and both
 *       digests — pin-pointing exactly where non-determinism crept in.
 *
 * The harness is kernel-agnostic: it takes a callback that produces a
 * `Float64Array` snapshot of whatever state you care about (positions,
 * velocities, energy contributions). Bit-identical snapshots → identical
 * digests → deterministic.
 */

import {
  MpiOrchestrator,
  type StepKernel,
  type StepReport,
} from "./mpiOrchestrator";

// ── Bit-stable hashing ───────────────────────────────────────────────────

/**
 * FNV-1a 64-bit hash, returned as a 16-char lowercase hex string.
 * Operates over the raw byte view of any TypedArray, so it sees every
 * sign / exponent / mantissa bit of every float. Pure JS; no Node deps.
 *
 * BigInt is used because JS numbers can't represent 64-bit unsigned
 * integers exactly. The cost (~50-100 ns/byte) is fine for snapshot-
 * sized buffers (≤ a few hundred KB / step) but don't drop it on a
 * gigabyte of state per frame.
 */
export function fnv1a64Hex(bytes: Uint8Array): string {
  const FNV_OFFSET = 0xcbf29ce484222325n;
  const FNV_PRIME  = 0x00000100000001b3n;
  const MASK       = 0xffffffffffffffffn;
  let h = FNV_OFFSET;
  for (let i = 0; i < bytes.length; i++) {
    h = (h ^ BigInt(bytes[i])) & MASK;
    h = (h * FNV_PRIME) & MASK;
  }
  return h.toString(16).padStart(16, "0");
}

/** Hash any TypedArray view — pulls the raw bytes from the underlying buffer. */
export function hashFloat64Array(a: Float64Array): string {
  return fnv1a64Hex(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
}

export function hashFloat32Array(a: Float32Array): string {
  return fnv1a64Hex(new Uint8Array(a.buffer, a.byteOffset, a.byteLength));
}

// ── Fixed reduction tree ─────────────────────────────────────────────────

/**
 * Deterministic reduction over rank-indexed contributions.
 *
 * Rule: contributions MUST be combined in ascending rank order (the
 * "fixed tree"). For sum/product we additionally use Neumaier (compensated)
 * summation so the result is stable even when magnitudes vary by orders
 * of magnitude — bit-identical across runs as long as the input array
 * is bit-identical.
 *
 * `op === "min" | "max"` is associative AND commutative on floats, so a
 * straight left-fold suffices.
 */
export type DeterministicOp = "sum" | "min" | "max";

export function deterministicReduce(
  contribs: ArrayLike<number>,
  op: DeterministicOp,
): number {
  const n = contribs.length;
  if (n === 0) return op === "sum" ? 0 : op === "min" ? Infinity : -Infinity;
  if (op === "min") {
    let m = contribs[0];
    for (let i = 1; i < n; i++) if (contribs[i] < m) m = contribs[i];
    return m;
  }
  if (op === "max") {
    let m = contribs[0];
    for (let i = 1; i < n; i++) if (contribs[i] > m) m = contribs[i];
    return m;
  }
  // Neumaier compensated sum — eliminates the order-dependence that plain
  // Kahan still has when adjacent magnitudes invert.
  let sum = 0;
  let comp = 0;
  for (let i = 0; i < n; i++) {
    const v = contribs[i];
    const t = sum + v;
    if (Math.abs(sum) >= Math.abs(v)) comp += (sum - t) + v;
    else                              comp += (v - t) + sum;
    sum = t;
  }
  return sum + comp;
}

// ── Deterministic LCG (so kernels can be seeded reproducibly) ────────────

/**
 * Mulberry32 — 32-bit state, period 2³², perfectly reproducible across
 * platforms (no Math.random differences). Returns floats in [0, 1).
 */
export function makeRng(seed: number): () => number {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── Replay harness ───────────────────────────────────────────────────────

export interface DeterminismResult {
  /** True iff every snapshot digest matched between runs A and B. */
  deterministic: boolean;
  /** Number of timesteps actually executed (== nSteps unless aborted). */
  steps: number;
  /** Per-step digests from run A and run B (parallel arrays). */
  digestsA: string[];
  digestsB: string[];
  /** First step where digests diverged, or -1 if all matched. */
  firstDivergentStep: number;
  /** Side-by-side digest at the divergent step (or null if none). */
  divergence: { step: number; a: string; b: string } | null;
  /** Per-step report from run A (run B is asserted identical). */
  reportsA: StepReport[];
}

export interface DeterminismOptions {
  /** Number of timesteps to drive. */
  nSteps: number;
  /** Number of MPI ranks. */
  size: number;
  /** Global particle count. */
  N: number;
  /** Initial timestep. */
  dt0: number;
  /**
   * Factory for the kernel under test. Called twice (once per run) with
   * a fresh deterministic RNG seeded from the same seed each time. The
   * factory MUST close over no shared mutable state — otherwise run B
   * inherits run A's drift and we fail to detect non-determinism.
   */
  kernelFactory: (rng: () => number) => StepKernel;
  /**
   * Snapshot extractor: called after every step, returns the state to
   * hash. Must produce a Float64Array whose bytes deterministically
   * encode whatever you want to verify (positions, energies, etc.).
   */
  snapshot: (orch: MpiOrchestrator, step: number, report: StepReport) => Float64Array;
  /** RNG seed shared by both runs (default: 0xC0FFEE). */
  seed?: number;
}

/**
 * Run the kernel TWICE under identical configuration and verify each
 * timestep's snapshot hashes to the same digest. If any step diverges,
 * `firstDivergentStep` and the two digests are reported.
 *
 * Note: this harness intentionally does NOT use any source of platform
 * non-determinism (Date.now, Math.random, performance.now-driven jitter)
 * to drive the kernel. The kernel must consume `rng` for any randomness
 * it needs. If your kernel pulls from Math.random or system entropy,
 * the harness will (correctly) flag it as non-deterministic.
 */
export async function checkDeterminism(opts: DeterminismOptions): Promise<DeterminismResult> {
  const seed = opts.seed ?? 0xC0FFEE;

  const runOnce = async (): Promise<{ digests: string[]; reports: StepReport[] }> => {
    const orch = new MpiOrchestrator({
      size: opts.size, N: opts.N, dt0: opts.dt0,
      // Zero collective latency so test wall-time is dominated by kernel,
      // and so we don't introduce scheduler-dependent ordering.
      collectiveLatencyMs: 0,
      // Long deadlock window — these tests can be I/O-light but RNG-heavy.
      deadlockTimeoutMs: 30_000,
    });
    const rng = makeRng(seed);
    const kernel = opts.kernelFactory(rng);
    const digests: string[] = [];
    const reports: StepReport[] = [];
    for (let i = 0; i < opts.nSteps; i++) {
      const rep = await orch.runStep(kernel);
      reports.push(rep);
      const snap = opts.snapshot(orch, i, rep);
      digests.push(hashFloat64Array(snap));
    }
    return { digests, reports };
  };

  const a = await runOnce();
  const b = await runOnce();

  let firstDivergentStep = -1;
  for (let i = 0; i < a.digests.length; i++) {
    if (a.digests[i] !== b.digests[i]) { firstDivergentStep = i; break; }
  }

  return {
    deterministic: firstDivergentStep === -1,
    steps: a.digests.length,
    digestsA: a.digests,
    digestsB: b.digests,
    firstDivergentStep,
    divergence: firstDivergentStep === -1 ? null : {
      step: firstDivergentStep,
      a: a.digests[firstDivergentStep],
      b: b.digests[firstDivergentStep],
    },
    reportsA: a.reports,
  };
}
