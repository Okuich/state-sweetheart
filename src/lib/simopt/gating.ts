/**
 * Simulation Optimization — Gating System.
 *
 *   Gate A — accuracy degradation ≤ 2%
 *     (relative-error of routed answers vs full-fidelity baseline)
 *
 *   Gate B — acceleration ≥ 20%
 *     (compute time saved vs estimated full-solve cost)
 *
 *   Gate C — solver stability improvement ≥ 10%
 *     (routed convergence rate vs baseline failure rate)
 *
 * Passing all three unlocks the Geometry-Aware Physics layer.
 */

import type { SimOptStats } from "./types";

export interface SimGateThresholds {
  /** Max accuracy degradation (e.g. 0.02 = 2%). */
  maxAccuracyDelta: number;
  /** Min compute saving fraction (0.20 = 20%). */
  minAcceleration: number;
  /** Min stability improvement vs baseline (0.10 = 10%). */
  minStabilityImprovement: number;
  /** Min samples before a gate is evaluable. */
  minSamples: number;
}

export const DEFAULT_SIM_GATES: SimGateThresholds = {
  maxAccuracyDelta: 0.02,
  minAcceleration: 0.20,
  minStabilityImprovement: 0.10,
  minSamples: 24,
};

export interface SimGateStatus {
  pass: boolean;
  value: number;
  threshold: number;
  ready: boolean;
}

export interface SimGateReport {
  A: SimGateStatus; // accuracy degradation
  B: SimGateStatus; // acceleration
  C: SimGateStatus; // stability improvement
  unlocked: boolean;
  samples: number;
}

/**
 * Stateful evaluator. Caller records (estError, fullMs, routedMs,
 * routedConverged) per request and reads back the gate report.
 *
 * Baseline stability is observed from full-solve outcomes: the
 * evaluator tracks routed and baseline (full) convergence rates
 * separately so it can compute a meaningful delta even when most
 * traffic is served from cache / ROM.
 */
export class SimGateEvaluator {
  private errors: number[] = [];   // estimated relative error per request
  private fullMs = 0;              // sum of est full-solve ms
  private routedMs = 0;            // sum of actual routed ms
  private samples = 0;
  // Stability tracking — split by whether the request fell back to a full solve.
  private baselineOutcomes: number[] = []; // 1/0 for full-solve runs only
  private routedOutcomes:   number[] = []; // 1/0 for all routed runs
  private readonly cap = 256;

  constructor(public thresholds: SimGateThresholds = DEFAULT_SIM_GATES) {}

  /** Record one routed request outcome. */
  record(args: {
    estError: number;
    fullMs: number;
    routedMs: number;
    converged: boolean;
    stage: "cache" | "rom" | "full";
  }): void {
    this.samples++;
    this.errors.push(Math.max(0, args.estError));
    if (this.errors.length > this.cap) this.errors.shift();
    this.fullMs += Math.max(0, args.fullMs);
    this.routedMs += Math.max(0, args.routedMs);
    const ok = args.converged ? 1 : 0;
    this.routedOutcomes.push(ok);
    if (this.routedOutcomes.length > this.cap) this.routedOutcomes.shift();
    if (args.stage === "full") {
      this.baselineOutcomes.push(ok);
      if (this.baselineOutcomes.length > this.cap) this.baselineOutcomes.shift();
    }
  }

  reset(): void {
    this.errors = [];
    this.fullMs = this.routedMs = 0;
    this.samples = 0;
    this.baselineOutcomes = [];
    this.routedOutcomes = [];
  }

  report(stats?: SimOptStats): SimGateReport {
    const t = this.thresholds;
    const ready = this.samples >= t.minSamples;

    // A. accuracy degradation = mean of estError (relative).
    const meanErr = this.errors.length
      ? this.errors.reduce((s, x) => s + x, 0) / this.errors.length
      : 0;

    // B. acceleration = (estFull - routed) / estFull.
    const accel = this.fullMs > 0 ? Math.max(0, (this.fullMs - this.routedMs) / this.fullMs) : 0;

    // C. stability improvement = routedRate - baselineRate.
    //    If we don't yet have baseline samples, fall back to 0 (not ready).
    const routedRate = mean(this.routedOutcomes);
    const baselineRate = this.baselineOutcomes.length >= 8 ? mean(this.baselineOutcomes) : NaN;
    const stabDelta = Number.isFinite(baselineRate) ? routedRate - baselineRate : 0;
    const readyC = ready && this.baselineOutcomes.length >= 8;

    void stats; // reserved for future engine-side cross-checks
    return {
      A: { pass: ready && meanErr <= t.maxAccuracyDelta,
           value: meanErr, threshold: t.maxAccuracyDelta, ready },
      B: { pass: ready && accel >= t.minAcceleration,
           value: accel, threshold: t.minAcceleration, ready },
      C: { pass: readyC && stabDelta >= t.minStabilityImprovement,
           value: Number.isFinite(stabDelta) ? stabDelta : 0,
           threshold: t.minStabilityImprovement, ready: readyC },
      unlocked: false, // filled below
      samples: this.samples,
    } as SimGateReport;
  }
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0; for (const x of xs) s += x;
  return s / xs.length;
}

/** Convenience: finalize the `unlocked` flag. */
export function finalize(r: SimGateReport): SimGateReport {
  r.unlocked = r.A.pass && r.B.pass && r.C.pass;
  return r;
}
