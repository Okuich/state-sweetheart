/**
 * Material Intelligence — Gating System.
 *
 *   Gate A — Recommendation relevance > 85%
 *     (mean composite score of the top-K recommendations, with
 *      feasibility as a gating multiplier — irrelevant picks
 *      can't artificially inflate the score)
 *
 *   Gate B — Constraint satisfaction > 95%
 *     (fraction of top-K recommendations passing all hard
 *      constraints across a stream of design scenarios)
 *
 *   Gate C — Inference latency < 200 ms
 *     (rolling p95 of recommend() wall time)
 *
 * Passing all three unlocks the Procurement Optimization Layer.
 */

import type { MaterialScore } from "./types";

export interface MatGateThresholds {
  /** Min mean relevance of top-K (0..1, default 0.85). */
  minRelevance: number;
  /** Min fraction of top-K that satisfy all hard constraints (default 0.95). */
  minConstraintSat: number;
  /** Max p95 latency (ms, default 200). */
  maxLatencyMs: number;
  /** K = top-K used for relevance + constraint metrics. */
  topK: number;
  /** Min recorded scenarios before A/B are evaluable. */
  minScenarios: number;
}

export const DEFAULT_MAT_GATES: MatGateThresholds = {
  minRelevance: 0.85,
  minConstraintSat: 0.95,
  maxLatencyMs: 200,
  topK: 5,
  minScenarios: 8,
};

export interface MatGateStatus {
  pass: boolean;
  value: number;
  threshold: number;
  ready: boolean;
}

export interface MatGateReport {
  A: MatGateStatus; // relevance
  B: MatGateStatus; // constraint satisfaction
  C: MatGateStatus; // latency
  unlocked: boolean;
  scenarios: number;
  latency: { p50: number; p95: number; max: number; n: number };
}

export class MatGateEvaluator {
  private relSum = 0;
  private satSum = 0;
  private scenarios = 0;
  private latencies: number[] = [];
  private readonly latCap = 256;

  constructor(public thresholds: MatGateThresholds = DEFAULT_MAT_GATES) {}

  /**
   * Record one design scenario.
   *   ranked   — recommendations sorted best-first (output of recommend())
   *   latency  — wall-clock ms taken by recommend()
   */
  record(ranked: MaterialScore[], latencyMs: number): void {
    const k = Math.min(this.thresholds.topK, ranked.length);
    if (k === 0) return;
    const top = ranked.slice(0, k);
    // Relevance: weight composite score by feasibility (infeasible = 0.3×).
    const rel = top.reduce(
      (acc, s) => acc + s.score * (s.feasible ? 1 : 0.3),
      0,
    ) / k;
    const sat = top.filter((s) => s.feasible).length / k;

    this.relSum += rel;
    this.satSum += sat;
    this.scenarios++;
    this.latencies.push(latencyMs);
    if (this.latencies.length > this.latCap) this.latencies.shift();
  }

  reset(): void {
    this.relSum = this.satSum = 0;
    this.scenarios = 0;
    this.latencies = [];
  }

  report(): MatGateReport {
    const t = this.thresholds;
    const relevance = this.scenarios > 0 ? this.relSum / this.scenarios : 0;
    const constraintSat = this.scenarios > 0 ? this.satSum / this.scenarios : 0;
    const { p50, p95, max } = percentiles(this.latencies);

    const readyAB = this.scenarios >= t.minScenarios;
    const readyC = this.latencies.length >= Math.min(8, t.minScenarios);

    const A: MatGateStatus = {
      pass: readyAB && relevance > t.minRelevance,
      value: relevance, threshold: t.minRelevance, ready: readyAB,
    };
    const B: MatGateStatus = {
      pass: readyAB && constraintSat > t.minConstraintSat,
      value: constraintSat, threshold: t.minConstraintSat, ready: readyAB,
    };
    const C: MatGateStatus = {
      pass: readyC && p95 < t.maxLatencyMs,
      value: p95, threshold: t.maxLatencyMs, ready: readyC,
    };

    return {
      A, B, C,
      unlocked: A.pass && B.pass && C.pass,
      scenarios: this.scenarios,
      latency: { p50, p95, max, n: this.latencies.length },
    };
  }
}

function percentiles(xs: number[]): { p50: number; p95: number; max: number } {
  if (xs.length === 0) return { p50: 0, p95: 0, max: 0 };
  const s = [...xs].sort((a, b) => a - b);
  const at = (q: number) =>
    s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))];
  return { p50: at(0.5), p95: at(0.95), max: s[s.length - 1] };
}
