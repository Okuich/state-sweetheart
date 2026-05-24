/**
 * Predictive Maintenance — Gating System.
 *
 * Evaluates the engine against three production-readiness gates:
 *
 *   Gate A — False Positive Rate < 8%
 *     (fraction of healthy samples that produced an alert)
 *
 *   Gate B — Prediction Accuracy > 80%
 *     (TP + TN over total labeled samples, balanced)
 *
 *   Gate C — Inference Latency < 100 ms
 *     (rolling p95 of engine.ingest() wall time)
 *
 * Passing all three gates unlocks the Digital Twin Layer.
 *
 * The evaluator is pure / framework-agnostic — feed it labeled outcomes
 * and per-sample latencies, read back the gate report.
 */

import type { MaintenanceScore } from "./types";

export type GroundTruth = "healthy" | "fault";

export interface GateThresholds {
  /** Maximum allowed false-positive rate (default 0.08). */
  maxFPR: number;
  /** Minimum balanced accuracy (default 0.80). */
  minAccuracy: number;
  /** Maximum allowed p95 inference latency in ms (default 100). */
  maxLatencyMs: number;
  /** Decision threshold on riskScore for "alert / no alert". */
  alertThreshold: number;
  /** Minimum samples per class before a gate is considered evaluable. */
  minSamples: number;
}

export const DEFAULT_GATES: GateThresholds = {
  maxFPR: 0.08,
  minAccuracy: 0.80,
  maxLatencyMs: 100,
  alertThreshold: 0.45,
  minSamples: 20,
};

export interface GateStatus {
  pass: boolean;
  value: number;
  threshold: number;
  ready: boolean; // enough data to evaluate
}

export interface GateReport {
  A: GateStatus; // false-positive rate
  B: GateStatus; // accuracy
  C: GateStatus; // latency p95
  unlocked: boolean;
  counts: { healthy: number; fault: number; tp: number; tn: number; fp: number; fn: number };
  latency: { p50: number; p95: number; max: number; n: number };
}

export class GateEvaluator {
  private tp = 0;
  private tn = 0;
  private fp = 0;
  private fn = 0;
  private healthy = 0;
  private fault = 0;
  // Rolling latency window (ring buffer) to keep p95 reactive.
  private latencies: number[] = [];
  private readonly latCap = 256;

  constructor(public thresholds: GateThresholds = DEFAULT_GATES) {}

  /**
   * Record a single labeled outcome.
   *   truth   — ground-truth state of the asset at that instant
   *   score   — the engine's MaintenanceScore for that ingest
   *   latency — wall-clock ms taken by engine.ingest()
   */
  record(truth: GroundTruth, score: MaintenanceScore, latencyMs: number): void {
    const alert = score.riskScore >= this.thresholds.alertThreshold;
    if (truth === "healthy") {
      this.healthy++;
      if (alert) this.fp++; else this.tn++;
    } else {
      this.fault++;
      if (alert) this.tp++; else this.fn++;
    }
    this.latencies.push(latencyMs);
    if (this.latencies.length > this.latCap) this.latencies.shift();
  }

  reset(): void {
    this.tp = this.tn = this.fp = this.fn = 0;
    this.healthy = this.fault = 0;
    this.latencies = [];
  }

  report(): GateReport {
    const t = this.thresholds;

    const fpr = this.healthy > 0 ? this.fp / this.healthy : 0;
    const tpr = this.fault   > 0 ? this.tp / this.fault   : 0;
    const tnr = this.healthy > 0 ? this.tn / this.healthy : 0;
    // Balanced accuracy: average of class accuracies; robust to skew.
    const balAcc = this.fault > 0 && this.healthy > 0 ? 0.5 * (tpr + tnr) : 0;

    const { p50, p95, max } = percentiles(this.latencies);

    const readyAB = this.healthy >= t.minSamples && this.fault >= t.minSamples;
    const readyC  = this.latencies.length >= Math.min(20, t.minSamples);

    const A: GateStatus = {
      pass: readyAB && fpr < t.maxFPR,
      value: fpr, threshold: t.maxFPR, ready: readyAB,
    };
    const B: GateStatus = {
      pass: readyAB && balAcc > t.minAccuracy,
      value: balAcc, threshold: t.minAccuracy, ready: readyAB,
    };
    const C: GateStatus = {
      pass: readyC && p95 < t.maxLatencyMs,
      value: p95, threshold: t.maxLatencyMs, ready: readyC,
    };

    return {
      A, B, C,
      unlocked: A.pass && B.pass && C.pass,
      counts: {
        healthy: this.healthy, fault: this.fault,
        tp: this.tp, tn: this.tn, fp: this.fp, fn: this.fn,
      },
      latency: { p50, p95, max, n: this.latencies.length },
    };
  }
}

function percentiles(xs: number[]): { p50: number; p95: number; max: number } {
  if (xs.length === 0) return { p50: 0, p95: 0, max: 0 };
  const s = [...xs].sort((a, b) => a - b);
  const at = (q: number) => s[Math.min(s.length - 1, Math.max(0, Math.floor(q * (s.length - 1))))];
  return { p50: at(0.5), p95: at(0.95), max: s[s.length - 1] };
}
