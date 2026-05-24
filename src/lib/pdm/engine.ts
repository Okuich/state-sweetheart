/**
 * Predictive Maintenance Metric Engine.
 *
 * Maintains a per-asset rolling state in latent space, tracks distance
 * to the asset's known-good (optimal) manifold and historical failure
 * states, and produces a fused MaintenanceScore on every ingest.
 *
 *   ingest(reading)  →  pack → embed → drift / kNN / trajectory → score
 *
 * Designed to run inline with the existing telemetry bus: ingest cost
 * is O(d·k + |library|·k) per sample with a handful of float ops on top.
 */

import {
  AdaptiveThreshold, RunningMoments, kmeans, nearest, sigmaDistance, trajectoryDistance,
} from "./anomaly";
import { fitEmbedding, packReading, project, type EmbeddingModel } from "./embeddings";
import type { LabeledState, MaintenanceScore, SensorReading } from "./types";

export interface PDMConfig {
  /** Latent dim for the PCA projection. */
  latentDim?: number;
  /** Trajectory window length (samples). */
  windowLen?: number;
  /** Min samples before scoring becomes meaningful. */
  warmupSamples?: number;
  /** Max latent vectors held per asset. */
  historyCap?: number;
  /** Number of cluster centroids for similarity clustering. */
  clusters?: number;
  /** Weights for fused risk score. */
  weights?: { drift?: number; failure?: number; optimal?: number; trajectory?: number };
}

interface AssetState {
  rm: RunningMoments;
  history: Float64Array[];        // latent vectors (rolling)
  rawHistory: Float64Array[];     // raw state vectors (for refitting)
  threshold: AdaptiveThreshold;
  lastRisk: number;
  lastRiskRate: number;           // d(risk)/dt
  lastT: number;
  embedding: EmbeddingModel | null;
  centroids: Float64Array[];
}

export class PredictiveMaintenanceEngine {
  private cfg: Required<PDMConfig>;
  private assets = new Map<string, AssetState>();
  private optimal = new Map<string, Float64Array[]>();  // latent per asset
  private failures = new Map<string, LabeledState[]>(); // latent per asset
  private failureTraj = new Map<string, Float64Array[][]>(); // labeled trajectories

  constructor(cfg: PDMConfig = {}) {
    this.cfg = {
      latentDim: cfg.latentDim ?? 4,
      windowLen: cfg.windowLen ?? 16,
      warmupSamples: cfg.warmupSamples ?? 24,
      historyCap: cfg.historyCap ?? 512,
      clusters: cfg.clusters ?? 4,
      weights: {
        drift: cfg.weights?.drift ?? 0.45,
        failure: cfg.weights?.failure ?? 0.30,
        optimal: cfg.weights?.optimal ?? 0.15,
        trajectory: cfg.weights?.trajectory ?? 0.10,
      },
    };
  }

  /** Seed the engine with known-good and failure exemplars (raw vectors). */
  registerExemplars(assetId: string, opts: {
    optimal?: SensorReading[]; failures?: SensorReading[]; failureTrajectories?: SensorReading[][];
  }): void {
    const st = this.ensure(assetId);
    const all = [
      ...(opts.optimal ?? []).map((r) => packReading(r)),
      ...(opts.failures ?? []).map((r) => packReading(r)),
      ...(opts.failureTrajectories ?? []).flat().map((r) => packReading(r)),
    ];
    if (all.length >= 2) {
      st.embedding = fitEmbedding(all, this.cfg.latentDim);
    }
    if (st.embedding) {
      if (opts.optimal?.length) {
        this.optimal.set(assetId, opts.optimal.map((r) => project(st.embedding!, packReading(r))));
      }
      if (opts.failures?.length) {
        this.failures.set(assetId, opts.failures.map((r) => ({
          v: project(st.embedding!, packReading(r)), label: r.regime ?? "failure",
        })));
      }
      if (opts.failureTrajectories?.length) {
        this.failureTraj.set(assetId, opts.failureTrajectories.map((tr) =>
          tr.map((r) => project(st.embedding!, packReading(r)))));
      }
    }
  }

  /** Feed one telemetry reading. Returns the fused score. */
  ingest(reading: SensorReading): MaintenanceScore {
    const st = this.ensure(reading.assetId);
    const raw = packReading(reading, st.rm.n > 0 ? st.rm.mean : undefined);
    st.rawHistory.push(raw);
    if (st.rawHistory.length > this.cfg.historyCap) st.rawHistory.shift();

    // Lazily (re)fit embedding once we have enough raw history.
    if (!st.embedding && st.rawHistory.length >= Math.max(6, this.cfg.latentDim + 2)) {
      st.embedding = fitEmbedding(st.rawHistory, this.cfg.latentDim);
    }
    const latent = st.embedding ? project(st.embedding, raw) : raw.slice();
    if (st.rm.d !== latent.length) (st.rm as { d: number }).d = latent.length;
    if (st.rm.mean.length !== latent.length) {
      st.rm = new RunningMoments(latent.length);
    }
    st.rm.update(latent);
    st.history.push(latent);
    if (st.history.length > this.cfg.historyCap) st.history.shift();

    // Refresh clusters periodically.
    if (st.history.length >= 8 && st.history.length % 32 === 0) {
      st.centroids = kmeans(st.history, this.cfg.clusters, 12).centroids;
    }

    return this.score(reading.assetId, reading.t, latent, st);
  }

  /** Get most recent score without ingesting. */
  snapshot(assetId: string): MaintenanceScore | null {
    const st = this.assets.get(assetId);
    if (!st || st.history.length === 0) return null;
    return this.score(assetId, st.lastT, st.history[st.history.length - 1], st);
  }

  /** All assets sorted by priority (urgent first). */
  prioritized(): MaintenanceScore[] {
    const rank: Record<MaintenanceScore["priority"], number> = { urgent: 3, schedule: 2, watch: 1, ok: 0 };
    const out: MaintenanceScore[] = [];
    for (const id of this.assets.keys()) {
      const s = this.snapshot(id);
      if (s) out.push(s);
    }
    out.sort((a, b) => rank[b.priority] - rank[a.priority] || b.riskScore - a.riskScore);
    return out;
  }

  reset(): void { this.assets.clear(); this.optimal.clear(); this.failures.clear(); this.failureTraj.clear(); }

  // ---------------- internals ----------------

  private ensure(id: string): AssetState {
    let st = this.assets.get(id);
    if (!st) {
      st = {
        rm: new RunningMoments(this.cfg.latentDim),
        history: [], rawHistory: [],
        threshold: new AdaptiveThreshold(0.05, 3),
        lastRisk: 0, lastRiskRate: 0, lastT: 0,
        embedding: null, centroids: [],
      };
      this.assets.set(id, st);
    }
    return st;
  }

  private score(assetId: string, t: number, latent: Float64Array, st: AssetState): MaintenanceScore {
    // 1. Drift in σ (diagonal Mahalanobis from running mean).
    const sigma = sigmaDistance(st.rm, latent);
    st.threshold.update(sigma);
    const driftNorm = clamp01(sigma / Math.max(1e-3, st.threshold.threshold));

    // 2. Distance to optimal / failure libraries (per-asset).
    const opt = this.optimal.get(assetId) ?? [];
    const fail = (this.failures.get(assetId) ?? []).map((l) => l.v);
    const dOpt = opt.length ? nearest(opt, latent).dist : 0;
    const dFail = fail.length ? nearest(fail, latent).dist : Infinity;
    const refScale = Math.max(1e-3, st.threshold.ewma + 1e-3);
    const optNorm = clamp01(dOpt / refScale);
    const failNorm = fail.length ? 1 - clamp01(dFail / (refScale * 2)) : 0;

    // 3. Trajectory similarity to known failure trajectories.
    const w = this.cfg.windowLen;
    const tail = st.history.slice(-w);
    const trajs = this.failureTraj.get(assetId) ?? [];
    let trajNorm = 0;
    if (tail.length === w && trajs.length) {
      let best = Infinity;
      for (const tr of trajs) {
        if (tr.length < w) continue;
        const d = trajectoryDistance(tail, tr);
        if (d < best) best = d;
      }
      if (Number.isFinite(best)) trajNorm = 1 - clamp01(best / (refScale * 2));
    }

    // 4. Cluster id (nearest centroid).
    let clusterId = -1;
    if (st.centroids.length) {
      const { idx } = nearest(st.centroids, latent);
      clusterId = idx;
    }

    // 5. Fused risk.
    const W = this.cfg.weights;
    const warmup = st.rm.n < this.cfg.warmupSamples ? 0.3 : 1;
    const risk = clamp01(
      warmup * (
        (W.drift ?? 0) * driftNorm +
        (W.failure ?? 0) * failNorm +
        (W.optimal ?? 0) * optNorm +
        (W.trajectory ?? 0) * trajNorm
      )
    );

    // 6. Failure horizon via linear extrapolation of risk-rate.
    const dt = Math.max(1e-3, t - (st.lastT || t));
    const rate = (risk - st.lastRisk) / dt;
    st.lastRiskRate = 0.7 * st.lastRiskRate + 0.3 * rate;
    const horizon = st.lastRiskRate > 1e-6 ? Math.max(1, (1 - risk) / st.lastRiskRate) : Infinity;
    st.lastRisk = risk; st.lastT = t;

    const priority: MaintenanceScore["priority"] =
      risk >= 0.85 ? "urgent" :
      risk >= 0.6  ? "schedule" :
      risk >= 0.35 ? "watch" : "ok";

    return {
      assetId, t, riskScore: risk,
      driftSigma: sigma, distToOptimal: dOpt,
      distToFailure: Number.isFinite(dFail) ? dFail : -1,
      failureHorizon: horizon, clusterId, priority,
    };
  }
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
