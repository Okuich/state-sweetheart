/**
 * Simulation Optimization Engine.
 *
 *   request(desc) →
 *     1. featurize + project to latent
 *     2. mesh quality / instability guard
 *     3. nearest-neighbor search in the cache → cache hit
 *     4. ROM surrogate eligibility check       → rom hit
 *     5. otherwise route to best solver        → full solve
 *
 * The engine is pure logic — callers supply the actual solve via
 * runSolve(desc, decision). The engine handles caching, decision
 * confidence, error bounds, and book-keeping for stats.
 */

import { fitEmbedding, packDescriptor, project, RAW_FEATURE_DIM, type EmbeddingModel } from "./embeddings";
import { evalMeshQuality } from "./mesh";
import type { CachedSim, RouteDecision, SimDescriptor, SimOptStats, SimResult, SolverKind } from "./types";

export interface SimOptConfig {
  latentDim?: number;
  cacheCap?: number;
  /** Latent distance below which we serve from cache. */
  cacheRadius?: number;
  /** Latent distance below which a ROM surrogate is acceptable. */
  romRadius?: number;
  /** Max acceptable estimated relative error for cache hits. */
  maxCacheError?: number;
  /** Floor mesh quality before we refuse to reuse. */
  minMeshQuality?: number;
}

const DEFAULTS: Required<SimOptConfig> = {
  latentDim: 6,
  cacheCap: 256,
  cacheRadius: 0.35,
  romRadius: 1.2,
  maxCacheError: 0.05,
  minMeshQuality: 0.4,
};

export class SimulationOptimizationEngine {
  private cfg: Required<SimOptConfig>;
  private cache: CachedSim[] = [];
  private embedding: EmbeddingModel | null = null;
  private rawHistory: Float64Array[] = [];
  // History of "full" residuals used to estimate convergence speedup.
  private baselineIters: number[] = [];
  private routedIters: number[] = [];
  // Stability rolling window — 1 if routing decision converged, 0 otherwise.
  private decisionOutcomes: number[] = [];

  private stats: SimOptStats = {
    totalRequests: 0, cacheHits: 0, romHits: 0, fullSolves: 0,
    computeSavedMs: 0, estComputeFullMs: 0,
    convergenceSpeedup: 1, stabilityScore: 1,
  };

  constructor(cfg: SimOptConfig = {}) {
    this.cfg = { ...DEFAULTS, ...cfg };
  }

  /** Decide what to do with a request — does not run the solve. */
  route(desc: SimDescriptor): RouteDecision {
    const raw = packDescriptor(desc);
    this.rawHistory.push(raw);
    if (this.rawHistory.length > this.cfg.cacheCap * 2) this.rawHistory.shift();

    // Lazily (re)fit PCA once we have a few exemplars.
    if (!this.embedding && this.rawHistory.length >= Math.max(6, this.cfg.latentDim + 2)) {
      this.embedding = fitEmbedding(this.rawHistory, this.cfg.latentDim);
    }
    const latent = this.embedding ? project(this.embedding, raw) : raw.slice();

    const mq = evalMeshQuality(desc.mesh);
    // Adjust effective condition number with mesh penalty.
    const effKappa = desc.logKappa + mq.conditioningPenalty;

    // 1. Nearest cache entry.
    const { idx, dist } = nearest(this.cache, latent);
    const nearest1 = idx >= 0 ? this.cache[idx] : undefined;

    // Cache hit policy: same domain, same regime, mesh not degenerate,
    // latent distance below radius, estimated error below tolerance.
    if (nearest1 && mq.score >= this.cfg.minMeshQuality) {
      if (
        nearest1.desc.domain === desc.domain &&
        nearest1.desc.regime === desc.regime &&
        dist <= this.cfg.cacheRadius
      ) {
        const estError = estimateRelativeError(dist, this.cfg.cacheRadius);
        if (estError <= this.cfg.maxCacheError) {
          return {
            solver: nearest1.result.solver, stage: "cache",
            confidence: 1 - dist / this.cfg.cacheRadius,
            reason: `cache hit · d=${dist.toFixed(3)} · err≈${(estError * 100).toFixed(2)}%`,
            estError, nearestId: nearest1.desc.id, nearestDist: dist,
          };
        }
      }

      // 2. ROM eligibility: same domain, larger radius, SPD problems
      // benefit most from POD-Galerkin reuse.
      if (
        nearest1.desc.domain === desc.domain &&
        dist <= this.cfg.romRadius &&
        (desc.spd || desc.nonlinearity < 0.4)
      ) {
        const estError = estimateRelativeError(dist, this.cfg.romRadius) * 1.5;
        return {
          solver: "rom", stage: "rom",
          confidence: 1 - dist / this.cfg.romRadius,
          reason: `ROM surrogate · d=${dist.toFixed(3)} · err≈${(estError * 100).toFixed(1)}%`,
          estError, nearestId: nearest1.desc.id, nearestDist: dist,
        };
      }
    }

    // 3. Full solve — pick the best solver for the regime.
    const solver = pickSolver(desc, effKappa, mq.instability);
    return {
      solver, stage: "full",
      confidence: solverConfidence(desc, solver, mq.instability),
      reason: `full solve · ${solver} · effκ=10^${effKappa.toFixed(1)} · mq=${mq.score.toFixed(2)}`,
      nearestId: nearest1?.desc.id, nearestDist: nearest1 ? dist : undefined,
    };
  }

  /** Record the outcome of a routing decision; updates cache and stats. */
  record(desc: SimDescriptor, decision: RouteDecision, result: SimResult): void {
    this.stats.totalRequests++;
    const raw = packDescriptor(desc);
    const latent = this.embedding ? project(this.embedding, raw) : raw.slice();
    const estFullMs = baselineCostMs(desc);
    this.stats.estComputeFullMs += estFullMs;

    if (decision.stage === "cache") {
      this.stats.cacheHits++;
      this.stats.computeSavedMs += Math.max(0, estFullMs - result.elapsedMs);
      const hit = this.cache.find((c) => c.desc.id === decision.nearestId);
      if (hit) hit.hits++;
    } else if (decision.stage === "rom") {
      this.stats.romHits++;
      this.stats.computeSavedMs += Math.max(0, estFullMs - result.elapsedMs);
    } else {
      this.stats.fullSolves++;
      this.baselineIters.push(result.iters);
      if (this.baselineIters.length > 64) this.baselineIters.shift();
      // Insert into cache (LRU on hits).
      this.cache.push({ desc, latent, result, hits: 0 });
      if (this.cache.length > this.cfg.cacheCap) {
        // Evict the least-useful: lowest hits, oldest.
        let worst = 0;
        for (let i = 1; i < this.cache.length; i++) {
          if (this.cache[i].hits < this.cache[worst].hits) worst = i;
        }
        this.cache.splice(worst, 1);
      }
    }
    this.routedIters.push(result.iters);
    if (this.routedIters.length > 64) this.routedIters.shift();
    this.decisionOutcomes.push(result.converged ? 1 : 0);
    if (this.decisionOutcomes.length > 64) this.decisionOutcomes.shift();

    // Speedup: baseline iters / routed iters (clamped).
    const meanBase = mean(this.baselineIters);
    const meanRoute = mean(this.routedIters);
    this.stats.convergenceSpeedup = meanRoute > 0 ? Math.max(0.5, meanBase / meanRoute) : 1;
    this.stats.stabilityScore = mean(this.decisionOutcomes);
  }

  /** Convenience: route, run, record. */
  request(desc: SimDescriptor, runSolve: (d: SimDescriptor, dec: RouteDecision) => SimResult): {
    decision: RouteDecision; result: SimResult;
  } {
    const decision = this.route(desc);
    const result = runSolve(desc, decision);
    this.record(desc, decision, result);
    return { decision, result };
  }

  snapshot(): SimOptStats { return { ...this.stats }; }
  cacheEntries(): readonly CachedSim[] { return this.cache; }

  reset(): void {
    this.cache = []; this.embedding = null; this.rawHistory = [];
    this.baselineIters = []; this.routedIters = []; this.decisionOutcomes = [];
    this.stats = {
      totalRequests: 0, cacheHits: 0, romHits: 0, fullSolves: 0,
      computeSavedMs: 0, estComputeFullMs: 0,
      convergenceSpeedup: 1, stabilityScore: 1,
    };
  }
}

// ---------------- helpers ----------------

function nearest(cache: CachedSim[], q: Float64Array): { idx: number; dist: number } {
  if (cache.length === 0) return { idx: -1, dist: Infinity };
  let best = 0, bestD = l2(cache[0].latent, q);
  for (let i = 1; i < cache.length; i++) {
    const d = l2(cache[i].latent, q);
    if (d < bestD) { bestD = d; best = i; }
  }
  return { idx: best, dist: bestD };
}

function l2(a: Float64Array, b: Float64Array): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

/**
 * Empirical relative-error model: error grows quadratically with the
 * normalized latent distance to the nearest cached solution.
 */
function estimateRelativeError(dist: number, radius: number): number {
  const r = Math.min(1, dist / Math.max(1e-9, radius));
  return 0.005 + 0.06 * r * r;
}

function pickSolver(desc: SimDescriptor, logKappa: number, instability: number): SolverKind {
  const n = desc.ndof;
  // Tiny systems: direct.
  if (n <= 5_000) return "direct";
  // SPD + reasonable conditioning → CG/AMG.
  if (desc.spd && desc.nonlinearity < 0.2) {
    if (n >= 200_000 || logKappa >= 6) return "amg";
    return "cg";
  }
  // Nonsymmetric or mildly nonlinear → BiCGSTAB / GMRES.
  if (logKappa >= 7 || instability >= 0.5 || desc.nonlinearity >= 0.6) return "gmres";
  return "bicgstab";
}

function solverConfidence(desc: SimDescriptor, solver: SolverKind, instability: number): number {
  let c = 0.85;
  // Mismatches lower confidence.
  if (solver === "cg" && !desc.spd) c -= 0.4;
  if (solver === "amg" && desc.nonlinearity > 0.5) c -= 0.2;
  c -= 0.25 * instability;
  return Math.max(0.05, Math.min(1, c));
}

function baselineCostMs(desc: SimDescriptor): number {
  // O(n · log n) toy model, scaled by κ — purely heuristic for UI stats.
  const n = Math.max(1, desc.ndof);
  return 0.0006 * n * Math.log2(n + 1) * (1 + 0.15 * desc.logKappa);
}

function mean(xs: number[]): number {
  if (xs.length === 0) return 0;
  let s = 0; for (const x of xs) s += x;
  return s / xs.length;
}

export { RAW_FEATURE_DIM };
