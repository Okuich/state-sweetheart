// Multi-start calibration
// ─────────────────────────────────────────────────────────────
// Wraps the single-start Levenberg-Marquardt fitter with a deterministic
// multi-start strategy. For each "material seed" we draw N_STARTS
// quasi-random initial guesses inside the parameter bounds (Halton
// sequence — no PRNG state needed, deterministic per seed), run a short
// fit from each, then return the best fit by RSS plus the full set of
// candidates so the caller can show consensus / variance.
//
// This is robust against the local-minima failure mode of vanilla LM
// when the initial guess is far from the truth and the data is noisy
// (e.g. wrong basin around k or A).

import {
  type FitResult,
  type MeasurementSample,
  type ModelParams,
  calibrate,
} from "./calibration";

const KEYS: (keyof ModelParams)[] = ["k", "c", "m", "A"];
const BOUNDS: Record<keyof ModelParams, [number, number]> = {
  k: [1, 200],
  c: [0.01, 10],
  m: [0.05, 5],
  A: [0.1, 3],
};

export interface MultiStartOptions {
  /** Number of independent starts. */
  starts?: number;
  /** Inner-LM iterations per start (kept short — convergence happens). */
  iterPerStart?: number;
  /** Final polishing iterations on the best fit. */
  polishIter?: number;
  /** Material seed — different seeds explore different starts. */
  seed?: number;
  /** Stop early if RSS drops below this. */
  rssTarget?: number;
}

export interface MultiStartCandidate {
  start: ModelParams;
  fit: FitResult;
  /** Index of this start within the run. */
  index: number;
  /** True iff this candidate became the winner. */
  best: boolean;
}

export interface MultiStartResult {
  /** Best fit across all starts (after final polish). */
  best: FitResult;
  /** Best ModelParams alias for convenience. */
  params: ModelParams;
  /** All starts with their per-start fit (sorted by RSS, ascending). */
  candidates: MultiStartCandidate[];
  /** Per-parameter mean and stddev across the top-K candidates. */
  consensus: ParameterConsensus;
  /** Total LM iterations summed across all starts (for reporting). */
  totalIters: number;
  /** RSS of the worst candidate kept (sanity / dispersion). */
  worstRss: number;
  /** True iff at least 2 candidates landed in the same basin (within tol). */
  basinAgreement: boolean;
}

export interface ParameterConsensus {
  mean: ModelParams;
  std: ModelParams;
  /** Number of candidates included in the consensus (top-K). */
  n: number;
}

/** Deterministic 1D Halton sequence on base b for index i (skip i=0). */
function halton(i: number, b: number): number {
  let f = 1, r = 0, n = i;
  while (n > 0) {
    f /= b;
    r += f * (n % b);
    n = Math.floor(n / b);
  }
  return r; // ∈ [0,1)
}

/** Generate quasi-random initial parameter draws. */
export function generateStarts(count: number, seed: number): ModelParams[] {
  // Different bases per axis — classic Halton choice.
  const bases = [2, 3, 5, 7];
  const offset = Math.max(1, (seed | 0) >>> 0) % 997; // skip-ahead per seed
  const out: ModelParams[] = [];
  for (let i = 0; i < count; i++) {
    const idx = i + 1 + offset; // skip the degenerate i=0 zero
    const u = bases.map((b) => halton(idx, b));
    const p: ModelParams = {
      k: lerp(BOUNDS.k[0], BOUNDS.k[1], u[0]),
      c: lerp(BOUNDS.c[0], BOUNDS.c[1], u[1]),
      m: lerp(BOUNDS.m[0], BOUNDS.m[1], u[2]),
      A: lerp(BOUNDS.A[0], BOUNDS.A[1], u[3]),
    };
    out.push(p);
  }
  return out;
}

function lerp(a: number, b: number, t: number) { return a + (b - a) * t; }

/** Mean & stddev across a candidate set for each parameter. */
function consensus(cands: MultiStartCandidate[], topK: number): ParameterConsensus {
  const k = Math.min(topK, cands.length);
  const subset = cands.slice(0, k).map((c) => c.fit.params);
  const mean = {} as ModelParams;
  const std = {} as ModelParams;
  for (const key of KEYS) {
    let sum = 0;
    for (const p of subset) sum += p[key];
    mean[key] = sum / k;
    let acc = 0;
    for (const p of subset) acc += (p[key] - mean[key]) ** 2;
    std[key] = Math.sqrt(acc / Math.max(1, k - 1));
  }
  return { mean, std, n: k };
}

/** Two parameter sets land in the same basin if all relative diffs < tol. */
function inSameBasin(a: ModelParams, b: ModelParams, tol = 0.05): boolean {
  for (const key of KEYS) {
    const denom = Math.max(Math.abs(a[key]), Math.abs(b[key]), 1e-9);
    if (Math.abs(a[key] - b[key]) / denom > tol) return false;
  }
  return true;
}

export function calibrateMultiStart(
  data: MeasurementSample[],
  opts: MultiStartOptions = {},
): MultiStartResult {
  const starts = Math.max(1, Math.floor(opts.starts ?? 8));
  const iterPerStart = Math.max(2, Math.floor(opts.iterPerStart ?? 12));
  const polishIter = Math.max(0, Math.floor(opts.polishIter ?? 30));
  const seed = opts.seed ?? 0;
  const rssTarget = opts.rssTarget ?? -Infinity;

  const startPoints = generateStarts(starts, seed);
  const candidates: MultiStartCandidate[] = [];
  let totalIters = 0;

  for (let i = 0; i < startPoints.length; i++) {
    const fit = calibrate(startPoints[i], data, iterPerStart);
    totalIters += fit.iters;
    candidates.push({ start: startPoints[i], fit, index: i, best: false });
    if (fit.rss <= rssTarget) break;
  }

  candidates.sort((a, b) => a.fit.rss - b.fit.rss);

  // Polish the winner with extra iterations from its converged point.
  const winnerSeed = candidates[0].fit.params;
  const polished = polishIter > 0 ? calibrate(winnerSeed, data, polishIter) : candidates[0].fit;
  totalIters += polished.iters;
  candidates[0] = { ...candidates[0], fit: polished, best: true };
  candidates.sort((a, b) => a.fit.rss - b.fit.rss);

  const cons = consensus(candidates, Math.min(3, candidates.length));
  const basinAgreement =
    candidates.length >= 2 && inSameBasin(candidates[0].fit.params, candidates[1].fit.params);

  return {
    best: candidates[0].fit,
    params: candidates[0].fit.params,
    candidates,
    consensus: cons,
    totalIters,
    worstRss: candidates[candidates.length - 1].fit.rss,
    basinAgreement,
  };
}
