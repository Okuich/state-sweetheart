/**
 * Simulation Optimization Metric Engine — shared types.
 *
 * A simulation is represented as a fixed-dim feature vector capturing
 * problem geometry, material, BC, solver hints, and discretization.
 * The engine reasons in this metric space to skip redundant work,
 * pick the right solver, and bound the error of a reused result.
 */

export type SolverKind =
  | "cg"          // conjugate gradient (SPD)
  | "bicgstab"    // nonsymmetric Krylov
  | "gmres"       // robust nonsymmetric
  | "amg"         // algebraic multigrid (large SPD)
  | "direct"      // sparse LU (small)
  | "rom";        // reduced-order surrogate

export type SimDomain = "fem" | "cfd" | "thermal" | "structural" | "fluid";

export interface MeshStats {
  /** Number of elements. */
  nElems: number;
  /** Number of nodes. */
  nNodes: number;
  /** Worst aspect ratio (1 = ideal). */
  aspectMax: number;
  /** Mean aspect ratio. */
  aspectMean: number;
  /** Smallest dihedral / interior angle in radians. */
  minAngle: number;
  /** Largest dihedral / interior angle in radians. */
  maxAngle: number;
  /** Min element scaled-Jacobian (1 = ideal, ≤0 = inverted). */
  jacobianMin: number;
  /** Skewness in [0,1]; 0 = perfect. */
  skewness: number;
}

export interface SimDescriptor {
  /** Stable opaque id for caching. */
  id: string;
  domain: SimDomain;
  /** Problem size (DoF). */
  ndof: number;
  /** Sparse matrix density estimate (nnz / n²) in (0,1]. */
  density: number;
  /** Condition-number proxy (log10 κ). */
  logKappa: number;
  /** Symmetric positive definite? */
  spd: boolean;
  /** Nonlinearity index in [0,1]. */
  nonlinearity: number;
  /** Mesh quality summary. */
  mesh: MeshStats;
  /** Material / regime tag (e.g. "steel-elastic", "navier-stokes-Re=200"). */
  regime: string;
  /** Boundary-condition fingerprint vector (small, fixed size). */
  bcFingerprint: number[];
  /** Optional time-step / load-step index for trajectory mapping. */
  step?: number;
  /** Optional wall-clock budget hint (ms). */
  budgetMs?: number;
}

export interface SimResult {
  /** Final residual norm. */
  residual: number;
  /** Iterations taken. */
  iters: number;
  /** Wall-clock ms. */
  elapsedMs: number;
  /** Solver used. */
  solver: SolverKind;
  /** Whether it converged within tolerance. */
  converged: boolean;
  /** Compact solution signature (mean, std, energy, max). */
  signature: Float64Array;
}

export interface CachedSim {
  desc: SimDescriptor;
  /** Latent embedding of the descriptor. */
  latent: Float64Array;
  result: SimResult;
  /** Number of cache hits this entry served. */
  hits: number;
}

export interface RouteDecision {
  solver: SolverKind;
  /** "cache" | "rom" | "full" — pipeline stage that should run. */
  stage: "cache" | "rom" | "full";
  /** Confidence in the decision in [0,1]. */
  confidence: number;
  /** Reason string for UI / logging. */
  reason: string;
  /** Estimated error if a cached / ROM answer is reused (relative). */
  estError?: number;
  /** Nearest cache entry id, if any. */
  nearestId?: string;
  /** Distance to nearest cache entry in latent space. */
  nearestDist?: number;
}

export interface SimOptStats {
  totalRequests: number;
  cacheHits: number;
  romHits: number;
  fullSolves: number;
  computeSavedMs: number;
  estComputeFullMs: number;
  convergenceSpeedup: number; // dimensionless, vs baseline
  stabilityScore: number;     // in [0,1], higher = more stable routing
}
