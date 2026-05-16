/**
 * Algebraic-style geometric multigrid V-cycle.
 *
 * Caller provides a hierarchy of operators (A_l) plus restriction (R_l) and
 * prolongation (P_l = Rᵀ) matrices. Pre- and post-smoothing uses damped
 * Jacobi; the coarsest level is solved by a few CG iterations to keep this
 * dependency-light.
 *
 * Use this either as a standalone solver or as a CG preconditioner.
 */
import { type CSRMatrix, axpy, spmv } from "../sparse";
import { jacobi } from "./jacobi";
import { cg } from "./cg";

export interface MGLevel {
  /** Operator at this level. */
  A: CSRMatrix;
  /** Restriction to next-coarser level. `null` on coarsest level. */
  R: CSRMatrix | null;
  /** Prolongation from next-coarser level. `null` on coarsest level. */
  P: CSRMatrix | null;
}

export interface MGOptions {
  preSmooth: number;
  postSmooth: number;
  omega: number;
  /** Max iterations for the coarse-grid solve. */
  coarseIter: number;
}

export const DEFAULT_MG_OPTS: MGOptions = {
  preSmooth: 2,
  postSmooth: 2,
  omega: 2 / 3,
  coarseIter: 50,
};

/** One V-cycle on `level`. `x` is updated in place to approximate A·x = b. */
export function vcycle(
  levels: MGLevel[],
  level: number,
  b: Float64Array,
  x: Float64Array,
  opts: MGOptions = DEFAULT_MG_OPTS,
): void {
  const { A, R, P } = levels[level];
  if (R === null || P === null) {
    cg(A, b, x, { maxIter: opts.coarseIter, tol: 1e-10 });
    return;
  }
  jacobi(A, b, x, { iterations: opts.preSmooth, omega: opts.omega });
  // residual r = b - A·x
  const r = new Float64Array(A.n);
  spmv(A, x, r);
  for (let i = 0; i < A.n; i++) r[i] = b[i] - r[i];
  // coarse rhs = R·r
  const rc = new Float64Array(R.n);
  spmv(R, r, rc);
  const ec = new Float64Array(R.n);
  vcycle(levels, level + 1, rc, ec, opts);
  // correction: x += P·ec
  const e = new Float64Array(A.n);
  spmv(P, ec, e);
  axpy(1, e, x);
  jacobi(A, b, x, { iterations: opts.postSmooth, omega: opts.omega });
}

/** Build a CG preconditioner that runs a single V-cycle of the hierarchy. */
export function makeMGPreconditioner(
  levels: MGLevel[],
  opts: MGOptions = DEFAULT_MG_OPTS,
) {
  return (r: Float64Array, z: Float64Array) => {
    z.fill(0);
    vcycle(levels, 0, r, z, opts);
  };
}
