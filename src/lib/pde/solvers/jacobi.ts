/**
 * Damped Jacobi smoother — used standalone and as a multigrid relaxation.
 */
import { type CSRMatrix, diag, spmv } from "../sparse";

export interface JacobiOptions {
  iterations: number;
  /** Damping factor ω ∈ (0,1]; 2/3 is the standard smoother choice. */
  omega: number;
}

export function jacobi(
  A: CSRMatrix,
  b: Float64Array,
  x: Float64Array,
  opts: JacobiOptions,
): void {
  const d = diag(A);
  const Ax = new Float64Array(A.n);
  const omega = opts.omega;
  for (let it = 0; it < opts.iterations; it++) {
    spmv(A, x, Ax);
    for (let i = 0; i < A.n; i++) {
      if (d[i] !== 0) x[i] += omega * (b[i] - Ax[i]) / d[i];
    }
  }
}
