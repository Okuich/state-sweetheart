/**
 * Preconditioned Conjugate Gradient for symmetric positive-definite systems.
 *
 * Default preconditioner is Jacobi (diagonal scaling). Callers may supply
 * a custom one (e.g. multigrid V-cycle) for harder problems.
 */
import { type CSRMatrix, axpby, axpy, diag, dot, norm2, spmv } from "../sparse";

export interface CGOptions {
  maxIter?: number;
  tol?: number;
  /** Optional preconditioner: applies M⁻¹ · r → z, both length n. */
  preconditioner?: (r: Float64Array, z: Float64Array) => void;
}

export interface CGResult {
  iterations: number;
  residual: number;
  converged: boolean;
}

export function cg(
  A: CSRMatrix,
  b: Float64Array,
  x: Float64Array,
  opts: CGOptions = {},
): CGResult {
  const n = A.n;
  const maxIter = opts.maxIter ?? Math.max(100, 2 * n);
  const tol = opts.tol ?? 1e-8;
  const precond = opts.preconditioner ?? makeJacobiPreconditioner(A);

  const r = new Float64Array(n);
  const z = new Float64Array(n);
  const p = new Float64Array(n);
  const Ap = new Float64Array(n);

  // r = b - A·x
  spmv(A, x, r);
  for (let i = 0; i < n; i++) r[i] = b[i] - r[i];

  const bNorm = Math.max(norm2(b), 1e-30);
  let rNorm = norm2(r);
  if (rNorm / bNorm < tol) {
    return { iterations: 0, residual: rNorm / bNorm, converged: true };
  }

  precond(r, z);
  p.set(z);
  let rz = dot(r, z);

  for (let it = 1; it <= maxIter; it++) {
    spmv(A, p, Ap);
    const pAp = dot(p, Ap);
    if (!Number.isFinite(pAp) || pAp <= 0) {
      return { iterations: it, residual: rNorm / bNorm, converged: false };
    }
    const alpha = rz / pAp;
    axpy(alpha, p, x);
    axpy(-alpha, Ap, r);
    rNorm = norm2(r);
    if (rNorm / bNorm < tol) {
      return { iterations: it, residual: rNorm / bNorm, converged: true };
    }
    precond(r, z);
    const rzNext = dot(r, z);
    const beta = rzNext / rz;
    axpby(1, z, beta, p); // p = z + β·p
    rz = rzNext;
  }
  return { iterations: maxIter, residual: rNorm / bNorm, converged: false };
}

function makeJacobiPreconditioner(A: CSRMatrix) {
  const d = diag(A);
  const dInv = new Float64Array(A.n);
  for (let i = 0; i < A.n; i++) dInv[i] = d[i] !== 0 ? 1 / d[i] : 0;
  return (r: Float64Array, z: Float64Array) => {
    for (let i = 0; i < A.n; i++) z[i] = dInv[i] * r[i];
  };
}
