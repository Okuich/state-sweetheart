/**
 * Poisson equation utilities.
 *
 *   −∇·(κ ∇u) = f      in Ω
 *           u = g_D    on Γ_D
 *      (κ∇u)·n = g_N    on Γ_N
 *
 * `solvePoisson` accepts a pre-assembled stiffness matrix K (from
 * `assembleFEMLaplacian` or `assembleFVMLaplacian`), a source term `f`
 * per vertex, optional Dirichlet pins, and optional Neumann nodal loads,
 * then solves with PCG.
 */
import { applyDirichlet, type CSRMatrix } from "./sparse";
import { cg, type CGOptions, type CGResult } from "./solvers/cg";

export interface DirichletBC {
  /** Vertex index. */
  index: number;
  /** Prescribed value. */
  value: number;
}

export interface PoissonProblem {
  K: CSRMatrix;
  /** Per-vertex lumped mass (for source weighting). Length = n. */
  massLumped: Float64Array;
  /** Per-vertex source density f (not yet multiplied by mass). */
  source?: Float64Array;
  /** Pre-integrated nodal loads (e.g. Neumann flux contributions). */
  loads?: Float64Array;
  dirichlet?: ReadonlyArray<DirichletBC>;
  cg?: CGOptions;
}

export interface PoissonSolution {
  u: Float64Array;
  rhs: Float64Array;
  result: CGResult;
}

export function solvePoisson(problem: PoissonProblem): PoissonSolution {
  const n = problem.K.n;
  const rhs = new Float64Array(n);
  if (problem.source) {
    for (let i = 0; i < n; i++) rhs[i] += problem.source[i] * problem.massLumped[i];
  }
  if (problem.loads) {
    for (let i = 0; i < n; i++) rhs[i] += problem.loads[i];
  }
  // Clone K so applyDirichlet doesn't mutate the caller's matrix.
  const K: CSRMatrix = {
    n: problem.K.n,
    rowPtr: problem.K.rowPtr.slice(),
    colIdx: problem.K.colIdx.slice(),
    values: problem.K.values.slice(),
  };
  if (problem.dirichlet) {
    for (const bc of problem.dirichlet) applyDirichlet(K, rhs, bc.index, bc.value);
  }
  const u = new Float64Array(n);
  if (problem.dirichlet) {
    for (const bc of problem.dirichlet) u[bc.index] = bc.value;
  }
  const result = cg(K, rhs, u, problem.cg);
  return { u, rhs, result };
}
