/**
 * Unified Laplacian Physics Engine — PDE core.
 *
 *   sparse → laplacian → poisson
 *                    ↘ solvers (cg, jacobi, multigrid)
 *
 * Domain-specific engines (thermal, electrostatic, potential flow,
 * structural, harmonic navigation) build on these primitives in later
 * phases. The same CSR layouts are GPU-portable for the planned
 * WebGPU backend.
 */
export {
  type CSRMatrix, buildCSR, spmv, diag, dot, axpy, axpby, norm2, applyDirichlet,
} from "./sparse";
export {
  assembleFEMLaplacian, assembleFVMLaplacian,
  type FEMMeshInput, type AssemblyResult,
} from "./laplacian";
export {
  solvePoisson, type PoissonProblem, type PoissonSolution, type DirichletBC,
} from "./poisson";
export { cg, type CGOptions, type CGResult } from "./solvers/cg";
export { jacobi, type JacobiOptions } from "./solvers/jacobi";
export {
  vcycle, makeMGPreconditioner, DEFAULT_MG_OPTS,
  type MGLevel, type MGOptions,
} from "./solvers/multigrid";
