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
export {
  assembleThermalStiffness, solveThermal,
  type ThermalProblem, type ThermalSolution, type KappaTensor,
} from "./thermal";
export {
  solveElectrostatic, traceFieldLine,
  type ElectrostaticProblem, type ElectrostaticSolution,
  type FieldLineOptions, type SampleE,
} from "./electrostatic";
export {
  differentiateThermal,
  targetTemperatureLoss,
  fluxMagnitudeLoss,
  inverseDesignKappa,
  inverseDesignThermal,
  inverseDesignSource,
  inverseDesignLoads,
  type DifferentiableThermalProblem,
  type ThermalSensitivities,
  type ThermalGradients,
  type InverseDesignOptions,
  type InverseDesignResult,
  type InverseDesignTargets,
  type InverseDesignAllOptions,
  type InverseDesignAllResult,
} from "./differentiable";
export {
  differentiateThermalTensor,
  initThermalAdjointGPU,
  computeKappaGradGPU,
  inverseDesignKappaGPU,
  isGPUTensor,
  toFloat32,
  toFloat64,
  type ThermalTensor,
  type GPUTensor,
  type DifferentiableThermalProblemTensor,
  type ThermalGradientTensors,
  type ThermalAdjointBackend,
  type ThermalAdjointGPUContext,
  type InverseDesignKappaGPUOptions,
  type InverseDesignKappaGPUResult,
} from "./differentiable.gpu";
export {
  solvePotentialFlow, makeVelocitySampler,
  type PotentialFlowProblem, type PotentialFlowSolution,
} from "./potentialFlow";
export {
  buildPODBasis, buildReducedModel, solveReduced, projectToReduced, liftToFull,
  type PODBasis, type PODOptions, type ReducedModel,
} from "./rom";
