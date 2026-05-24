// Physics-Informed AI types
export type PhysField = Float32Array; // flattened NxN scalar

export interface PIConfig {
  gridN: number;          // resolution per side
  latentDim: number;      // compressed manifold dimension
  operatorModes: number;  // spectral modes for neural operator
  pdeViscosity: number;
  pdeDt: number;
}

export interface SurrogateMetrics {
  fullSolveMs: number;
  surrogateMs: number;
  speedup: number;             // fullSolveMs / surrogateMs
  costReductionPct: number;    // 1 - surrogate/full
  l2Error: number;             // vs full solve
  predImprovementPct: number;  // vs naive linear extrap baseline
  pdeResidual: number;         // physics-informed loss
  manifoldDim: number;         // effective latent dim used
}

export interface PIState {
  config: PIConfig;
  field: PhysField;
  latent: Float32Array;
  step: number;
  metrics: SurrogateMetrics;
}
