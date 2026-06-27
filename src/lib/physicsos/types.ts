// Physics OS — coupled multi-physics co-simulation types.
// Unified state on a shared N×N grid: temperature T, pressure p,
// velocity (u,v), and skin/structural stress σ (von-Mises-ish scalar).

export type Field = Float32Array;

export interface CoupledState {
  N: number;
  T: Field;  // thermal (K, normalized)
  p: Field;  // pressure (Pa, normalized)
  u: Field;  // x-velocity
  v: Field;  // y-velocity
  s: Field;  // structural stress scalar
}

export interface PhysicsOSConfig {
  gridN: number;
  layers: number;          // KAN depth
  splineGrid: number;      // spline control points per edge
  dt: number;
  viscosity: number;
  thermalDiffusivity: number;
  youngsModulus: number;   // E (normalized)
  thermalExpansion: number;// α
  machRef: number;         // hypersonic reference Mach
}

export interface PhysicsOSMetrics {
  fullSolveMs: number;      // staggered baseline (T→flow→struct)
  coupledMs: number;        // unified PIKAN pass
  speedup: number;
  costReductionPct: number;
  coupledL2: number;        // L2 vs staggered (lower = better)
  predImprovementPct: number; // vs decoupled (no-cross-coupling) baseline
  couplingResidual: number; // joint residual across the three PDE residuals
  refineSteps: number;
}

export interface PhysicsOSState {
  config: PhysicsOSConfig;
  state: CoupledState;
  step: number;
  metrics: PhysicsOSMetrics;
}
