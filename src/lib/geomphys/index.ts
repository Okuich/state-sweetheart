/** Geometry-Aware Physics Intelligence — public surface. */
export {
  computeTopology, computeCurvature, deformationMetrics, predictStress,
  routeSolver, optimizeShape, analyze, makeIcoSphere, deformMesh,
} from "./engine";
export type {
  TriMesh, DeformationMetrics, CurvatureField, StressPrediction,
  TopologyProfile, SolverDecision, ShapeOptimizationResult, GeometryReport,
  SolverKind,
} from "./types";
