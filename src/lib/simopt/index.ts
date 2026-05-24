/**
 * Simulation Optimization — public surface.
 */
export {
  SimulationOptimizationEngine,
  type SimOptConfig,
} from "./engine";
export { packDescriptor, project, fitEmbedding, type EmbeddingModel } from "./embeddings";
export { evalMeshQuality, type MeshQuality } from "./mesh";
export {
  type SimDescriptor, type SimResult, type CachedSim, type RouteDecision,
  type SimOptStats, type SolverKind, type SimDomain, type MeshStats,
} from "./types";
export {
  SimGateEvaluator, DEFAULT_SIM_GATES, finalize as finalizeSimGateReport,
  type SimGateThresholds, type SimGateStatus, type SimGateReport,
} from "./gating";
