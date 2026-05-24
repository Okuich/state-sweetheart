/**
 * Predictive Maintenance Metric Engine — public surface.
 *
 *   import { PredictiveMaintenanceEngine, CHANNEL_ORDER } from "@/lib/pdm";
 *
 * See src/lib/pdm/engine.ts for the runtime loop:
 *   ingest(reading) → packReading → project → drift / kNN / trajectory → score
 */
export {
  PredictiveMaintenanceEngine,
  type PDMConfig,
} from "./engine";
export {
  packReading, fitEmbedding, project, temporalEmbed,
  type EmbeddingModel,
} from "./embeddings";
export {
  RunningMoments, AdaptiveThreshold,
  sigmaDistance, nearest, trajectoryDistance, kmeans, resetSeed,
} from "./anomaly";
export {
  CHANNEL_ORDER,
  type SensorChannel, type SensorReading, type StateVector,
  type LabeledState, type MaintenanceScore,
} from "./types";
export {
  GateEvaluator, DEFAULT_GATES,
  type GateThresholds, type GateStatus, type GateReport, type GroundTruth,
} from "./gating";
