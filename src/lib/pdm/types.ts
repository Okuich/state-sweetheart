/**
 * Predictive Maintenance Metric Engine — shared types.
 *
 * Sensors map to a fixed-order numeric channel vector. The engine is
 * channel-agnostic at runtime: missing channels are imputed from the
 * asset's running mean so partial telemetry never blows up the metric.
 */

export type SensorChannel =
  | "vibration"
  | "pressure"
  | "temperature"
  | "flow_rate"
  | "current"
  | "acoustic"
  | "hydraulic";

/** Stable channel ordering used by every state vector in the engine. */
export const CHANNEL_ORDER: readonly SensorChannel[] = [
  "vibration",
  "pressure",
  "temperature",
  "flow_rate",
  "current",
  "acoustic",
  "hydraulic",
] as const;

export interface SensorReading {
  /** Asset / machine identifier. */
  assetId: string;
  /** Wall-clock timestamp (ms or monotonic tick — both fine, must increase). */
  t: number;
  /** Partial channel map. Missing channels are imputed. */
  channels: Partial<Record<SensorChannel, number>>;
  /** Optional asset-supplied operating label (e.g. "idle", "load_3kW"). */
  regime?: string;
}

/** Dense, ordered state vector (length = CHANNEL_ORDER.length). */
export type StateVector = Float64Array;

export interface LabeledState {
  v: StateVector;
  /** Optional label: "optimal", "failure:<mode>", "degraded", … */
  label?: string;
  t?: number;
}

export interface MaintenanceScore {
  assetId: string;
  t: number;
  /** 0 = healthy, 1 = imminent failure. */
  riskScore: number;
  /** Anomaly score in σ units (Mahalanobis from running mean). */
  driftSigma: number;
  /** Distance to nearest optimal state (normalized). */
  distToOptimal: number;
  /** Distance to nearest known failure state (normalized). */
  distToFailure: number;
  /** Estimated samples until risk crosses 1.0 (∞ if drift not increasing). */
  failureHorizon: number;
  /** Cluster id (nearest-centroid). */
  clusterId: number;
  /** Maintenance priority bucket. */
  priority: "ok" | "watch" | "schedule" | "urgent";
}
