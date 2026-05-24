/**
 * Digital Twin Metric Space Layer — types.
 *
 * A twin represents an industrial system (factory, farm, fab line,
 * irrigation network, robotics cell, …) as an evolving point on a
 * physical-state manifold. Each tick the system samples a telemetry
 * frame; the engine projects it into a latent embedding, tracks
 * drift, locates bottlenecks, and forecasts the next horizon.
 */

export type SystemKind =
  | "factory" | "farm" | "fab_line" | "irrigation"
  | "robotics_cell" | "power_grid" | "generic";

export interface TelemetryFrame {
  twinId: string;
  /** wall-clock seconds */
  t: number;
  /** node/station ID → scalar metric map (throughput, temp, kWh, etc.) */
  stations: Record<string, Record<string, number>>;
  /** Optional system-level scalars (ambient temp, shift, …). */
  ambient?: Record<string, number>;
}

export interface TwinSpec {
  id: string;
  name: string;
  kind: SystemKind;
  /** Ordered station IDs — fixes embedding dimension. */
  stations: string[];
  /** Ordered metric channels per station. */
  channels: string[];
  /** Optional ordered ambient channels. */
  ambientChannels?: string[];
  /** Optimal operating point (per station × channel). Used for distance scoring. */
  optimal?: Record<string, Record<string, number>>;
  /** Latent embedding dim (default 6). */
  latentDim?: number;
}

export interface TwinState {
  twinId: string;
  t: number;
  /** Latent coordinates after PCA-style projection. */
  latent: Float64Array;
  /** L2 distance from optimal manifold (0 = on optimum). */
  distance: number;
  /** Rolling drift magnitude (σ multiple). */
  driftSigma: number;
  /** 0..1 — efficiency surface sample at current state. */
  efficiency: number;
  /** 0..1 — anomaly probability from topology novelty. */
  anomalyProb: number;
  /** Station ID localized as the dominant bottleneck (or null). */
  bottleneckStation: string | null;
  /** Per-station bottleneck weights (0..1). */
  stationLoad: Record<string, number>;
  /** Forecast horizon (latent points, length = forecastSteps). */
  forecast: Float64Array[];
  /** Forecast distance trajectory (length = forecastSteps). */
  forecastDistance: number[];
  /** Estimated energy-optimization headroom (0..1). */
  energyHeadroom: number;
}

export interface TwinEngineOpts {
  /** PCA latent dim (default 6). */
  latentDim?: number;
  /** Warm-up frames before forecasts/anomaly become valid (default 16). */
  warmup?: number;
  /** Forecast horizon in steps (default 12). */
  forecastSteps?: number;
  /** EMA drift smoothing (default 0.15). */
  driftAlpha?: number;
  /** Max stored history per twin (default 512). */
  historyLimit?: number;
}
