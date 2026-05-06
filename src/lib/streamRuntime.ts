// Streaming Physics Runtime
// ─────────────────────────────────────────────────────────────
// Continuous, low-latency simulation synchronized with a live
// telemetry stream. Keeps a rolling window of (sensor, predict)
// pairs and exposes lag / drift / prediction-error signals.
//
// Pure TS — synthetic sensors driven by a deterministic generator
// so the panel is self-contained and reproducible.

export interface StreamSample {
  t: number;          // ms since stream start
  sensor: number;     // measured value (with noise)
  truth: number;      // ground-truth (for offline error report)
  predict: number;    // model forecast for this t
  fused: number;      // assimilated estimate (Kalman-lite)
  residual: number;   // sensor - predict
}

export interface StreamConfig {
  windowMs: number;       // rolling horizon
  tickMs: number;         // ingest cadence
  sensorNoise: number;    // σ
  assimGain: number;      // 0..1, Kalman-lite gain on residual
  forecastMs: number;     // how far ahead to extrapolate
  driftRate: number;      // truth slow drift
  signalHz: number;       // base oscillation
}

export const DEFAULT_STREAM_CFG: StreamConfig = {
  windowMs: 6000,
  tickMs: 50,
  sensorNoise: 0.08,
  assimGain: 0.35,
  forecastMs: 400,
  driftRate: 0.05,
  signalHz: 0.6,
};

// Deterministic noise so traces look natural but reproducible.
function hashNoise(t: number, salt: number) {
  const x = Math.sin(t * 12.9898 + salt * 78.233) * 43758.5453;
  return (x - Math.floor(x)) - 0.5;
}

export interface StreamState {
  cfg: StreamConfig;
  t0: number;
  samples: StreamSample[];
  lastFused: number;
  lastVel: number;
  // online error stats
  mae: number;
  rmse: number;
  count: number;
  ingestLagMs: number;
}

export function createStream(cfg: StreamConfig = DEFAULT_STREAM_CFG): StreamState {
  return {
    cfg,
    t0: performance.now(),
    samples: [],
    lastFused: 0,
    lastVel: 0,
    mae: 0,
    rmse: 0,
    count: 0,
    ingestLagMs: 0,
  };
}

function truthAt(t: number, cfg: StreamConfig) {
  const tt = t / 1000;
  return Math.sin(2 * Math.PI * cfg.signalHz * tt)
       + 0.3 * Math.sin(2 * Math.PI * cfg.signalHz * 2.7 * tt + 1.3)
       + cfg.driftRate * tt * 0.1;
}

export function tickStream(s: StreamState, nowMs: number): StreamSample {
  const cfg = s.cfg;
  const t = nowMs - s.t0;
  const truth = truthAt(t, cfg);
  const sensor = truth + cfg.sensorNoise * 2 * hashNoise(t, 1);

  // Predict: linear extrapolation from last fused state.
  const dt = cfg.tickMs / 1000;
  const predict = s.lastFused + s.lastVel * dt;
  const residual = sensor - predict;
  const fused = predict + cfg.assimGain * residual;
  const newVel = (fused - s.lastFused) / Math.max(dt, 1e-3);

  s.lastFused = fused;
  s.lastVel = newVel;

  const sample: StreamSample = { t, sensor, truth, predict, fused, residual };
  s.samples.push(sample);

  // Trim window
  const cutoff = t - cfg.windowMs;
  while (s.samples.length && s.samples[0].t < cutoff) s.samples.shift();

  // Online error vs truth
  const err = Math.abs(fused - truth);
  s.count++;
  s.mae += (err - s.mae) / s.count;
  s.rmse = Math.sqrt(s.rmse * s.rmse + (err * err - s.rmse * s.rmse) / s.count);

  // Synthetic ingest lag (jittered, tied to tick)
  s.ingestLagMs = 8 + 6 * Math.abs(hashNoise(t, 9));

  return sample;
}

export function forecast(s: StreamState, horizonMs: number): { t: number; v: number }[] {
  const dt = s.cfg.tickMs / 1000;
  const steps = Math.max(1, Math.round(horizonMs / s.cfg.tickMs));
  const out: { t: number; v: number }[] = [];
  let v = s.lastFused;
  let vel = s.lastVel;
  const last = s.samples[s.samples.length - 1]?.t ?? 0;
  for (let i = 1; i <= steps; i++) {
    // mild damping toward zero so forecast doesn't run away
    vel = vel * 0.995;
    v = v + vel * dt;
    out.push({ t: last + i * s.cfg.tickMs, v });
  }
  return out;
}
