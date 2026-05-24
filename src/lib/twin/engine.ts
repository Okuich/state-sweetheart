/**
 * Digital Twin Metric Space Layer — engine.
 *
 * For each TelemetryFrame the engine:
 *
 *   1. Packs the frame into a raw feature vector (stations × channels).
 *   2. Updates running mean / variance per dimension (Welford).
 *   3. Projects the standardized vector into a latent space via online
 *      PCA-lite (cyclic-Jacobi eigen on a windowed covariance).
 *   4. Measures distance to the optimal manifold and drift σ.
 *   5. Computes an efficiency surface sample (1 - bounded-distance).
 *   6. Localizes the dominant bottleneck station by per-station deviation.
 *   7. Forecasts the next H latent points via local linear extrapolation
 *      and maps them back to expected distance trajectories.
 *
 * Multi-system synchronization is handled by maintaining a registry of
 * per-twin EngineState instances keyed by twinId — they share no
 * mutable state, so factories / farms / fab lines can be analyzed and
 * compared (sync()) without cross-contamination.
 */

import type {
  TelemetryFrame, TwinSpec, TwinState, TwinEngineOpts,
} from "./types";

interface TwinSlot {
  spec: TwinSpec;
  // Running stats per raw dimension.
  n: number;
  mean: Float64Array;
  m2: Float64Array;
  // Centered window for online PCA (ring buffer of standardized frames).
  window: Float64Array[];
  windowCap: number;
  // Cached eigen-basis (rawDim × latentDim).
  basis: Float64Array[] | null;
  basisAge: number;
  // History of latent trajectories for anomaly + forecast.
  history: { t: number; latent: Float64Array; distance: number }[];
  driftEMA: number;
  // Anomaly topology: latent neighbor table.
  anomalyRadius: number;
}

function zeros(n: number): Float64Array { return new Float64Array(n); }

export class DigitalTwinEngine {
  private opts: Required<TwinEngineOpts>;
  private twins = new Map<string, TwinSlot>();

  constructor(opts: TwinEngineOpts = {}) {
    this.opts = {
      latentDim: opts.latentDim ?? 6,
      warmup: opts.warmup ?? 16,
      forecastSteps: opts.forecastSteps ?? 12,
      driftAlpha: opts.driftAlpha ?? 0.15,
      historyLimit: opts.historyLimit ?? 512,
    };
  }

  register(spec: TwinSpec): void {
    const rawDim = spec.stations.length * spec.channels.length
      + (spec.ambientChannels?.length ?? 0);
    this.twins.set(spec.id, {
      spec, n: 0, mean: zeros(rawDim), m2: zeros(rawDim),
      window: [], windowCap: 64,
      basis: null, basisAge: 0,
      history: [], driftEMA: 0, anomalyRadius: 0,
    });
  }

  reset(): void { this.twins.clear(); }

  has(twinId: string): boolean { return this.twins.has(twinId); }

  list(): TwinSpec[] { return [...this.twins.values()].map((s) => s.spec); }

  // ---------- ingest ----------

  ingest(frame: TelemetryFrame): TwinState {
    const slot = this.twins.get(frame.twinId);
    if (!slot) throw new Error(`Unknown twin: ${frame.twinId}`);

    const raw = packFrame(frame, slot.spec);
    // Welford running stats
    slot.n += 1;
    for (let i = 0; i < raw.length; i++) {
      const d = raw[i] - slot.mean[i];
      slot.mean[i] += d / slot.n;
      slot.m2[i]   += d * (raw[i] - slot.mean[i]);
    }

    // Standardize
    const std = new Float64Array(raw.length);
    for (let i = 0; i < raw.length; i++) {
      const v = slot.n > 1 ? Math.sqrt(slot.m2[i] / (slot.n - 1)) : 1;
      std[i] = (raw[i] - slot.mean[i]) / Math.max(1e-6, v);
    }

    // Update window (ring)
    slot.window.push(std);
    if (slot.window.length > slot.windowCap) slot.window.shift();

    // Recompute basis every 8 frames after warmup.
    if (slot.n >= this.opts.warmup
        && (slot.basis === null || slot.basisAge >= 8)) {
      slot.basis = computeBasis(slot.window, this.opts.latentDim);
      slot.basisAge = 0;
    } else {
      slot.basisAge += 1;
    }

    const latent = slot.basis
      ? project(std, slot.basis)
      : new Float64Array(this.opts.latentDim);

    // Distance to optimal manifold (in raw / physical units).
    const distance = distanceToOptimal(frame, slot.spec);

    // Drift EMA on distance derivative.
    const last = slot.history[slot.history.length - 1];
    const ddist = last ? Math.abs(distance - last.distance) : 0;
    slot.driftEMA = (1 - this.opts.driftAlpha) * slot.driftEMA
                  + this.opts.driftAlpha * ddist;
    const driftSigma = slot.driftEMA / Math.max(1e-6, distance * 0.05 + 0.1);

    // Efficiency surface sample.
    const efficiency = clamp01(1 - distance / Math.max(1, 1 + distance));

    // Bottleneck localization: per-station deviation from optimum.
    const { bottleneckStation, stationLoad } = localizeBottleneck(frame, slot.spec);

    // Anomaly topology: novelty vs prior latent neighborhood.
    const anomalyProb = computeAnomaly(latent, slot);

    // Forecast horizon via local linear extrapolation on latent path.
    const { forecast, forecastDistance } = forecastHorizon(
      slot, latent, distance, this.opts.forecastSteps,
    );

    // Energy headroom: heuristic — high efficiency & low drift → low headroom.
    const energyHeadroom = clamp01(0.6 * (1 - efficiency) + 0.4 * Math.min(1, driftSigma / 3));

    // Push to history (clipped).
    slot.history.push({ t: frame.t, latent, distance });
    if (slot.history.length > this.opts.historyLimit) slot.history.shift();
    // Track anomaly radius as 90th-pct of inter-frame latent jumps.
    slot.anomalyRadius = updateAnomalyRadius(slot);

    return {
      twinId: frame.twinId, t: frame.t, latent, distance, driftSigma,
      efficiency, anomalyProb, bottleneckStation, stationLoad,
      forecast, forecastDistance, energyHeadroom,
    };
  }

  /** Multi-system synchronization: pairwise distance in latent space. */
  sync(aId: string, bId: string): { distance: number; aligned: boolean } | null {
    const a = this.twins.get(aId), b = this.twins.get(bId);
    if (!a || !b) return null;
    const la = a.history[a.history.length - 1]?.latent;
    const lb = b.history[b.history.length - 1]?.latent;
    if (!la || !lb) return null;
    const n = Math.min(la.length, lb.length);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const d = la[i] - lb[i];
      s += d * d;
    }
    const distance = Math.sqrt(s);
    return { distance, aligned: distance < 1.5 };
  }

  /** Historical replay — returns previously-ingested latent path. */
  replay(twinId: string): { t: number; latent: Float64Array; distance: number }[] {
    return this.twins.get(twinId)?.history.slice() ?? [];
  }
}

// ----------------------------------------------------------------
// packing / scoring
// ----------------------------------------------------------------

function packFrame(frame: TelemetryFrame, spec: TwinSpec): Float64Array {
  const ambN = spec.ambientChannels?.length ?? 0;
  const out = new Float64Array(spec.stations.length * spec.channels.length + ambN);
  let k = 0;
  for (const st of spec.stations) {
    const row = frame.stations[st] ?? {};
    for (const ch of spec.channels) out[k++] = Number(row[ch] ?? 0);
  }
  if (spec.ambientChannels) {
    for (const ch of spec.ambientChannels) out[k++] = Number(frame.ambient?.[ch] ?? 0);
  }
  return out;
}

function distanceToOptimal(frame: TelemetryFrame, spec: TwinSpec): number {
  if (!spec.optimal) return 0;
  let s = 0, n = 0;
  for (const st of spec.stations) {
    const opt = spec.optimal[st]; if (!opt) continue;
    const row = frame.stations[st] ?? {};
    for (const ch of spec.channels) {
      const o = opt[ch]; if (o == null) continue;
      const d = (Number(row[ch] ?? 0) - o) / Math.max(1, Math.abs(o));
      s += d * d; n++;
    }
  }
  return n > 0 ? Math.sqrt(s / n) : 0;
}

function localizeBottleneck(frame: TelemetryFrame, spec: TwinSpec):
  { bottleneckStation: string | null; stationLoad: Record<string, number> } {
  const load: Record<string, number> = {};
  let worst: string | null = null, worstV = -1;
  for (const st of spec.stations) {
    const opt = spec.optimal?.[st] ?? {};
    const row = frame.stations[st] ?? {};
    let s = 0, n = 0;
    for (const ch of spec.channels) {
      const o = opt[ch];
      if (o == null) continue;
      const d = (Number(row[ch] ?? 0) - o) / Math.max(1, Math.abs(o));
      s += d * d; n++;
    }
    const v = n > 0 ? Math.sqrt(s / n) : 0;
    load[st] = clamp01(v);
    if (v > worstV) { worstV = v; worst = st; }
  }
  // normalize loads to 0..1 by max
  const m = Math.max(1e-6, ...Object.values(load));
  for (const k of Object.keys(load)) load[k] = load[k] / m;
  return { bottleneckStation: worstV > 0.15 ? worst : null, stationLoad: load };
}

function computeAnomaly(latent: Float64Array, slot: TwinSlot): number {
  if (slot.history.length < 8) return 0;
  // Distance to nearest historical latent (k=1 novelty).
  let best = Infinity;
  const start = Math.max(0, slot.history.length - 64);
  for (let i = start; i < slot.history.length; i++) {
    const h = slot.history[i].latent;
    let s = 0;
    const n = Math.min(latent.length, h.length);
    for (let k = 0; k < n; k++) { const d = latent[k] - h[k]; s += d * d; }
    s = Math.sqrt(s);
    if (s < best) best = s;
  }
  const r = Math.max(0.2, slot.anomalyRadius);
  return clamp01(best / (2 * r));
}

function updateAnomalyRadius(slot: TwinSlot): number {
  if (slot.history.length < 8) return 0.5;
  const jumps: number[] = [];
  const start = Math.max(1, slot.history.length - 32);
  for (let i = start; i < slot.history.length; i++) {
    const a = slot.history[i - 1].latent, b = slot.history[i].latent;
    let s = 0; const n = Math.min(a.length, b.length);
    for (let k = 0; k < n; k++) { const d = a[k] - b[k]; s += d * d; }
    jumps.push(Math.sqrt(s));
  }
  jumps.sort((a, b) => a - b);
  return jumps[Math.floor(jumps.length * 0.9)] || 0.5;
}

function forecastHorizon(
  slot: TwinSlot, latent: Float64Array, distance: number, H: number,
): { forecast: Float64Array[]; forecastDistance: number[] } {
  const forecast: Float64Array[] = [];
  const fd: number[] = [];
  // Local linear extrapolation using last 4 history points.
  const tail = slot.history.slice(-4);
  if (tail.length < 2) {
    for (let h = 0; h < H; h++) { forecast.push(latent.slice()); fd.push(distance); }
    return { forecast, forecastDistance: fd };
  }
  const first = tail[0].latent, last = tail[tail.length - 1].latent;
  const dt = Math.max(1, tail.length - 1);
  const n = latent.length;
  const v = new Float64Array(n);
  for (let i = 0; i < n; i++) v[i] = (last[i] - first[i]) / dt;
  // Distance trend
  const dvel = (tail[tail.length - 1].distance - tail[0].distance) / dt;
  for (let h = 1; h <= H; h++) {
    const p = new Float64Array(n);
    for (let i = 0; i < n; i++) p[i] = latent[i] + v[i] * h;
    forecast.push(p);
    fd.push(Math.max(0, distance + dvel * h));
  }
  return { forecast, forecastDistance: fd };
}

// ----------------------------------------------------------------
// online PCA (cyclic Jacobi on windowed covariance)
// ----------------------------------------------------------------

function computeBasis(window: Float64Array[], k: number): Float64Array[] {
  if (window.length === 0) return [];
  const d = window[0].length;
  const cov = new Float64Array(d * d);
  for (const v of window) {
    for (let i = 0; i < d; i++) {
      for (let j = i; j < d; j++) {
        cov[i * d + j] += v[i] * v[j];
      }
    }
  }
  const inv = 1 / Math.max(1, window.length - 1);
  for (let i = 0; i < d; i++) {
    for (let j = i; j < d; j++) {
      cov[i * d + j] *= inv;
      cov[j * d + i] = cov[i * d + j];
    }
  }
  return jacobiTopK(cov, d, k);
}

function jacobiTopK(A: Float64Array, n: number, k: number): Float64Array[] {
  const a = new Float64Array(A);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  const maxSweeps = 24;
  for (let s = 0; s < maxSweeps; s++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        off += apq * apq;
        if (Math.abs(apq) < 1e-10) continue;
        const app = a[p * n + p], aqq = a[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t), si = t * c;
        for (let i = 0; i < n; i++) {
          const aip = a[i * n + p], aiq = a[i * n + q];
          a[i * n + p] = c * aip - si * aiq;
          a[i * n + q] = si * aip + c * aiq;
        }
        for (let j = 0; j < n; j++) {
          const apj = a[p * n + j], aqj = a[q * n + j];
          a[p * n + j] = c * apj - si * aqj;
          a[q * n + j] = si * apj + c * aqj;
        }
        for (let i = 0; i < n; i++) {
          const vip = V[i * n + p], viq = V[i * n + q];
          V[i * n + p] = c * vip - si * viq;
          V[i * n + q] = si * vip + c * viq;
        }
      }
    }
    if (off < 1e-12) break;
  }
  const eig: { v: number; idx: number }[] = [];
  for (let i = 0; i < n; i++) eig.push({ v: a[i * n + i], idx: i });
  eig.sort((x, y) => y.v - x.v);
  const out: Float64Array[] = [];
  for (let r = 0; r < Math.min(k, n); r++) {
    const col = new Float64Array(n);
    const j = eig[r].idx;
    for (let i = 0; i < n; i++) col[i] = V[i * n + j];
    out.push(col);
  }
  return out;
}

function project(v: Float64Array, basis: Float64Array[]): Float64Array {
  const out = new Float64Array(basis.length);
  for (let k = 0; k < basis.length; k++) {
    let s = 0;
    const b = basis[k];
    for (let i = 0; i < v.length; i++) s += v[i] * b[i];
    out[k] = s;
  }
  return out;
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
