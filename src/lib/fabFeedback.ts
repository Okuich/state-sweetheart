// Fabrication Feedback Calibration System
// ─────────────────────────────────────────────────────────────
// Closes the loop between Physics OS predictions and measured
// fabrication outcomes. Maintains a rolling buffer of (predicted,
// measured) tuples across four channels:
//   • dimensional   (µm)
//   • thermal       (°C)
//   • tolerance     (% of spec)
//   • surface QA    (defect index 0..1)
//
// On each batch: residual analysis → bias/scale calibration via
// online ridge least-squares → updates per-channel correction
// parameters → propagates posterior uncertainty (variance shrinks
// with N).

export type Channel = "dimensional" | "thermal" | "tolerance" | "surface";

export type Observation = {
  id: string;
  channel: Channel;
  predicted: number;
  measured: number;
  partId: string;
  ts: number;
};

// affine correction:  measured ≈ scale * predicted + bias
export type ChannelModel = {
  scale: number;
  bias: number;
  // posterior covariance approximated by inverse fisher info
  varScale: number;
  varBias: number;
  n: number;       // # observations seen
  rmse: number;    // residual RMSE (post-correction)
  bias0: number;   // raw mean residual (pre-correction)
};

export type CalibrationState = Record<Channel, ChannelModel>;

export const CHANNELS: Channel[] = ["dimensional", "thermal", "tolerance", "surface"];

export const CHANNEL_UNITS: Record<Channel, string> = {
  dimensional: "µm",
  thermal:     "°C",
  tolerance:   "%spec",
  surface:     "idx",
};

export function initState(): CalibrationState {
  const mk = (): ChannelModel => ({
    scale: 1, bias: 0,
    varScale: 1, varBias: 1,
    n: 0, rmse: 0, bias0: 0,
  });
  return {
    dimensional: mk(),
    thermal:     mk(),
    tolerance:   mk(),
    surface:     mk(),
  };
}

// ─── deterministic PRNG ─────────────────────────────────────
function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rng: () => number, mu = 0, sigma = 1) {
  const u = Math.max(1e-9, rng());
  const v = rng();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// Ground-truth (hidden) per-channel bias + scale used by the synthetic
// fab telemetry generator. The calibrator must recover something close.
const TRUTH: Record<Channel, { scale: number; bias: number; sigma: number }> = {
  dimensional: { scale: 1.04,  bias: 12,   sigma: 8   },
  thermal:     { scale: 0.96,  bias: -3.5, sigma: 1.2 },
  tolerance:   { scale: 1.10,  bias: 4,    sigma: 2.5 },
  surface:     { scale: 1.00,  bias: 0.05, sigma: 0.04 },
};

const PRED_RANGES: Record<Channel, [number, number]> = {
  dimensional: [50,  600],
  thermal:     [40,  220],
  tolerance:   [10,   90],
  surface:     [0.0,  0.6],
};

let _seq = 0;
const uid = (p: string) => `${p}_${Date.now().toString(36)}_${(_seq++).toString(36)}`;

export function generateBatch(n: number, seed = 1): Observation[] {
  const rng = mulberry32(seed);
  const out: Observation[] = [];
  for (let i = 0; i < n; i++) {
    const ch = CHANNELS[Math.floor(rng() * CHANNELS.length)];
    const [lo, hi] = PRED_RANGES[ch];
    const predicted = lo + rng() * (hi - lo);
    const t = TRUTH[ch];
    const measured = t.scale * predicted + t.bias + gauss(rng, 0, t.sigma);
    out.push({
      id: uid("obs"),
      channel: ch,
      predicted,
      measured,
      partId: `P-${(1000 + Math.floor(rng() * 9000)).toString()}`,
      ts: Date.now() - Math.floor(rng() * 86400_000),
    });
  }
  return out;
}

// ─── residual analysis ──────────────────────────────────────
export type ResidualStats = {
  channel: Channel;
  n: number;
  meanRaw: number;        // mean(measured − predicted)
  rmseRaw: number;
  meanCorrected: number;  // mean(measured − (scale*pred+bias))
  rmseCorrected: number;
  reduction: number;      // (rmseRaw − rmseCorrected)/rmseRaw, [0..1]
};

export function residuals(obs: Observation[], m: ChannelModel, ch: Channel): ResidualStats {
  const sub = obs.filter((o) => o.channel === ch);
  const n = sub.length;
  if (n === 0) {
    return { channel: ch, n: 0, meanRaw: 0, rmseRaw: 0, meanCorrected: 0, rmseCorrected: 0, reduction: 0 };
  }
  let sR = 0, ssR = 0, sC = 0, ssC = 0;
  for (const o of sub) {
    const r = o.measured - o.predicted;
    const c = o.measured - (m.scale * o.predicted + m.bias);
    sR += r; ssR += r * r;
    sC += c; ssC += c * c;
  }
  const rmseRaw = Math.sqrt(ssR / n);
  const rmseCor = Math.sqrt(ssC / n);
  return {
    channel: ch,
    n,
    meanRaw: sR / n,
    rmseRaw,
    meanCorrected: sC / n,
    rmseCorrected: rmseCor,
    reduction: rmseRaw === 0 ? 0 : Math.max(0, (rmseRaw - rmseCor) / rmseRaw),
  };
}

// ─── online ridge least-squares per channel ─────────────────
// Solves [scale,bias] = argmin Σ (measured − scale*pred − bias)^2 + λ‖θ‖²
// Closed form on accumulated sufficient statistics.
type Suff = { Sxx: number; Sx: number; Sy: number; Sxy: number; Syy: number; n: number };
const SUFF: Record<Channel, Suff> = {
  dimensional: empty(), thermal: empty(), tolerance: empty(), surface: empty(),
};
function empty(): Suff { return { Sxx: 0, Sx: 0, Sy: 0, Sxy: 0, Syy: 0, n: 0 }; }

export function resetSuff() {
  for (const c of CHANNELS) SUFF[c] = empty();
}

export function ingestBatch(state: CalibrationState, batch: Observation[], lambda = 1e-3): CalibrationState {
  // accumulate sufficient stats per channel
  for (const o of batch) {
    const s = SUFF[o.channel];
    s.Sxx += o.predicted * o.predicted;
    s.Sx  += o.predicted;
    s.Sy  += o.measured;
    s.Sxy += o.predicted * o.measured;
    s.Syy += o.measured * o.measured;
    s.n   += 1;
  }
  // recompute models
  const next: CalibrationState = { ...state };
  for (const c of CHANNELS) {
    const s = SUFF[c];
    if (s.n < 2) continue;
    // Normal equations: [[Sxx Sx],[Sx n]] [scale,bias]^T = [Sxy, Sy]^T  + λI
    const a11 = s.Sxx + lambda;
    const a12 = s.Sx;
    const a22 = s.n + lambda;
    const det = a11 * a22 - a12 * a12;
    if (Math.abs(det) < 1e-9) continue;
    const scale = ( a22 * s.Sxy - a12 * s.Sy ) / det;
    const bias  = (-a12 * s.Sxy + a11 * s.Sy ) / det;
    // residual variance
    const ssr = s.Syy - scale * s.Sxy - bias * s.Sy;
    const sigma2 = Math.max(1e-9, ssr / Math.max(1, s.n - 2));
    // posterior variance ≈ sigma² · (XᵀX + λI)^-1 diag
    const varScale = sigma2 * ( a22 / det);
    const varBias  = sigma2 * ( a11 / det);
    next[c] = {
      scale, bias,
      varScale: Math.max(0, varScale),
      varBias:  Math.max(0, varBias),
      n: s.n,
      rmse: Math.sqrt(Math.max(0, ssr) / s.n),
      bias0: (s.Sy - s.Sx) / s.n, // mean(measured - predicted)
    };
  }
  return next;
}

// confidence shrinkage signal: σ(scale) shrinks like 1/√N
export function uncertainty(m: ChannelModel) {
  return {
    sdScale: Math.sqrt(m.varScale),
    sdBias:  Math.sqrt(m.varBias),
    confidence: m.n === 0 ? 0 : Math.min(1, m.n / 50), // visual gauge
  };
}

// apply correction: convert raw prediction → calibrated prediction
export function correct(m: ChannelModel, predicted: number) {
  return m.scale * predicted + m.bias;
}
