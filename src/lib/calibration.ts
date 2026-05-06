// Real-World Calibration Framework
// ─────────────────────────────────────────────────────────────
// Fit a parametric model (mass-spring-damper) to a measured
// time series. Pure TS — Levenberg-Marquardt-lite over (k, c, m)
// against a synthetic "measured" trace with injected noise.

export interface MeasurementSample { t: number; y: number; }
export interface ModelParams { k: number; c: number; m: number; A: number; }

export const TRUE_PARAMS: ModelParams = { k: 42, c: 0.6, m: 1.0, A: 1.0 };

// Closed-form damped oscillator: x(t)=A·e^(-ζωt)·cos(ωd·t)
export function modelAt(p: ModelParams, t: number): number {
  const omega = Math.sqrt(Math.max(p.k / Math.max(p.m, 1e-6), 1e-9));
  const zeta = p.c / (2 * Math.sqrt(Math.max(p.k * p.m, 1e-9)));
  const wd = omega * Math.sqrt(Math.max(1 - zeta * zeta, 1e-9));
  return p.A * Math.exp(-zeta * omega * t) * Math.cos(wd * t);
}

function noise(t: number, salt: number) {
  const x = Math.sin(t * 91.123 + salt * 17.7) * 4938.21;
  return (x - Math.floor(x)) - 0.5;
}

export function generateMeasurements(N = 120, T = 6, sigma = 0.04, drift = 0.02): MeasurementSample[] {
  const out: MeasurementSample[] = [];
  for (let i = 0; i < N; i++) {
    const t = (i / (N - 1)) * T;
    const y = modelAt(TRUE_PARAMS, t) + sigma * 2 * noise(t, 1) + drift * t * 0.05;
    out.push({ t, y });
  }
  return out;
}

export interface FitResult {
  params: ModelParams;
  history: { iter: number; rss: number; rmse: number }[];
  residuals: number[];
  rmse: number;
  rss: number;
  r2: number;
  iters: number;
}

const KEYS: (keyof ModelParams)[] = ["k", "c", "m", "A"];
const BOUNDS: Record<keyof ModelParams, [number, number]> = {
  k: [1, 200], c: [0.01, 10], m: [0.05, 5], A: [0.1, 3],
};

function clamp(p: ModelParams): ModelParams {
  const out = { ...p };
  for (const k of KEYS) out[k] = Math.max(BOUNDS[k][0], Math.min(BOUNDS[k][1], out[k]));
  return out;
}

function residuals(p: ModelParams, data: MeasurementSample[]): number[] {
  return data.map((s) => s.y - modelAt(p, s.t));
}

function rss(r: number[]) { return r.reduce((s, x) => s + x * x, 0); }

// Numerical Jacobian via central differences
function jacobian(p: ModelParams, data: MeasurementSample[]): number[][] {
  const J: number[][] = data.map(() => [0, 0, 0, 0]);
  for (let j = 0; j < KEYS.length; j++) {
    const k = KEYS[j];
    const span = BOUNDS[k][1] - BOUNDS[k][0];
    const h = Math.max(span * 1e-4, 1e-6);
    const pp = { ...p, [k]: p[k] + h };
    const pm = { ...p, [k]: p[k] - h };
    for (let i = 0; i < data.length; i++) {
      const dyp = -(data[i].y - modelAt(pp, data[i].t));
      const dym = -(data[i].y - modelAt(pm, data[i].t));
      J[i][j] = (dyp - dym) / (2 * h);
    }
  }
  return J;
}

// Solve (JᵀJ + λI) δ = Jᵀr  via Gauss-Jordan on small (4x4) system.
function solveLM(J: number[][], r: number[], lambda: number): number[] {
  const n = 4;
  const JtJ: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const Jtr: number[] = Array(n).fill(0);
  for (let i = 0; i < J.length; i++) {
    for (let a = 0; a < n; a++) {
      Jtr[a] += J[i][a] * r[i];
      for (let b = 0; b < n; b++) JtJ[a][b] += J[i][a] * J[i][b];
    }
  }
  for (let a = 0; a < n; a++) JtJ[a][a] *= (1 + lambda);
  // Gauss-Jordan
  const M: number[][] = JtJ.map((row, i) => [...row, Jtr[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r2 = col + 1; r2 < n; r2++) if (Math.abs(M[r2][col]) > Math.abs(M[piv][col])) piv = r2;
    [M[col], M[piv]] = [M[piv], M[col]];
    const d = M[col][col] || 1e-12;
    for (let c2 = col; c2 <= n; c2++) M[col][c2] /= d;
    for (let r2 = 0; r2 < n; r2++) if (r2 !== col) {
      const f = M[r2][col];
      for (let c2 = col; c2 <= n; c2++) M[r2][c2] -= f * M[col][c2];
    }
  }
  return M.map((row) => row[n]);
}

export function calibrate(
  initial: ModelParams,
  data: MeasurementSample[],
  maxIter = 30,
): FitResult {
  let p = clamp(initial);
  let r = residuals(p, data);
  let curRss = rss(r);
  let lambda = 1e-2;
  const history: FitResult["history"] = [
    { iter: 0, rss: curRss, rmse: Math.sqrt(curRss / data.length) },
  ];

  let iters = 0;
  for (let it = 0; it < maxIter; it++) {
    iters = it + 1;
    const J = jacobian(p, data);
    const delta = solveLM(J, r, lambda);
    const candidate: ModelParams = clamp({
      k: p.k - delta[0], c: p.c - delta[1], m: p.m - delta[2], A: p.A - delta[3],
    });
    const rNew = residuals(candidate, data);
    const rssNew = rss(rNew);
    if (rssNew < curRss) {
      p = candidate; r = rNew; curRss = rssNew;
      lambda = Math.max(lambda * 0.7, 1e-8);
    } else {
      lambda = Math.min(lambda * 2.5, 1e6);
    }
    history.push({ iter: iters, rss: curRss, rmse: Math.sqrt(curRss / data.length) });
    if (history.length > 2 && Math.abs(history[history.length - 2].rss - curRss) < 1e-9) break;
  }

  const ymean = data.reduce((s, d) => s + d.y, 0) / data.length;
  const sst = data.reduce((s, d) => s + (d.y - ymean) ** 2, 0);
  const r2 = 1 - curRss / Math.max(sst, 1e-9);

  return {
    params: p, history, residuals: r,
    rss: curRss, rmse: Math.sqrt(curRss / data.length), r2, iters,
  };
}
