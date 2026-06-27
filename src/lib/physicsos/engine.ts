// Physics OS — coupled Neural-Operator Co-Simulation Engine.
//
// Wraps a PIKAN that maps the unified 5-channel state at a cell + its
// 4 neighbors → next 5-channel state in a SINGLE pass. This is the
// "doesn't care if it's thermal or velocity" claim: same spline network,
// every channel is just another scalar input.
//
// Ground-truth baseline: staggered solver that runs thermal, then flow,
// then structural sub-steps sequentially (industry-standard co-sim).

import type {
  CoupledState, PhysicsOSConfig, PhysicsOSMetrics, PhysicsOSState,
} from "./types";
import { type KAN, makeKAN, kanForward, kanRefine } from "./pikan";

export function defaultConfig(): PhysicsOSConfig {
  return {
    gridN: 32,
    layers: 3,
    splineGrid: 12,
    dt: 0.04,
    viscosity: 0.015,
    thermalDiffusivity: 0.02,
    youngsModulus: 1.2,
    thermalExpansion: 0.18,
    machRef: 6.0,
  };
}

const TAU = Math.PI * 2;

// ---------- scenario init: hypersonic shockwave hitting a skin panel ----------
export function initShockScenario(N: number): CoupledState {
  const T = new Float32Array(N * N);
  const p = new Float32Array(N * N);
  const u = new Float32Array(N * N);
  const v = new Float32Array(N * N);
  const s = new Float32Array(N * N);
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      const x = i / (N - 1), y = j / (N - 1);
      // Oblique shock front near x≈0.35
      const shock = 1 / (1 + Math.exp((x - 0.35) * 18));
      T[j * N + i] = 0.2 + 0.7 * shock + 0.05 * Math.sin(TAU * 3 * y);
      p[j * N + i] = 0.3 + 0.6 * shock;
      u[j * N + i] = 0.8 * (1 - shock) + 0.1;
      v[j * N + i] = 0.05 * Math.sin(TAU * 2 * y);
      s[j * N + i] = 0.0;
    }
  }
  return { N, T, p, u, v, s };
}

// ---------- staggered "ground-truth" baseline ----------
// Sequential: 1) heat diffusion on T (uses dissipation source from u,v)
//             2) advect (u,v) with pressure gradient + viscous diffusion
//             3) update σ from thermal strain + pressure load
function laplacian(f: Float32Array, N: number, i: number, j: number): number {
  const xm = f[j * N + ((i - 1 + N) % N)];
  const xp = f[j * N + ((i + 1) % N)];
  const ym = f[((j - 1 + N) % N) * N + i];
  const yp = f[((j + 1) % N) * N + i];
  return xm + xp + ym + yp - 4 * f[j * N + i];
}
function grad(f: Float32Array, N: number, i: number, j: number): [number, number] {
  const xm = f[j * N + ((i - 1 + N) % N)];
  const xp = f[j * N + ((i + 1) % N)];
  const ym = f[((j - 1 + N) % N) * N + i];
  const yp = f[((j + 1) % N) * N + i];
  return [(xp - xm) * 0.5, (yp - ym) * 0.5];
}

export function staggeredStep(st: CoupledState, cfg: PhysicsOSConfig): CoupledState {
  const { N } = st;
  const T2 = new Float32Array(N * N);
  const p2 = new Float32Array(N * N);
  const u2 = new Float32Array(N * N);
  const v2 = new Float32Array(N * N);
  const s2 = new Float32Array(N * N);
  const dt = cfg.dt;
  // Sub-step 1: thermal
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    const dissip = cfg.viscosity * (st.u[idx] ** 2 + st.v[idx] ** 2);
    T2[idx] = st.T[idx] + dt * (cfg.thermalDiffusivity * laplacian(st.T, N, i, j) + 0.4 * dissip);
  }
  // Sub-step 2: flow (advection + pressure grad + viscosity)
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    const [gpx, gpy] = grad(st.p, N, i, j);
    u2[idx] = st.u[idx] + dt * (-gpx + cfg.viscosity * laplacian(st.u, N, i, j));
    v2[idx] = st.v[idx] + dt * (-gpy + cfg.viscosity * laplacian(st.v, N, i, j));
    p2[idx] = st.p[idx] + dt * (0.5 * laplacian(st.p, N, i, j) - 0.3 * (st.u[idx] - st.v[idx]));
  }
  // Sub-step 3: structure (thermal strain + pressure load)
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    const thermalStrain = cfg.thermalExpansion * (T2[idx] - 0.3);
    const load = p2[idx];
    s2[idx] = 0.6 * st.s[idx] + dt * cfg.youngsModulus * (Math.abs(thermalStrain) + 0.5 * load);
  }
  return { N, T: T2, p: p2, u: u2, v: v2, s: s2 };
}

// Decoupled baseline: each field evolved INDEPENDENTLY (no cross-coupling).
// Used to measure the prediction improvement from the coupled PIKAN.
export function decoupledStep(st: CoupledState, cfg: PhysicsOSConfig): CoupledState {
  const { N } = st;
  const T2 = new Float32Array(N * N);
  const p2 = new Float32Array(N * N);
  const u2 = new Float32Array(N * N);
  const v2 = new Float32Array(N * N);
  const s2 = new Float32Array(N * N);
  const dt = cfg.dt;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    T2[idx] = st.T[idx] + dt * cfg.thermalDiffusivity * laplacian(st.T, N, i, j);
    u2[idx] = st.u[idx] + dt * cfg.viscosity * laplacian(st.u, N, i, j);
    v2[idx] = st.v[idx] + dt * cfg.viscosity * laplacian(st.v, N, i, j);
    p2[idx] = st.p[idx] + dt * 0.5 * laplacian(st.p, N, i, j);
    s2[idx] = 0.9 * st.s[idx];
  }
  return { N, T: T2, p: p2, u: u2, v: v2, s: s2 };
}

// ---------- PIKAN-based coupled step ----------
// Input vector per cell (15-D): [Tc,pc,uc,vc,sc, Te,pe,ue,ve, Tw,pw,uw,vw, Tn, Ts]
// Output vector (5-D): [T', p', u', v', s']
//
// In practice we use a compact 9-D summary: center 5 channels + 4 neighbor Laplacians.
// This keeps the KAN small (9→H→H→5) while still seeing differential info.

const IN_DIM = 9;
const OUT_DIM = 5;

export function buildCouplingKAN(cfg: PhysicsOSConfig): KAN {
  const hidden = Math.max(6, Math.round(cfg.gridN / 4));
  const dims = [IN_DIM, hidden, hidden, OUT_DIM].slice(0, cfg.layers + 1);
  if (dims[dims.length - 1] !== OUT_DIM) dims.push(OUT_DIM);
  return makeKAN(dims, cfg.splineGrid, 17);
}

function packInput(st: CoupledState, i: number, j: number, out: Float32Array): void {
  const { N } = st;
  const idx = j * N + i;
  out[0] = clamp(st.T[idx]);
  out[1] = clamp(st.p[idx]);
  out[2] = clamp(st.u[idx]);
  out[3] = clamp(st.v[idx]);
  out[4] = clamp(st.s[idx]);
  out[5] = clamp(0.25 * laplacian(st.T, N, i, j));
  out[6] = clamp(0.25 * laplacian(st.p, N, i, j));
  out[7] = clamp(0.25 * laplacian(st.u, N, i, j));
  out[8] = clamp(0.25 * laplacian(st.v, N, i, j));
}
const clamp = (x: number) => x < -1 ? -1 : x > 1 ? 1 : x;

export function coupledStep(st: CoupledState, net: KAN): CoupledState {
  const { N } = st;
  const T2 = new Float32Array(N * N);
  const p2 = new Float32Array(N * N);
  const u2 = new Float32Array(N * N);
  const v2 = new Float32Array(N * N);
  const s2 = new Float32Array(N * N);
  const xin = new Float32Array(IN_DIM);
  const scratch = new Float32Array(net.G);
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    packInput(st, i, j, xin);
    const y = kanForward(net, xin, scratch);
    const idx = j * N + i;
    // Output is bounded by tanh — interpret as delta scaled by current magnitude.
    T2[idx] = st.T[idx] + 0.5 * y[0];
    p2[idx] = st.p[idx] + 0.5 * y[1];
    u2[idx] = st.u[idx] + 0.5 * y[2];
    v2[idx] = st.v[idx] + 0.5 * y[3];
    s2[idx] = Math.max(0, st.s[idx] + 0.5 * y[4]);
  }
  return { N, T: T2, p: p2, u: u2, v: v2, s: s2 };
}

// Online refinement: sample K cells, fit PIKAN to staggered ground-truth delta.
export function refineCouplingKAN(
  st: CoupledState,
  net: KAN,
  cfg: PhysicsOSConfig,
  samples = 32,
): number {
  const truth = staggeredStep(st, cfg);
  const N = st.N;
  const data: { x: Float32Array; y: Float32Array }[] = [];
  let s = 1234;
  const rnd = () => { s = (s * 1664525 + 1013904223) >>> 0; return (s & 0xffffff) / 0xffffff; };
  for (let k = 0; k < samples; k++) {
    const i = Math.floor(rnd() * N);
    const j = Math.floor(rnd() * N);
    const x = new Float32Array(IN_DIM);
    packInput(st, i, j, x);
    const idx = j * N + i;
    // Target = predicted delta (clamped to tanh range so it's learnable).
    const y = new Float32Array(OUT_DIM);
    y[0] = clamp(2 * (truth.T[idx] - st.T[idx]));
    y[1] = clamp(2 * (truth.p[idx] - st.p[idx]));
    y[2] = clamp(2 * (truth.u[idx] - st.u[idx]));
    y[3] = clamp(2 * (truth.v[idx] - st.v[idx]));
    y[4] = clamp(2 * (truth.s[idx] - st.s[idx]));
    data.push({ x, y });
  }
  return kanRefine(net, data, 0.04);
}

// ---------- error & residual ----------
export function fieldL2(a: Float32Array, b: Float32Array): number {
  let s = 0, n = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i]; s += d * d; n += b[i] * b[i];
  }
  return Math.sqrt(s / Math.max(n, 1e-9));
}

export function coupledL2(a: CoupledState, b: CoupledState): number {
  return 0.2 * (
    fieldL2(a.T, b.T) + fieldL2(a.p, b.p) +
    fieldL2(a.u, b.u) + fieldL2(a.v, b.v) +
    fieldL2(a.s, b.s)
  );
}

// Joint coupling residual: how well the three physical couplings (thermal
// dissipation, pressure traction, thermo-mechanical strain) are satisfied.
export function couplingResidual(prev: CoupledState, next: CoupledState, cfg: PhysicsOSConfig): number {
  const N = prev.N;
  let r = 0;
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const idx = j * N + i;
    const expT = prev.T[idx] + cfg.dt * (cfg.thermalDiffusivity * laplacian(prev.T, N, i, j) +
      0.4 * cfg.viscosity * (prev.u[idx] ** 2 + prev.v[idx] ** 2));
    const [gpx, gpy] = grad(prev.p, N, i, j);
    const expU = prev.u[idx] + cfg.dt * (-gpx + cfg.viscosity * laplacian(prev.u, N, i, j));
    const expV = prev.v[idx] + cfg.dt * (-gpy + cfg.viscosity * laplacian(prev.v, N, i, j));
    const strain = cfg.thermalExpansion * (next.T[idx] - 0.3);
    const expS = 0.6 * prev.s[idx] + cfg.dt * cfg.youngsModulus * (Math.abs(strain) + 0.5 * next.p[idx]);
    r += (next.T[idx] - expT) ** 2 + (next.u[idx] - expU) ** 2 +
         (next.v[idx] - expV) ** 2 + (next.s[idx] - expS) ** 2;
  }
  return Math.sqrt(r / (N * N * 4));
}

// ---------- benchmark ----------
const nowMs = () => typeof performance !== "undefined" ? performance.now() : Date.now();

export function benchmark(
  cfg: PhysicsOSConfig,
  net: KAN,
  init: CoupledState,
  runs = 6,
): PhysicsOSMetrics {
  // warm
  staggeredStep(init, cfg);
  coupledStep(init, net);

  let t0 = nowMs();
  let a = init;
  for (let r = 0; r < runs; r++) a = staggeredStep(a, cfg);
  const fullSolveMs = (nowMs() - t0) / runs;

  t0 = nowMs();
  let b = init;
  for (let r = 0; r < runs; r++) b = coupledStep(b, net);
  const coupledMs = (nowMs() - t0) / runs;

  const truth = staggeredStep(init, cfg);
  const coupled = coupledStep(init, net);
  const decoupled = decoupledStep(init, cfg);
  const errC = coupledL2(coupled, truth);
  const errD = coupledL2(decoupled, truth) || 1e-9;
  const predImprovementPct = Math.max(0, 1 - errC / errD) * 100;

  return {
    fullSolveMs,
    coupledMs,
    speedup: fullSolveMs / Math.max(coupledMs, 1e-6),
    costReductionPct: Math.max(0, 1 - coupledMs / Math.max(fullSolveMs, 1e-6)) * 100,
    coupledL2: errC,
    predImprovementPct,
    couplingResidual: couplingResidual(init, coupled, cfg),
    refineSteps: 0,
  };
}

export function initState(cfg: PhysicsOSConfig = defaultConfig()): PhysicsOSState {
  return {
    config: cfg,
    state: initShockScenario(cfg.gridN),
    step: 0,
    metrics: {
      fullSolveMs: 0, coupledMs: 0, speedup: 0, costReductionPct: 0,
      coupledL2: 0, predImprovementPct: 0, couplingResidual: 0, refineSteps: 0,
    },
  };
}
