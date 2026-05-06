import { useEffect, useMemo, useRef, useState } from "react";
import { compileFieldExpr } from "@/lib/exprCompile";

export type ValidationIssue = { field: string; expected: string; got: string };
export type ValidationReport = { ok: boolean; issues: ValidationIssue[]; checkedAt: number };

/**
 * Runtime shape & dtype assertions for PhysicsState tensors.
 * Mirrors what `assert x.shape == (N, D)` / `x.dtype == float32` would do in PyTorch.
 */
function validateState(s: {
  N: number; D: number; dtype: "float32" | "float64";
  x: unknown; v: unknown; m: unknown; f: unknown;
}): ValidationReport {
  const issues: ValidationIssue[] = [];
  const N = s.N, D = s.D;
  const Ctor = s.dtype === "float64" ? Float64Array : Float32Array;
  const dtypeName = Ctor.name;

  const checkVec = (name: string, arr: unknown, len: number) => {
    if (!(arr instanceof Ctor)) {
      issues.push({ field: name, expected: dtypeName, got: (arr as ArrayBufferView | undefined)?.constructor?.name ?? typeof arr });
      return;
    }
    if (arr.length !== len) {
      issues.push({ field: name, expected: `length ${len}`, got: `length ${arr.length}` });
    }
    const stride = Math.max(1, Math.floor(arr.length / 256));
    for (let i = 0; i < arr.length; i += stride) {
      if (!Number.isFinite(arr[i])) {
        issues.push({ field: name, expected: "finite values", got: `${arr[i]} at index ${i}` });
        break;
      }
    }
  };

  if (!Number.isInteger(N) || N <= 0) issues.push({ field: "N", expected: "positive int", got: String(N) });
  if (D !== 2) issues.push({ field: "D", expected: "2", got: String(D) });

  checkVec("x", s.x, N * D);
  checkVec("v", s.v, N * D);
  checkVec("m", s.m, N);
  checkVec("f", s.f, N * D);

  return { ok: issues.length === 0, issues, checkedAt: performance.now() };
}


export type Dtype = "float32" | "float64";
export type Device = "cpu" | "webgpu";

export type SimParams = {
  gravity: number;
  gravityMode: "uniform" | "directional" | "zero";
  gravityAngle: number; // degrees, 0 = +x (right), 90 = +y (down)
  damping: number;
  dragMode: "explicit" | "exponential" | "force";
  attractor: number;
  particleCount: number;
  trail: number;
  paused: boolean;
  dtScale: number;
  stepOnce: number;
  springK: number;
  restLength: number;
  edgesPerNode: number;
  showEdges: boolean;
  pairwiseStrength: number;
  pairwiseRadius: number;
  integrator: "euler" | "semi-euler" | "verlet";
  dtype: Dtype;
  device: Device;
  constraintIters: number;
  field: "none" | "swirl" | "wells" | "ripple" | "custom";
  fieldStrength: number;
  customFieldSrc: string;
  subSteps: number;
  workers: number;
  showPartitions: boolean;
  optimize: boolean;
  objectiveLR: number;
  pairwiseMode: "lj" | "repel" | "attract";
  boundary: "walls" | "wrap" | "periodic";
  restitution: number;
  forceViz: "off" | "vectors" | "heatmap";
  potentialGrad: "analytic" | "finite-diff";
  fieldSampling: "auto" | "clamp" | "wrap" | "none";
  showFieldArrows: boolean;
  debugForces: boolean;
  adaptiveSubSteps: boolean;
  maxSubSteps: number;
  pairwiseAlgo: "grid" | "all-pairs";
  // ── Probabilistic runtime ─────────────────────────────────────────
  // stochastic: inject Gaussian noise into the force step (Langevin),
  //   advect a Monte Carlo ensemble of K position-offset replicas
  //   alongside the main state, and use the ensemble spread as a live
  //   estimate of σ_x(t). Cost is O(N·K) per sub-step.
  // confidenceZ: z-score for the rendered ellipse (1≈68%, 2≈95%).
  // constraintTol: probabilistic edge-stretch tolerance — diagnostics
  //   report P(|edge-rest|/rest < tol) under the Gaussian σ assumption.
  stochastic: boolean;
  noiseSigma: number;
  ensembleK: number;
  confidenceZ: number;
  constraintTol: number;
  showConfidence: boolean;
  // ── Digital Twin synchronization ──────────────────────────────────
  // M synthetic sensors stream Lissajous "ground-truth" positions with
  // Gaussian noise. Each frame the nearest particle is nudged toward
  // its assigned reading (Kalman-lite blend with gain assimGain).
  // anomalyZ: residual / sensorNoise; flags when the sim & telemetry
  // diverge. forecastSteps: ballistic lookahead drawn as a fade trail.
  twinEnabled: boolean;
  twinSensorCount: number;
  twinAssimGain: number;
  twinSensorNoise: number;
  twinAnomalyZ: number;
  twinForecastSteps: number;
  showTwin: boolean;
};

type FloatArr = Float32Array | Float64Array;

type State = {
  N: number;
  D: number;
  dtype: Dtype;
  device: Device;
  x: FloatArr;
  v: FloatArr;
  m: FloatArr;
  f: FloatArr;
  fPrev: FloatArr;
  hue: Float32Array;       // visual-only, not part of physics tensors
  edges: Int32Array;
  edgeRest: FloatArr;
  E: number;
  // Velocity-Verlet needs a valid f(x₀) the first time verletDrift runs
  // (the half-kick is v += ½·(fPrev/m)·dt). If we leave fPrev = 0 the
  // first sub-step silently drops gravity & every other body force from
  // the kick. `verletPrimed` flips true after we seed fPrev = f(x₀).
  verletPrimed: boolean;
  // ── Probabilistic ensemble ────────────────────────────────────────
  // dx/dy hold K Monte Carlo position OFFSETS per particle (relative to
  // the deterministic mean x). Layout: [k * N*2 + i*2 + d]. Velocities
  // for each replica are tracked in dv. K may change at runtime.
  K: number;
  ensX: Float32Array;
  ensV: Float32Array;
};

/** Allocate a typed array matching `dtype`. */
function emptyLike(len: number, dtype: Dtype): FloatArr {
  return dtype === "float64" ? new Float64Array(len) : new Float32Array(len);
}

/** Copy/cast `src` into a fresh array of the requested dtype. */
function castArray(src: FloatArr, dtype: Dtype): FloatArr {
  const Ctor = dtype === "float64" ? Float64Array : Float32Array;
  if (src instanceof Ctor) return new Ctor(src); // copy, same dtype
  const out = new Ctor(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i];
  return out;
}

/**
 * PhysicsState.to(device, dtype)
 *
 * Cast every tensor (x, v, m, f, fPrev, edgeRest) to the target dtype and
 * (logical) device. `f` is *re-initialized* to zeros with the right dtype so
 * we never carry stale forces across a device hop. Returns a new State; the
 * old buffers are left for GC, mirroring PyTorch's `.to()` semantics where
 * the call is a no-op when nothing changes.
 *
 * Devices supported in-browser:
 *   "cpu"    → typed arrays on the JS heap (always available)
 *   "webgpu" → falls back to CPU when navigator.gpu is undefined; we still
 *              record the requested device so the UI can surface it.
 */
function toDevice(s: State, device: Device, dtype: Dtype): State {
  if (s.device === device && s.dtype === dtype) return s;
  return {
    ...s,
    dtype,
    device,
    x: castArray(s.x, dtype),
    v: castArray(s.v, dtype),
    m: castArray(s.m, dtype),
    // f reinitialized to zeros on the new device/dtype — never reuse stale forces
    f: emptyLike(s.N * s.D, dtype),
    fPrev: emptyLike(s.N * s.D, dtype),
    // device/dtype changed → forces zeroed → must re-prime fPrev next frame
    verletPrimed: false,
    edgeRest: castArray(s.edgeRest, dtype),
  };
}

/**
 * PhysicsState.step(dt) — advance positions & velocities using a = f/m.
 *
 * integrator = "euler"  → semi-implicit (symplectic) Euler:
 *      v ← (v + a·dt) · (1 - damping·dt)
 *      x ← x + v·dt
 *
 * integrator = "verlet" → velocity-Verlet (2nd order, energy-stable),
 * split across the force evaluation so the kick really uses (a_old + a_new)/2:
 *
 *      // BEFORE recomputing forces (uses a_old from previous step, in s.fPrev)
 *      v ← v + ½·a_old·dt
 *      x ← x + v·dt                                         [verletDrift]
 *
 *      // recompute forces here → s.f now holds a_new
 *
 *      v ← (v + ½·a_new·dt) · (1 - damping·dt)
 *      s.fPrev ← s.f                                        [verletKick]
 *
 * Walls: elastic-ish reflection with restitution 0.7.
 */
function verletDrift(
  s: State,
  dt: number,
  a: number,
  b: number,
  w = 0,
  h = 0,
  boundary: Boundary = "walls",
  restitution = 0.7,
) {
  // First half-kick using PREVIOUS step's forces (cached in s.fPrev),
  // then drift positions with the half-updated velocity. Wall reflection
  // is applied IMMEDIATELY after the drift so the subsequent force
  // evaluation (springs, pairwise, field) sees in-bounds positions and
  // the velocity is consistent with the new pose.
  //
  // We also flip the normal component of fPrev (the cached a_old) for any
  // particle that just bounced — otherwise the *next* drift would re-apply
  // a half-kick that pushes the particle back through the wall it just
  // reflected off, producing the classic "stuck-to-wall" Verlet artifact.
  for (let i = a; i < b; i++) {
    const invM = 1 / s.m[i];
    const axOld = s.fPrev[i * 2]     * invM;
    const ayOld = s.fPrev[i * 2 + 1] * invM;
    s.v[i * 2]     += 0.5 * axOld * dt;
    s.v[i * 2 + 1] += 0.5 * ayOld * dt;
    s.x[i * 2]     += s.v[i * 2]     * dt;
    s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt;
  }
  if (boundary === "walls" && w > 0 && h > 0) {
    const e = restitution < 0 ? 0 : restitution > 1 ? 1 : restitution;
    for (let i = a; i < b; i++) {
      if (s.x[i * 2] < 0) {
        s.x[i * 2] = 0;
        if (s.v[i * 2] < 0)        s.v[i * 2]     = -s.v[i * 2]     * e;
        if (s.fPrev[i * 2] < 0)    s.fPrev[i * 2] = -s.fPrev[i * 2] * e;
      } else if (s.x[i * 2] > w) {
        s.x[i * 2] = w;
        if (s.v[i * 2] > 0)        s.v[i * 2]     = -s.v[i * 2]     * e;
        if (s.fPrev[i * 2] > 0)    s.fPrev[i * 2] = -s.fPrev[i * 2] * e;
      }
      if (s.x[i * 2 + 1] < 0) {
        s.x[i * 2 + 1] = 0;
        if (s.v[i * 2 + 1] < 0)        s.v[i * 2 + 1]     = -s.v[i * 2 + 1]     * e;
        if (s.fPrev[i * 2 + 1] < 0)    s.fPrev[i * 2 + 1] = -s.fPrev[i * 2 + 1] * e;
      } else if (s.x[i * 2 + 1] > h) {
        s.x[i * 2 + 1] = h;
        if (s.v[i * 2 + 1] > 0)        s.v[i * 2 + 1]     = -s.v[i * 2 + 1]     * e;
        if (s.fPrev[i * 2 + 1] > 0)    s.fPrev[i * 2 + 1] = -s.fPrev[i * 2 + 1] * e;
      }
    }
  } else if (boundary !== "walls" && w > 0 && h > 0) {
    // wrap / periodic: keep positions inside the canvas so the recomputed
    // forces (and minimum-image pairwise) see canonical coords.
    for (let i = a; i < b; i++) {
      let xi = s.x[i * 2], yi = s.x[i * 2 + 1];
      xi = xi - Math.floor(xi / w) * w;
      yi = yi - Math.floor(yi / h) * h;
      s.x[i * 2] = xi;
      s.x[i * 2 + 1] = yi;
    }
  }
}

function verletKick(
  s: State,
  dt: number,
  damping: number,
  a: number,
  b: number,
  dragMode: "explicit" | "exponential" | "force" = "explicit",
) {
  // Second half-kick using the NEW forces just computed for this step,
  // then cache them as fPrev for the next step's drift.
  const decay =
    dragMode === "force"        ? 1 :
    dragMode === "exponential"  ? Math.exp(-damping * dt) :
                                  Math.max(0, 1 - damping * dt);
  for (let i = a; i < b; i++) {
    const invM = 1 / s.m[i];
    const axNew = s.f[i * 2]     * invM;
    const ayNew = s.f[i * 2 + 1] * invM;
    s.v[i * 2]     = (s.v[i * 2]     + 0.5 * axNew * dt) * decay;
    s.v[i * 2 + 1] = (s.v[i * 2 + 1] + 0.5 * ayNew * dt) * decay;
    s.fPrev[i * 2]     = s.f[i * 2];
    s.fPrev[i * 2 + 1] = s.f[i * 2 + 1];
  }
}
/**
 * Differentiable scalar potential fields Φ(x, y) and helpers.
 *
 * In PyTorch you'd do:
 *     potential = field_fn(state.x).sum()
 *     forces    = -autograd.grad(potential, state.x)[0]
 *
 * We support TWO gradient backends with identical contracts:
 *   • "analytic"     — closed-form ∂Φ/∂x, ∂Φ/∂y. ~2× faster, no h-tuning,
 *                      bit-stable (no catastrophic cancellation), preferred.
 *   • "finite-diff"  — central differences. Plug-and-play fallback for any
 *                      Φ that doesn't ship an analytic gradient.
 */
type FieldName = "none" | "swirl" | "wells" | "ripple" | "custom";
export type PotentialGrad = "analytic" | "finite-diff";
export type FieldSampling = "auto" | "clamp" | "wrap" | "none";

/**
 * sampleCoords — map a particle's world position into the coordinate the
 * field is sampled at. This is what makes Φ behave correctly at canvas
 * edges, especially for "wrap"/"periodic" boundaries where a particle
 * that just teleported across the seam would otherwise see a wildly
 * different ∇Φ from one frame to the next.
 *
 *   "auto"  — follow the simulation boundary mode (the right answer 95%
 *             of the time): walls→clamp, wrap/periodic→wrap.
 *   "clamp" — clip x∈[0,w], y∈[0,h]. Useful for Gaussian wells/ripples
 *             where Φ has a meaningful "outside" but you don't want the
 *             field to run away if a particle briefly leaks past a wall.
 *   "wrap"  — modulo into [0,w)×[0,h), i.e. treat Φ as a torus. Required
 *             for periodic boundaries to keep ∇Φ continuous across the seam.
 *   "none"  — pass coords through untouched (legacy behavior; lets Φ be
 *             evaluated arbitrarily far outside the canvas).
 *
 * Returns [x, y] in the same units as the input.
 */
function resolveSampling(mode: FieldSampling, b: Boundary): "clamp" | "wrap" | "none" {
  if (mode !== "auto") return mode;
  return b === "walls" ? "clamp" : "wrap";
}

function sampleCoords(mode: "clamp" | "wrap" | "none", x: number, y: number, w: number, h: number): [number, number] {
  if (mode === "none") return [x, y];
  if (mode === "clamp") {
    return [
      x < 0 ? 0 : x > w ? w : x,
      y < 0 ? 0 : y > h ? h : y,
    ];
  }
  // wrap: positive-modulo so negative coords land back inside the box
  const xm = ((x % w) + w) % w;
  const ym = ((y % h) + h) % h;
  return [xm, ym];
}


// Pluggable, user-defined Φ. The compiler in src/lib/exprCompile.ts produces
// a pure-JS closure with no globals; we accept it here so the canvas never
// touches `eval` / `new Function` directly.
import type { FieldEnv } from "@/lib/exprCompile";
export type CustomFieldFn = (env: FieldEnv) => number;

function fieldPotential(name: FieldName, x: number, y: number, w: number, h: number, custom?: CustomFieldFn | null, t = 0): number {
  const cx = w * 0.5, cy = h * 0.5;
  const s = Math.max(w, h);
  const nx = (x - cx) / s;
  const ny = (y - cy) / s;
  switch (name) {
    case "swirl":
      return 0.5 * (nx * nx + ny * ny) + 0.25 * Math.sin(6 * Math.atan2(ny, nx));
    case "wells": {
      const d1 = (nx + 0.18) ** 2 + (ny - 0.0) ** 2;
      const d2 = (nx - 0.18) ** 2 + (ny + 0.0) ** 2;
      return -Math.exp(-d1 * 18) - Math.exp(-d2 * 18);
    }
    case "ripple": {
      const r = Math.sqrt(nx * nx + ny * ny);
      return Math.cos(r * 28) * Math.exp(-r * 2.5) * 0.4;
    }
    case "custom": {
      if (!custom) return 0;
      const r = Math.sqrt(nx * nx + ny * ny);
      const theta = Math.atan2(ny, nx);
      try {
        const v = custom({ nx, ny, x, y, w, h, r, theta, t });
        // Defensive: any NaN/Inf in user code → zero force this frame, no halt.
        return Number.isFinite(v) ? v : 0;
      } catch {
        return 0;
      }
    }
    default:
      return 0;
  }
}

/**
 * fieldGradAnalytic — closed-form ∇Φ in WORLD coordinates (so it matches the
 * finite-difference backend exactly). Each case is the pen-and-paper derivative
 * of the matching branch in fieldPotential, with the chain-rule factor (1/s)
 * for the (x,y) → (nx,ny) substitution applied once at the end.
 *
 * Returns [dΦ/dx, dΦ/dy]; ALL fields here have closed forms, so the
 * "analytic" backend never falls back. Adding a new Φ without an analytic
 * gradient: return null and the caller will use central differences.
 */
function fieldGradAnalytic(
  name: FieldName, x: number, y: number, w: number, h: number,
): [number, number] | null {
  const cx = w * 0.5, cy = h * 0.5;
  const s = Math.max(w, h);
  const inv = 1 / s;
  const nx = (x - cx) * inv;
  const ny = (y - cy) * inv;
  switch (name) {
    case "swirl": {
      // Φ = ½(nx²+ny²) + ¼ sin(6θ),  θ = atan2(ny, nx)
      // ∂Φ/∂nx = nx + ¼·cos(6θ)·6·(-ny/r²)
      // ∂Φ/∂ny = ny + ¼·cos(6θ)·6·( nx/r²)
      const r2 = nx * nx + ny * ny + 1e-12;
      const c6 = Math.cos(6 * Math.atan2(ny, nx));
      const k = 1.5 * c6 / r2; // = 0.25 * 6 * cos / r²
      const dnx = nx + k * (-ny);
      const dny = ny + k * ( nx);
      return [dnx * inv, dny * inv];
    }
    case "wells": {
      // Φ = -exp(-18·d1) - exp(-18·d2),  d1,2 = (nx±0.18)² + ny²
      // ∂/∂nx of -exp(-18 d) = exp(-18 d) · 18 · ∂d/∂nx
      const ax = nx + 0.18, bx = nx - 0.18;
      const d1 = ax * ax + ny * ny;
      const d2 = bx * bx + ny * ny;
      const e1 = Math.exp(-d1 * 18);
      const e2 = Math.exp(-d2 * 18);
      const dnx = 36 * (e1 * ax + e2 * bx);
      const dny = 36 * (e1 * ny + e2 * ny);
      return [dnx * inv, dny * inv];
    }
    case "ripple": {
      // Φ = 0.4 · cos(28r) · exp(-2.5r),  r = √(nx²+ny²)
      // dΦ/dr = 0.4 · (-28 sin(28r) - 2.5 cos(28r)) · exp(-2.5r)
      // ∂Φ/∂nx = dΦ/dr · nx/r,  ∂Φ/∂ny = dΦ/dr · ny/r
      const r = Math.sqrt(nx * nx + ny * ny) + 1e-12;
      const e = Math.exp(-2.5 * r);
      const dPhi_dr = 0.4 * (-28 * Math.sin(28 * r) - 2.5 * Math.cos(28 * r)) * e;
      const dnx = dPhi_dr * nx / r;
      const dny = dPhi_dr * ny / r;
      return [dnx * inv, dny * inv];
    }
    case "custom":
      // No closed form for user expressions → tell caller to use FD.
      return null;
    default:
      return [0, 0];
  }
}

/**
 * compute_potential_forces — adds  -∇Φ · strength  to state.f for every node.
 * Uses analytic gradients by default (faster, more stable); falls back to
 * central finite differences when requested or when an analytic gradient
 * is not registered for the active field.
 */
function computePotentialForces(s: State, name: FieldName, strength: number, w: number, h: number, mode: PotentialGrad = "analytic", sampling: FieldSampling = "auto", boundary: Boundary = "walls", custom: CustomFieldFn | null = null, t = 0) {
  computePotentialForces_range(s, name, strength, w, h, 0, s.N, mode, sampling, boundary, custom, t);
}

function computePotentialForces_range(s: State, name: FieldName, strength: number, w: number, h: number, a: number, b: number, mode: PotentialGrad = "analytic", sampling: FieldSampling = "auto", boundary: Boundary = "walls", custom: CustomFieldFn | null = null, t = 0) {
  if (name === "none" || strength === 0) return;
  if (name === "custom" && !custom) return; // no compiled fn → no-op
  const REF = 800;
  const sizeFactor = (Math.max(w, h) / REF) ** 2;
  const meanMass = 1.2;
  const scale = strength * 1500 * sizeFactor * meanMass;
  const samp = resolveSampling(sampling, boundary);
  if (mode === "analytic") {
    for (let i = a; i < b; i++) {
      const [x, y] = sampleCoords(samp, s.x[i * 2], s.x[i * 2 + 1], w, h);
      const g = fieldGradAnalytic(name, x, y, w, h);
      if (g === null) {
        const eps = 0.5 * (Math.max(w, h) / REF);
        const dphidx = (fieldPotential(name, x + eps, y, w, h, custom, t) - fieldPotential(name, x - eps, y, w, h, custom, t)) / (2 * eps);
        const dphidy = (fieldPotential(name, x, y + eps, w, h, custom, t) - fieldPotential(name, x, y - eps, w, h, custom, t)) / (2 * eps);
        s.f[i * 2]     += -dphidx * scale;
        s.f[i * 2 + 1] += -dphidy * scale;
      } else {
        s.f[i * 2]     += -g[0] * scale;
        s.f[i * 2 + 1] += -g[1] * scale;
      }
    }
    return;
  }
  const eps = 0.5 * (Math.max(w, h) / REF);
  for (let i = a; i < b; i++) {
    const [x, y] = sampleCoords(samp, s.x[i * 2], s.x[i * 2 + 1], w, h);
    const dphidx = (fieldPotential(name, x + eps, y, w, h, custom, t) - fieldPotential(name, x - eps, y, w, h, custom, t)) / (2 * eps);
    const dphidy = (fieldPotential(name, x, y + eps, w, h, custom, t) - fieldPotential(name, x, y - eps, w, h, custom, t)) / (2 * eps);
    s.f[i * 2]     += -dphidx * scale;
    s.f[i * 2 + 1] += -dphidy * scale;
  }
}


/**
 * project_constraints — Position-Based Dynamics (Gauss-Seidel) distance solver.
 *
 * For each iteration, every edge constraint pulls its two endpoints back to
 * `rest_length`, splitting the correction by inverse-mass. Velocities are
 * implicitly updated next integrator step (positions changed under them).
 */
function projectConstraints(s: State, iterations: number, dt: number) {
  if (iterations <= 0 || s.E === 0) return;
  const invDt = dt > 0 ? 1 / dt : 0;
  for (let it = 0; it < iterations; it++) {
    for (let e = 0; e < s.E; e++) {
      const i = s.edges[e * 2];
      const j = s.edges[e * 2 + 1];
      const dx = s.x[i * 2]     - s.x[j * 2];
      const dy = s.x[i * 2 + 1] - s.x[j * 2 + 1];
      const dist = Math.sqrt(dx * dx + dy * dy) + 1e-8;
      const rest = s.edgeRest[e];
      const wi = 1 / s.m[i], wj = 1 / s.m[j];
      const wsum = wi + wj;
      const c = (dist - rest) / dist / wsum;
      const cx = c * dx, cy = c * dy;
      s.x[i * 2]     -= wi * cx;
      s.x[i * 2 + 1] -= wi * cy;
      s.x[j * 2]     += wj * cx;
      s.x[j * 2 + 1] += wj * cy;
      // Reflect correction into velocity so motion stays consistent
      s.v[i * 2]     -= wi * cx * invDt * 0.5;
      s.v[i * 2 + 1] -= wi * cy * invDt * 0.5;
      s.v[j * 2]     += wj * cx * invDt * 0.5;
      s.v[j * 2 + 1] += wj * cy * invDt * 0.5;
    }
  }
}

type Boundary = "walls" | "wrap" | "periodic";

function stepState(
  s: State,
  dt: number,
  damping: number,
  w: number,
  h: number,
  integrator: "euler" | "semi-euler" | "verlet" = "semi-euler",
  boundary: Boundary = "walls",
  restitution: number = 0.7,
  dragMode: "explicit" | "exponential" | "force" = "explicit",
) {
  stepStateRange(s, dt, damping, w, h, integrator, 0, s.N, boundary, restitution, dragMode);
}

function stepStateRange(
  s: State,
  dt: number,
  damping: number,
  w: number,
  h: number,
  integrator: "euler" | "semi-euler" | "verlet" = "semi-euler",
  a = 0,
  b = s.N,
  boundary: Boundary = "walls",
  restitution: number = 0.7,
  dragMode: "explicit" | "exponential" | "force" = "explicit",
) {
  // Linear-drag decay factor applied to velocity each sub-step:
  //   "explicit"    → (1 − k·dt)        — cheap, classical, blows up if k·dt > 1
  //   "exponential" → exp(−k·dt)        — unconditionally stable, exact for the
  //                                        ODE  dv/dt = −k·v
  //   "force"       → drag is already in s.f as −k·m·v (added in the force
  //                   pipeline), so DO NOT decay velocity here (factor = 1)
  const decay =
    dragMode === "force"        ? 1 :
    dragMode === "exponential"  ? Math.exp(-damping * dt) :
                                  Math.max(0, 1 - damping * dt);
  if (integrator === "verlet") {
    // Verlet drift+kick are split across the force evaluation; see
    // verletKick() AFTER. This branch is now position-only damping wrap-up.
  } else if (integrator === "semi-euler") {
    for (let i = a; i < b; i++) {
      const invM = 1 / s.m[i];
      const ax = s.f[i * 2]     * invM;
      const ay = s.f[i * 2 + 1] * invM;
      s.v[i * 2]     = (s.v[i * 2]     + ax * dt) * decay;
      s.v[i * 2 + 1] = (s.v[i * 2 + 1] + ay * dt) * decay;
      s.x[i * 2]     += s.v[i * 2]     * dt;
      s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt;
    }
  } else {
    for (let i = a; i < b; i++) {
      const invM = 1 / s.m[i];
      const ax = s.f[i * 2]     * invM;
      const ay = s.f[i * 2 + 1] * invM;
      const vx0 = s.v[i * 2], vy0 = s.v[i * 2 + 1];
      s.v[i * 2]     = (vx0 + ax * dt) * decay;
      s.v[i * 2 + 1] = (vy0 + ay * dt) * decay;
      s.x[i * 2]     += vx0 * dt;
      s.x[i * 2 + 1] += vy0 * dt;
    }
  }
  // Boundary handling — three modes:
  //   walls:    inelastic reflection at the box edges with user-set
  //             restitution e ∈ [0,1] (0 = perfectly plastic, 1 = elastic).
  //             Tangential velocity is preserved; normal component flips
  //             and is scaled by -e.
  //   wrap:     positions teleport across edges; velocity unchanged
  //   periodic: same wrap, AND pairwise forces use the minimum-image
  //             convention so particles interact across the seam.
  if (boundary === "walls") {
    const e = restitution < 0 ? 0 : restitution > 1 ? 1 : restitution;
    // For verlet, also flip the normal component of fPrev when a particle
    // is reflected here. fPrev was just written to f at the end of the kick;
    // if the post-kick position landed past a wall, the next drift's
    // half-kick (v += ½·a_old·dt) would otherwise drive the particle back
    // into the wall it just bounced off, producing the classic
    // "stuck-to-wall" Verlet artifact.
    const flipPrev = integrator === "verlet";
    for (let i = a; i < b; i++) {
      if (s.x[i * 2] < 0) {
        s.x[i * 2] = 0;
        if (s.v[i * 2]     < 0) s.v[i * 2]     = -s.v[i * 2]     * e;
        if (flipPrev && s.fPrev[i * 2] < 0)     s.fPrev[i * 2]     = -s.fPrev[i * 2]     * e;
      } else if (s.x[i * 2] > w) {
        s.x[i * 2] = w;
        if (s.v[i * 2]     > 0) s.v[i * 2]     = -s.v[i * 2]     * e;
        if (flipPrev && s.fPrev[i * 2] > 0)     s.fPrev[i * 2]     = -s.fPrev[i * 2]     * e;
      }
      if (s.x[i * 2 + 1] < 0) {
        s.x[i * 2 + 1] = 0;
        if (s.v[i * 2 + 1] < 0) s.v[i * 2 + 1] = -s.v[i * 2 + 1] * e;
        if (flipPrev && s.fPrev[i * 2 + 1] < 0) s.fPrev[i * 2 + 1] = -s.fPrev[i * 2 + 1] * e;
      } else if (s.x[i * 2 + 1] > h) {
        s.x[i * 2 + 1] = h;
        if (s.v[i * 2 + 1] > 0) s.v[i * 2 + 1] = -s.v[i * 2 + 1] * e;
        if (flipPrev && s.fPrev[i * 2 + 1] > 0) s.fPrev[i * 2 + 1] = -s.fPrev[i * 2 + 1] * e;
      }
    }
  } else {
    // wrap & periodic both use modular position remapping
    for (let i = a; i < b; i++) {
      let xi = s.x[i * 2], yi = s.x[i * 2 + 1];
      xi = xi - Math.floor(xi / w) * w;
      yi = yi - Math.floor(yi / h) * h;
      s.x[i * 2] = xi;
      s.x[i * 2 + 1] = yi;
    }
  }
}


/**
 * scheduler.py — sync_boundaries(results)
 *
 *     for shared_node in boundary_nodes:
 *         avg = mean([r[node] for r in results])
 *         for r in results:
 *             r[node] = avg
 *
 * In the distributed version each worker keeps its own replica of the
 * boundary nodes (the "halo"). After a step, those replicas drift. The
 * canonical sync averages every replica and writes the mean back so all
 * workers agree on shared state.
 *
 * Here we have a single shared buffer, so the equivalent operation is:
 * for every node that owns a cross-partition edge, average its (x, v)
 * with the mean of its neighbors on the *other* side of the seam — that
 * is exactly what the all-reduce mean would settle to after one round.
 */
function syncBoundaries(s: State, partOf: (i: number) => number, idx?: BoundaryIndex | null) {
  if (s.E === 0) return;
  // Per-node accumulators for the "other-side" neighborhood mean.
  const sumX = new Float64Array(s.N * 2);
  const sumV = new Float64Array(s.N * 2);
  const cnt  = new Int32Array(s.N);

  // Fast path: precomputed boundary index iterates only seam pairs.
  // Each pair is (own, ghost) for partition q, so summing one direction
  // per partition reproduces the bidirectional accumulation below.
  if (idx && idx.N === s.N) {
    for (let q = 0; q < idx.W; q++) {
      const pr = idx.pairs[q];
      for (let k = 0; k < pr.length; k += 2) {
        const i = pr[k], j = pr[k + 1];
        sumX[i * 2]     += s.x[j * 2];
        sumX[i * 2 + 1] += s.x[j * 2 + 1];
        sumV[i * 2]     += s.v[j * 2];
        sumV[i * 2 + 1] += s.v[j * 2 + 1];
        cnt[i]++;
      }
    }
  } else {
    for (let e = 0; e < s.E; e++) {
      const i = s.edges[e * 2];
      const j = s.edges[e * 2 + 1];
      if (partOf(i) === partOf(j)) continue;
      // i sees j (across the seam) and vice-versa
      sumX[i * 2]     += s.x[j * 2];
      sumX[i * 2 + 1] += s.x[j * 2 + 1];
      sumV[i * 2]     += s.v[j * 2];
      sumV[i * 2 + 1] += s.v[j * 2 + 1];
      cnt[i]++;
      sumX[j * 2]     += s.x[i * 2];
      sumX[j * 2 + 1] += s.x[i * 2 + 1];
      sumV[j * 2]     += s.v[i * 2];
      sumV[j * 2 + 1] += s.v[i * 2 + 1];
      cnt[j]++;
    }
  }

  // Blend factor — full averaging (=1) jitters; a fraction matches the
  // "one round of all-reduce" smoothing from the distributed code.
  const a = 0.25;
  for (let i = 0; i < s.N; i++) {
    const c = cnt[i];
    if (c === 0) continue;
    const mx = sumX[i * 2] / c, my = sumX[i * 2 + 1] / c;
    const mvx = sumV[i * 2] / c, mvy = sumV[i * 2 + 1] / c;
    // r[node] = (1-a)*own + a*mean_of_replicas
    s.x[i * 2]     = (1 - a) * s.x[i * 2]     + a * mx;
    s.x[i * 2 + 1] = (1 - a) * s.x[i * 2 + 1] + a * my;
    s.v[i * 2]     = (1 - a) * s.v[i * 2]     + a * mvx;
    s.v[i * 2 + 1] = (1 - a) * s.v[i * 2 + 1] + a * mvy;
  }
}

/**
 * BoundaryIndex — preprocessed ghost/halo overlap descriptor per partition.
 *
 * For a partitioning into W workers (=GPUs), every cross-partition edge
 * (i,j) with partOf(i) ≠ partOf(j) creates a *ghost dependency*: worker
 * partOf(i) must read j's state, and partOf(j) must read i's state.
 *
 * Per worker q we precompute three flat Int32Arrays:
 *   • local[q]  — nodes OWNED by q that are referenced from another partition
 *                 (i.e. the "boundary nodes" that need to be SENT outward).
 *   • ghost[q]  — nodes OWNED by other partitions but referenced from q
 *                 (i.e. the halo cells q must RECEIVE before its kernel runs).
 *   • pairs[q]  — flat pairs (own, ghost) that q must reduce/average against.
 *
 * Built once per (N, W, edges) tuple and cached in a ref so the per-frame
 * sync_boundaries kernel iterates only over the seam, not all E edges.
 */
type BoundaryIndex = {
  W: number;
  N: number;
  edgeSig: number;
  local: Int32Array[];   // length W
  ghost: Int32Array[];   // length W
  pairs: Int32Array[];   // length W, flat [own0, ghost0, own1, ghost1, ...]
  totalLocal: number;
  totalGhost: number;
};

function buildBoundaryIndices(s: State, W: number): BoundaryIndex {
  const partOf = (i: number) => Math.min(W - 1, Math.floor((i * W) / s.N));
  const localSets: Set<number>[] = Array.from({ length: W }, () => new Set());
  const ghostSets: Set<number>[] = Array.from({ length: W }, () => new Set());
  const pairBuf:   number[][]    = Array.from({ length: W }, () => []);

  for (let e = 0; e < s.E; e++) {
    const i = s.edges[e * 2];
    const j = s.edges[e * 2 + 1];
    const pi = partOf(i), pj = partOf(j);
    if (pi === pj) continue;
    // i is owned by pi; from pi's POV, i is a LOCAL boundary node and j is a GHOST.
    localSets[pi].add(i); ghostSets[pi].add(j); pairBuf[pi].push(i, j);
    localSets[pj].add(j); ghostSets[pj].add(i); pairBuf[pj].push(j, i);
  }

  const local: Int32Array[] = [];
  const ghost: Int32Array[] = [];
  const pairs: Int32Array[] = [];
  let totalLocal = 0, totalGhost = 0;
  for (let q = 0; q < W; q++) {
    const lo = Int32Array.from(localSets[q]); lo.sort();
    const gh = Int32Array.from(ghostSets[q]); gh.sort();
    local.push(lo);
    ghost.push(gh);
    pairs.push(Int32Array.from(pairBuf[q]));
    totalLocal += lo.length;
    totalGhost += gh.length;
  }

  // Cheap structural fingerprint of the edge list — used as a cache key so
  // we rebuild only when topology actually changes (count + first/last node).
  const edgeSig = s.E === 0
    ? 0
    : (s.E * 1_000_003) ^ (s.edges[0] | 0) ^ ((s.edges[s.E * 2 - 1] | 0) << 13);

  return { W, N: s.N, edgeSig, local, ghost, pairs, totalLocal, totalGhost };
}

function buildEdges(N: number, perNode: number) {
  // Random sparse graph: each node connects to `perNode` neighbors
  const set = new Set<number>();
  const list: number[] = [];
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < perNode; k++) {
      const j = Math.floor(Math.random() * N);
      if (j === i) continue;
      const a = Math.min(i, j), b = Math.max(i, j);
      const key = a * 100000 + b;
      if (set.has(key)) continue;
      set.add(key);
      list.push(a, b);
    }
  }
  return new Int32Array(list);
}

function initState(
  N: number, w: number, h: number, perNode: number, rest: number,
  dtype: Dtype = "float32", device: Device = "cpu",
): State {
  const x = emptyLike(N * 2, dtype);
  const v = emptyLike(N * 2, dtype);
  const m = emptyLike(N, dtype);
  const f = emptyLike(N * 2, dtype);
  const fPrev = emptyLike(N * 2, dtype);
  const hue = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    x[i * 2] = Math.random() * w;
    x[i * 2 + 1] = Math.random() * h;
    const a = Math.random() * Math.PI * 2;
    const s = 30 + Math.random() * 40;
    v[i * 2] = Math.cos(a) * s;
    v[i * 2 + 1] = Math.sin(a) * s;
    m[i] = 0.6 + Math.random() * 1.8;
    hue[i] = Math.random();
  }
  const edges = buildEdges(N, perNode);
  const E = edges.length / 2;
  const edgeRest = emptyLike(E, dtype);
  edgeRest.fill(rest);
  // Ensemble starts at K=0 (off); allocated lazily when stochastic mode flips on.
  return { N, D: 2, dtype, device, x, v, m, f, fPrev, hue, edges, edgeRest, E, verletPrimed: false, K: 0, ensX: new Float32Array(0), ensV: new Float32Array(0) };
}

/** (Re)allocate the Monte Carlo ensemble in-place. Replicas start at the
 *  deterministic mean (zero offset) with zero relative velocity, so the
 *  spread grows organically from the noise/dynamics rather than being
 *  seeded from an arbitrary prior. */
function ensureEnsemble(s: State, K: number) {
  if (s.K === K) return;
  s.K = K;
  s.ensX = new Float32Array(K * s.N * 2);
  s.ensV = new Float32Array(K * s.N * 2);
}

/** Box–Muller — two unit-variance Gaussians per call. */
function randn2(out: [number, number]) {
  let u = Math.random();
  if (u < 1e-12) u = 1e-12;
  const v = Math.random();
  const r = Math.sqrt(-2 * Math.log(u));
  const t = 2 * Math.PI * v;
  out[0] = r * Math.cos(t);
  out[1] = r * Math.sin(t);
}

/** Standard-normal CDF (Abramowitz & Stegun 7.1.26 erf approximation).
 *  Used to convert a probabilistic edge tolerance into P(|stretch|<tol). */
function normCdf(z: number): number {
  const sign = z < 0 ? -1 : 1;
  const ax = Math.abs(z) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * ax);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-ax * ax);
  return 0.5 * (1 + sign * y);
}

export function PhysicsCanvas({
  params,
  pointerRef,
  onValidation,
  onLoss,
}: {
  params: SimParams;
  pointerRef: React.MutableRefObject<{ x: number; y: number; active: boolean; mode: 1 | -1 }>;
  onValidation?: (r: ValidationReport) => void;
  onLoss?: (loss: number) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef<State | null>(null);
  const paramsRef = useRef(params);
  paramsRef.current = params;
  const lastValidationRef = useRef(0);
  const stepOnceRef = useRef(0);
  const boundaryIdxRef = useRef<BoundaryIndex | null>(null);
  const prevIntegratorRef = useRef<string>("");
  const onValidationRef = useRef(onValidation);
  onValidationRef.current = onValidation;
  const onLossRef = useRef(onLoss);
  onLossRef.current = onLoss;
  const lossEmaRef = useRef(0);
  const energyBaselineRef = useRef<number | null>(null);
  const energyBaselineNRef = useRef(0);
  // Rolling history of E_total and Δ for the sparkline + drift statistics.
  // Capacity ≈ a few seconds at 60 fps; cheap fixed-size circular buffer.
  const ENERGY_HIST_CAP = 240;
  const energyHistRef = useRef<Float32Array>(new Float32Array(ENERGY_HIST_CAP));
  const driftHistRef  = useRef<Float32Array>(new Float32Array(ENERGY_HIST_CAP));
  const energyHistLenRef = useRef(0);
  const energyHistHeadRef = useRef(0);
  // EMA of |Δ| and |Δ|² → smoothed drift magnitude and RMS drift.
  const driftAbsEmaRef = useRef(0);
  const driftSqEmaRef  = useRef(0);
  // Track integrator changes so switching Euler↔Verlet rebases the baseline
  // and clears history (otherwise the sparkline shows a meaningless step).
  const prevIntegratorEnergyRef = useRef<string>("");
  const lastSubStepsRef = useRef(1);
  const fpsEmaRef = useRef(60);
  // ── Digital Twin telemetry (synthetic IoT/sensor stream) ─────────
  // Each sensor has: a Lissajous phase pair, an assigned particle id
  // (re-bound on count change), the latest reading (px,py) with noise,
  // and the residual = ||sensor − particle|| in σ-units (anomaly score).
  const twinSensorsRef = useRef<{
    px: number; py: number;       // latest noisy reading
    bound: number;                // particle index it's tracking
    ax: number; ay: number;       // Lissajous frequencies
    phx: number; phy: number;     // phase offsets
    residual: number;             // |reading − sim| in pixels
    z: number;                    // residual / sensorNoise (z-score)
  }[]>([]);
  const twinAnomalyCountRef = useRef(0);
  const twinResidualEmaRef = useRef(0);

  // Compile the user-provided Φ exactly when the source string changes.
  // useMemo gives us a stable reference per source (cheap, parse is < 1 ms),
  // and a ref keeps it readable from the rAF loop without re-subscribing.
  const compiled = useMemo(() => compileFieldExpr(params.customFieldSrc || "0"), [params.customFieldSrc]);
  const customFnRef = useRef<((env: import("@/lib/exprCompile").FieldEnv) => number) | null>(null);
  customFnRef.current = compiled.ok ? compiled.fn : null;
  const tStartRef = useRef(performance.now());

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let last = performance.now();

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const r = canvas.getBoundingClientRect();
      canvas.width = r.width * dpr;
      canvas.height = r.height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (!stateRef.current) {
        const p = paramsRef.current;
        stateRef.current = initState(p.particleCount, r.width, r.height, p.edgesPerNode, p.restLength, p.dtype, p.device);
      }
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const step = (now: number) => {
      const p0 = paramsRef.current;
      const rawDt = p0.paused ? 1 / 60 : Math.min(0.033, (now - last) / 1000);
      const dt = rawDt * (p0.dtScale ?? 1);
      last = now;
      const p = paramsRef.current;
      const r = canvas.getBoundingClientRect();
      const w = r.width, h = r.height;

      let s = stateRef.current!;
      if (s.N !== p.particleCount) {
        s = initState(p.particleCount, w, h, p.edgesPerNode, p.restLength, p.dtype, p.device);
        stateRef.current = s;
      } else if (s.dtype !== p.dtype || s.device !== p.device) {
        // PhysicsState.to(device, dtype) — re-cast all tensors, re-init f
        s = toDevice(s, p.device, p.dtype);
        stateRef.current = s;
      }

      // Trail fade
      ctx.fillStyle = `oklch(0.16 0.02 260 / ${1 - p.trail})`;
      ctx.fillRect(0, 0, w, h);

      // Field visualization (faint isolines / heatmap of Φ)
      if (p.field !== "none" && p.fieldStrength !== 0) {
        const cell = 28;
        const cols = Math.ceil(w / cell);
        const rows = Math.ceil(h / cell);
        // Sample to find range
        let pmin = Infinity, pmax = -Infinity;
        const samples = new Float32Array(cols * rows);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const v = fieldPotential(p.field, c * cell + cell / 2, r * cell + cell / 2, w, h, customFnRef.current, (now - tStartRef.current) / 1000);
            samples[r * cols + c] = v;
            if (v < pmin) pmin = v;
            if (v > pmax) pmax = v;
          }
        }
        const range = pmax - pmin || 1;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const t = (samples[r * cols + c] - pmin) / range;
            const a = 0.05 + 0.18 * Math.abs(t - 0.5) * 2;
            const hueDeg = 200 + t * 120;
            ctx.fillStyle = `oklch(0.55 0.14 ${hueDeg} / ${a})`;
            ctx.fillRect(c * cell, r * cell, cell, cell);
          }
        }
      }

      // Runtime shape & dtype guards — verify x, v, m, f BEFORE the step
      const report = validateState(s);
      if (now - lastValidationRef.current > 250) {
        lastValidationRef.current = now;
        onValidationRef.current?.(report);
      }
      if (!report.ok) {
        // Skip simulation if invariants broken; still render last frame
        raf = requestAnimationFrame(step);
        return;
      }

      const stepRequested = (p.stepOnce ?? 0) > stepOnceRef.current;
      if (stepRequested) stepOnceRef.current = p.stepOnce ?? 0;
      if (!p.paused || stepRequested) {
        // run_simulation(state, config): for t in range(config.steps): ...
        // Adaptive sub-stepping: when edges are stretched well past their
        // rest length OR particles are moving fast enough that one Euler
        // sub-step would jump > ~half a pairwise radius, increase subSteps
        // up to maxSubSteps. This keeps spring & contact resolution stable
        // through transient impacts without paying the cost every frame.
        let subSteps = Math.max(1, p.subSteps | 0);
        if (p.adaptiveSubSteps) {
          // worst-case velocity (pixels / second)
          let vmax2 = 0;
          for (let i = 0; i < s.N; i++) {
            const vx = s.v[i * 2], vy = s.v[i * 2 + 1];
            const v2 = vx * vx + vy * vy;
            if (v2 > vmax2) vmax2 = v2;
          }
          const vmax = Math.sqrt(vmax2);
          // worst-case edge stretch ratio (|edge| / rest)
          let stretchMax = 1;
          const E = (s.edges.length / 2) | 0;
          for (let e = 0; e < E; e++) {
            const i = s.edges[e * 2], j = s.edges[e * 2 + 1];
            const dx = s.x[j * 2] - s.x[i * 2];
            const dy = s.x[j * 2 + 1] - s.x[i * 2 + 1];
            const L = Math.hypot(dx, dy);
            const rest = s.edgeRest[e] || 1;
            const r = L / rest;
            if (r > stretchMax) stretchMax = r;
          }
          // Triggers: ≥1 sub-step per (vmax * dt) / (radius/2),
          // and ≥1 per (stretch - 1) / 0.15 (i.e. 1 extra per 15% over rest).
          const motionTrigger = (vmax * dt) / Math.max(1, p.pairwiseRadius * 0.5);
          const stretchTrigger = Math.max(0, stretchMax - 1.0) / 0.15;
          const need = Math.ceil(Math.max(motionTrigger, stretchTrigger, 1));
          subSteps = Math.min(Math.max(p.maxSubSteps | 0, subSteps), Math.max(subSteps, need));
        }
        const subDt = dt / subSteps;
        lastSubStepsRef.current = subSteps;
        const k = p.springK;
        const pStr = p.pairwiseStrength;
        const pRad = p.pairwiseRadius;
        const r2max = pRad * pRad;
        const norm = pRad * 0.5;

        // ── scheduler.py ──────────────────────────────────────────────
        // DistributedSimulator.step(partitions): partition nodes across
        // `workers`, run each worker's local pipeline, then sync_boundaries
        // by re-projecting edges that cross partition borders. Workers run
        // sequentially here (single thread) but each sees only its own slice
        // of state — same data-dependency pattern as the Ray/MPI version.
        const W = Math.max(1, Math.min(p.workers | 0, s.N));
        const partOf = (i: number) => Math.min(W - 1, Math.floor((i * W) / s.N));
        const partStart = (q: number) => Math.floor((q * s.N) / W);
        const partEnd   = (q: number) => Math.floor(((q + 1) * s.N) / W);

        for (let t = 0; t < subSteps; t++) {
          // Re-prime fPrev whenever we (re)enter verlet from another integrator.
          // s.fPrev still holds whatever was there before — could be zeros
          // (fresh state) or stale euler-era forces — neither is valid as a₀.
          if (p.integrator === "verlet" && prevIntegratorRef.current !== "verlet") {
            s.verletPrimed = false;
          }
          prevIntegratorRef.current = p.integrator;
          // Velocity-Verlet drift uses the PREVIOUS step's forces (s.fPrev)
          // for the first half-kick, then advances positions. This must run
          // BEFORE we recompute forces for the new positions.
          if (p.integrator === "verlet" && s.verletPrimed) {
            for (let q = 0; q < W; q++) {
              verletDrift(s, subDt, partStart(q), partEnd(q), w, h, p.boundary, p.restitution);
            }
          }

          // 1. zero forces — state.f.zero_()
          s.f.fill(0);

          // 2. per-worker local forces (each worker owns nodes [a,b))
          for (let q = 0; q < W; q++) {
            const a = partStart(q), b = partEnd(q);

            // gravity (local) — three modes:
            //   "zero"        → no body force
            //   "uniform"     → classic +y body force (down)
            //   "directional" → vector force along gravityAngle (degrees)
            const gMode = p.gravityMode ?? "uniform";
            if (gMode !== "zero" && p.gravity !== 0) {
              if (gMode === "directional") {
                const ang = ((p.gravityAngle ?? 90) * Math.PI) / 180;
                const gx = Math.cos(ang) * p.gravity;
                const gy = Math.sin(ang) * p.gravity;
                for (let i = a; i < b; i++) {
                  s.f[i * 2]     += gx * s.m[i];
                  s.f[i * 2 + 1] += gy * s.m[i];
                }
              } else {
                for (let i = a; i < b; i++) {
                  s.f[i * 2 + 1] += p.gravity * s.m[i];
                }
              }
            }

            // linear drag as a body force: F_drag = −k · m · v
            // Only active when dragMode === "force"; the integrator's velocity
            // decay is disabled in that case so we don't double-count.
            if (p.dragMode === "force" && p.damping > 0) {
              const kDrag = p.damping;
              for (let i = a; i < b; i++) {
                const m = s.m[i];
                s.f[i * 2]     -= kDrag * m * s.v[i * 2];
                s.f[i * 2 + 1] -= kDrag * m * s.v[i * 2 + 1];
              }
            }

            // pointer attractor (local)
            if (pointerRef.current.active) {
              const px = pointerRef.current.x, py = pointerRef.current.y;
              const sign = pointerRef.current.mode;
              const G = p.attractor * sign;
              for (let i = a; i < b; i++) {
                const dx = px - s.x[i * 2];
                const dy = py - s.x[i * 2 + 1];
                const r2 = dx * dx + dy * dy + 400;
                const inv = 1 / Math.sqrt(r2);
                const acc = (G * s.m[i]) / r2;
                s.f[i * 2]     += dx * inv * acc * 1000;
                s.f[i * 2 + 1] += dy * inv * acc * 1000;
              }
            }

            // pairwise: handled globally below via uniform spatial grid
            // (kept here as a no-op slot so the worker pipeline order is preserved)

            // potential field (local)
            computePotentialForces_range(s, p.field, p.fieldStrength, w, h, a, b, p.potentialGrad, p.fieldSampling, p.boundary, customFnRef.current, (now - tStartRef.current) / 1000);
          }

          // 2b. pairwise via uniform spatial grid — O(N) instead of O(N²).
          // Bin every node into a cell of size = pairwiseRadius. Each pair
          // is only tested against neighbors in the same or 4 forward
          // cells (so each unordered pair is visited once). 5–20× faster
          // than all-pairs for typical pRad — the sim now scales past 5k.
          if (pStr !== 0 && pRad > 0 && p.pairwiseAlgo === "grid") {
            const cell = pRad;
            const gw = Math.max(1, Math.ceil(w / cell));
            const gh = Math.max(1, Math.ceil(h / cell));
            const nCells = gw * gh;
            const cellCount = new Int32Array(nCells);
            const cellOf = new Int32Array(s.N);
            for (let i = 0; i < s.N; i++) {
              const cxi = Math.min(gw - 1, Math.max(0, (s.x[i * 2] / cell) | 0));
              const cyi = Math.min(gh - 1, Math.max(0, (s.x[i * 2 + 1] / cell) | 0));
              const c = cyi * gw + cxi;
              cellOf[i] = c;
              cellCount[c]++;
            }
            // Counting-sort prefix-sum → contiguous per-cell ranges in `order`.
            const cellStart = new Int32Array(nCells + 1);
            for (let c = 0; c < nCells; c++) cellStart[c + 1] = cellStart[c] + cellCount[c];
            const cursor = new Int32Array(nCells);
            const order = new Int32Array(s.N);
            for (let i = 0; i < s.N; i++) {
              const c = cellOf[i];
              order[cellStart[c] + cursor[c]++] = i;
            }
            for (let cy = 0; cy < gh; cy++) {
              for (let cx = 0; cx < gw; cx++) {
                const c = cy * gw + cx;
                const aS = cellStart[c], aE = cellStart[c + 1];
                if (aS === aE) continue;
                for (let dyc = 0; dyc <= 1; dyc++) {
                  for (let dxc = -1; dxc <= 1; dxc++) {
                    if (dyc === 0 && dxc < 0) continue; // dedupe pairs
                    const nxi = cx + dxc, nyi = cy + dyc;
                    if (nxi < 0 || nxi >= gw || nyi >= gh) continue;
                    const cn = nyi * gw + nxi;
                    const bS = cellStart[cn], bE = cellStart[cn + 1];
                    if (bS === bE) continue;
                    const same = c === cn;
                    for (let ai = aS; ai < aE; ai++) {
                      const i = order[ai];
                      const xi = s.x[i * 2], yi = s.x[i * 2 + 1];
                      const startB = same ? ai + 1 : bS;
                      for (let bi = startB; bi < bE; bi++) {
                        const j = order[bi];
                        let dx = xi - s.x[j * 2];
                        let dy = yi - s.x[j * 2 + 1];
                        // Minimum-image convention for periodic boundary —
                        // wrap the displacement to the [-w/2, w/2] interval
                        // so a particle near the right edge "sees" its
                        // neighbor near the left edge across the seam.
                        if (p.boundary === "periodic") {
                          if (dx >  w * 0.5) dx -= w; else if (dx < -w * 0.5) dx += w;
                          if (dy >  h * 0.5) dy -= h; else if (dy < -h * 0.5) dy += h;
                        }
                        const r2 = dx * dx + dy * dy;
                        if (r2 > r2max || r2 < 1e-4) continue;
                        const dist = Math.sqrt(r2);
                        // Pairwise force model — sign convention: +fmag pushes apart.
                        //   "lj":      Lennard-Jones-like  (σ/r)¹² − (σ/r)⁶
                        //              short-range repulsion + medium-range attraction
                        //   "repel":   pure (σ/r)² soft-core repulsion
                        //   "attract": (1 − r/rad) linear well, attractive only
                        let fmag: number;
                        if (p.pairwiseMode === "repel") {
                          fmag = pStr * (norm * norm) / r2;
                        } else if (p.pairwiseMode === "attract") {
                          fmag = -pStr * (1 - dist / pRad);
                        } else {
                          // lj-like (current behavior, kept as default)
                          fmag = pStr * (norm * norm / r2 - norm / dist);
                        }
                        const fx = (dx / dist) * fmag;
                        const fy = (dy / dist) * fmag;
                        s.f[i * 2]     += fx;
                        s.f[i * 2 + 1] += fy;
                        s.f[j * 2]     -= fx;
                        s.f[j * 2 + 1] -= fy;
                      }
                    }
                  }
                }
              }
            }
          }

          // 2c. all-pairs fallback — O(N²) reference path.
          if (pStr !== 0 && pRad > 0 && p.pairwiseAlgo === "all-pairs") {
            for (let i = 0; i < s.N; i++) {
              const xi = s.x[i * 2], yi = s.x[i * 2 + 1];
              for (let j = i + 1; j < s.N; j++) {
                let dx = xi - s.x[j * 2];
                let dy = yi - s.x[j * 2 + 1];
                if (p.boundary === "periodic") {
                  if (dx >  w * 0.5) dx -= w; else if (dx < -w * 0.5) dx += w;
                  if (dy >  h * 0.5) dy -= h; else if (dy < -h * 0.5) dy += h;
                }
                const r2 = dx * dx + dy * dy;
                if (r2 > r2max || r2 < 1e-4) continue;
                const dist = Math.sqrt(r2);
                let fmag: number;
                if (p.pairwiseMode === "repel") {
                  fmag = pStr * (norm * norm) / r2;
                } else if (p.pairwiseMode === "attract") {
                  fmag = -pStr * (1 - dist / pRad);
                } else {
                  fmag = pStr * (norm * norm / r2 - norm / dist);
                }
                const fx = (dx / dist) * fmag;
                const fy = (dy / dist) * fmag;
                s.f[i * 2]     += fx;
                s.f[i * 2 + 1] += fy;
                s.f[j * 2]     -= fx;
                s.f[j * 2 + 1] -= fy;
              }
            }
          }


          // 3. springs on ALL edges — interior edges are local to one
          // worker; boundary edges (i,j in different partitions) are the
          // sync points exchanged between workers.
          for (let e = 0; e < s.E; e++) {
            const i = s.edges[e * 2];
            const j = s.edges[e * 2 + 1];
            const dx = s.x[i * 2]     - s.x[j * 2];
            const dy = s.x[i * 2 + 1] - s.x[j * 2 + 1];
            const dist = Math.sqrt(dx * dx + dy * dy) + 1e-8;
            const mag = -k * (dist - s.edgeRest[e]);
            const fx = (dx / dist) * mag;
            const fy = (dy / dist) * mag;
            s.f[i * 2]     += fx;
            s.f[i * 2 + 1] += fy;
            s.f[j * 2]     -= fx;
            s.f[j * 2 + 1] -= fy;
          }

          // 4. step(state, dt) — each worker integrates its own slice.
          // For velocity-Verlet, drift already happened above; here we apply
          // the second half-kick using the NEW forces, then cache f→fPrev.
          if (p.integrator === "verlet") {
            if (!s.verletPrimed) {
              // Priming substep: forces have just been evaluated at x₀ but
              // we did NOT drift, and we must not kick (no a_old to combine
              // with). Seed fPrev = f(x₀) so the next substep's verletDrift
              // half-kick uses the correct initial acceleration. Boundary
              // handling still runs so reflections behave consistently.
              s.fPrev.set(s.f);
              s.verletPrimed = true;
              for (let q = 0; q < W; q++) {
                stepStateRange(s, subDt, p.damping, w, h, p.integrator, partStart(q), partEnd(q), p.boundary, p.restitution, p.dragMode);
              }
            } else {
              for (let q = 0; q < W; q++) {
                verletKick(s, subDt, p.damping, partStart(q), partEnd(q), p.dragMode);
                // still call stepStateRange for boundary handling (verlet branch is a no-op for motion)
                stepStateRange(s, subDt, p.damping, w, h, p.integrator, partStart(q), partEnd(q), p.boundary, p.restitution, p.dragMode);
              }
            }
          } else {
            for (let q = 0; q < W; q++) {
              stepStateRange(s, subDt, p.damping, w, h, p.integrator, partStart(q), partEnd(q), p.boundary, p.restitution, p.dragMode);
            }
          }

          // 5. sync_boundaries — re-project cross-partition edges so the
          // independently-stepped slices stay consistent at the seams.
          // Rebuild boundary_indices when N, W, or edge topology changes.
          projectConstraints(s, p.constraintIters, subDt);
          {
            const eSig = s.E === 0
              ? 0
              : (s.E * 1_000_003) ^ (s.edges[0] | 0) ^ ((s.edges[s.E * 2 - 1] | 0) << 13);
            const cur = boundaryIdxRef.current;
            if (!cur || cur.W !== W || cur.N !== s.N || cur.edgeSig !== eSig) {
              boundaryIdxRef.current = buildBoundaryIndices(s, W);
            }
          }
          syncBoundaries(s, partOf, boundaryIdxRef.current);

          // ── probabilistic_runtime.py ────────────────────────────
          // Monte Carlo uncertainty propagation. Each of K replicas
          // tracks a position-OFFSET δx_k from the deterministic mean.
          // Linearized dynamics around the mean trajectory:
          //   δv_k ← (δv_k + a_mean·0·dt + ξ·σ√dt) · (1 − damping·dt)
          //   δx_k ← δx_k + δv_k · dt
          // The mean-acceleration term cancels (already absorbed by the
          // deterministic state), leaving the noise injection (Langevin
          // term) and damping decay. Variance grows like σ²·t until
          // damping balances it ⇒ stationary σ_x ≈ σ/(damping·√(2γ)).
          const Kreq = p.stochastic ? Math.max(0, Math.min(64, p.ensembleK | 0)) : 0;
          if (Kreq !== s.K) ensureEnsemble(s, Kreq);
          if (s.K > 0) {
            const sig = Math.max(0, p.noiseSigma);
            const sqrtDt = Math.sqrt(subDt);
            const decay = 1 - p.damping * subDt;
            const tmp: [number, number] = [0, 0];
            for (let kk = 0; kk < s.K; kk++) {
              const base = kk * s.N * 2;
              for (let i = 0; i < s.N; i++) {
                randn2(tmp);
                const o = base + i * 2;
                s.ensV[o]     = (s.ensV[o]     + sig * sqrtDt * tmp[0]) * decay;
                s.ensV[o + 1] = (s.ensV[o + 1] + sig * sqrtDt * tmp[1]) * decay;
                s.ensX[o]     += s.ensV[o]     * subDt;
                s.ensX[o + 1] += s.ensV[o + 1] * subDt;
              }
            }
          }
        }
      }

      // ── digital_twin.py ───────────────────────────────────────────
      // Telemetry intake + state assimilation. Synthetic Lissajous
      // "ground-truth" sensors stream noisy positions; for each sensor
      // we (a) advance its trajectory, (b) draw a noisy reading, and
      // (c) blend the bound particle's position toward it with gain g
      // (Kalman-lite: x ← (1−g)·x + g·z, v gets a corrective impulse).
      // Anomalies are flagged when |residual|/σ_sensor exceeds twinAnomalyZ.
      if (p.twinEnabled && !p.paused && s.N > 0) {
        const M = Math.max(0, Math.min(64, p.twinSensorCount | 0));
        const sensors = twinSensorsRef.current;
        // Re-allocate sensors when count or particle pool changes
        if (sensors.length !== M) {
          sensors.length = 0;
          for (let i = 0; i < M; i++) {
            sensors.push({
              px: 0, py: 0,
              bound: i % Math.max(1, s.N),
              ax: 0.3 + Math.random() * 0.6,
              ay: 0.3 + Math.random() * 0.6,
              phx: Math.random() * Math.PI * 2,
              phy: Math.random() * Math.PI * 2,
              residual: 0, z: 0,
            });
          }
        }
        for (let i = 0; i < sensors.length; i++) {
          const sn = sensors[i];
          if (sn.bound >= s.N) sn.bound = i % s.N;
        }
        const tNow = (now - tStartRef.current) / 1000;
        const cx = w * 0.5, cy = h * 0.5;
        const rx = w * 0.38, ry = h * 0.38;
        const sigS = Math.max(0.1, p.twinSensorNoise);
        const g = Math.max(0, Math.min(1, p.twinAssimGain));
        let anomalyN = 0;
        let resAcc = 0;
        const tmp: [number, number] = [0, 0];
        for (let i = 0; i < sensors.length; i++) {
          const sn = sensors[i];
          // Synthetic ground truth = Lissajous around canvas center
          const gx = cx + rx * Math.sin(sn.ax * tNow + sn.phx);
          const gy = cy + ry * Math.sin(sn.ay * tNow + sn.phy);
          // Add Gaussian sensor noise
          randn2(tmp);
          sn.px = gx + sigS * tmp[0];
          sn.py = gy + sigS * tmp[1];
          // Residual vs bound particle (innovation)
          const i2 = sn.bound * 2;
          const rxi = sn.px - s.x[i2];
          const ryi = sn.py - s.x[i2 + 1];
          const r = Math.hypot(rxi, ryi);
          sn.residual = r;
          sn.z = r / sigS;
          if (sn.z > p.twinAnomalyZ) anomalyN++;
          resAcc += r;
          // Assimilate: nudge position by g·innovation, add velocity impulse
          s.x[i2]     += g * rxi;
          s.x[i2 + 1] += g * ryi;
          if (dt > 1e-6) {
            s.v[i2]     += (g * rxi) / dt * 0.25;
            s.v[i2 + 1] += (g * ryi) / dt * 0.25;
          }
        }
        twinAnomalyCountRef.current = anomalyN;
        const meanRes = sensors.length > 0 ? resAcc / sensors.length : 0;
        twinResidualEmaRef.current = twinResidualEmaRef.current * 0.85 + meanRes * 0.15;
      } else if (twinSensorsRef.current.length > 0 && !p.twinEnabled) {
        twinSensorsRef.current.length = 0;
        twinAnomalyCountRef.current = 0;
      }

      //   loss = objective(final_state); loss.backward()
      // Objective: drive every node toward the canvas center.
      //   L = ½ * mean(||x - target||²)
      //   ∂L/∂x_i = (x_i - target) / N    (the autograd "gradient")
      // We then descend along that gradient with learning rate `objectiveLR`,
      // which is what `optimizer.step()` would do after .backward().
      const tx = w * 0.5, ty = h * 0.5;
      let loss = 0;
      for (let i = 0; i < s.N; i++) {
        const dx = s.x[i * 2] - tx;
        const dy = s.x[i * 2 + 1] - ty;
        loss += 0.5 * (dx * dx + dy * dy);
      }
      loss /= Math.max(1, s.N);
      // Normalize to a more interpretable scale (canvas-diagonal²)
      const lossNorm = loss / ((w * w + h * h) * 0.25);
      lossEmaRef.current = lossEmaRef.current * 0.9 + lossNorm * 0.1;
      if (now - lastValidationRef.current < 50 || (now | 0) % 4 === 0) {
        onLossRef.current?.(lossEmaRef.current);
      }

      if (p.optimize && !p.paused && p.objectiveLR > 0) {
        // loss.backward() → grad on positions; descend
        const lr = p.objectiveLR;
        const invN = 1 / Math.max(1, s.N);
        for (let i = 0; i < s.N; i++) {
          const gx = (s.x[i * 2]     - tx) * invN;
          const gy = (s.x[i * 2 + 1] - ty) * invN;
          s.x[i * 2]     -= lr * gx;
          s.x[i * 2 + 1] -= lr * gy;
        }
      }

      // Target marker for the objective
      if (p.optimize) {
        ctx.strokeStyle = "oklch(0.85 0.18 85 / 0.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(tx, ty, 6, 0, Math.PI * 2);
        ctx.moveTo(tx - 12, ty); ctx.lineTo(tx + 12, ty);
        ctx.moveTo(tx, ty - 12); ctx.lineTo(tx, ty + 12);
        ctx.stroke();
      }

      // ── energy_monitor.py ─────────────────────────────────────────
      // Total mechanical energy each frame, decomposed by source.
      //   KE         = ½ Σ mᵢ |vᵢ|²
      //   PE_grav    = Σ mᵢ g (h - yᵢ)               (y grows downward)
      //   PE_spring  = Σ ½ k (|xᵢ - xⱼ| - rest)²
      //   PE_field   = Σ strength·1500·Φ(xᵢ, yᵢ)     (matches force scaling)
      //   PE_pairwise — omitted (mode-dependent integral, sketchy estimate)
      // Symplectic integrators (semi-Euler, velocity-Verlet) bound drift;
      // explicit Euler trends upward. Damping & PBD intentionally inject
      // / remove energy — drift here is informative, not a bug.
      let KE = 0;
      for (let i = 0; i < s.N; i++) {
        const vx = s.v[i * 2], vy = s.v[i * 2 + 1];
        KE += 0.5 * s.m[i] * (vx * vx + vy * vy);
      }
      let PE_grav = 0;
      const gMode = p.gravityMode ?? "uniform";
      if (gMode !== "zero" && p.gravity !== 0) {
        if (gMode === "directional") {
          const ang = ((p.gravityAngle ?? 90) * Math.PI) / 180;
          const gx = Math.cos(ang) * p.gravity;
          const gy = Math.sin(ang) * p.gravity;
          // PE = -m·(g · r) with reference at origin
          for (let i = 0; i < s.N; i++) {
            PE_grav -= s.m[i] * (gx * s.x[i * 2] + gy * s.x[i * 2 + 1]);
          }
        } else {
          for (let i = 0; i < s.N; i++) {
            PE_grav += s.m[i] * p.gravity * (h - s.x[i * 2 + 1]);
          }
        }
      }
      let PE_spring = 0;
      let cMax = 0;        // max |stretch / rest|  — peak constraint error
      let cRms = 0;        // RMS relative stretch  — overall constraint error
      if (s.E > 0) {
        for (let e = 0; e < s.E; e++) {
          const i2 = s.edges[e * 2], j2 = s.edges[e * 2 + 1];
          const ddx = s.x[i2 * 2]     - s.x[j2 * 2];
          const ddy = s.x[i2 * 2 + 1] - s.x[j2 * 2 + 1];
          const d = Math.sqrt(ddx * ddx + ddy * ddy);
          const rest = s.edgeRest[e] || 1;
          const stretch = d - rest;
          if (p.springK > 0) PE_spring += 0.5 * p.springK * stretch * stretch;
          const rel = Math.abs(stretch) / rest;
          if (rel > cMax) cMax = rel;
          cRms += rel * rel;
        }
        cRms = Math.sqrt(cRms / s.E);
      }
      let PE_field = 0;
      if (p.field !== "none" && p.fieldStrength !== 0) {
        const sc = p.fieldStrength * 1500 * (Math.max(w, h) / 800) ** 2 * 1.2;
        const samp = resolveSampling(p.fieldSampling, p.boundary);
        for (let i = 0; i < s.N; i++) {
          const [px, py] = sampleCoords(samp, s.x[i * 2], s.x[i * 2 + 1], w, h);
          PE_field += sc * fieldPotential(p.field, px, py, w, h, customFnRef.current, (now - tStartRef.current) / 1000);
        }
      }
      const PE_total = PE_grav + PE_spring + PE_field;
      const E_total  = KE + PE_total;

      // Rebase baseline when N changes OR integrator changes — comparing
      // drift across integrators only makes sense from a fresh zero.
      if (
        energyBaselineRef.current === null ||
        s.N !== energyBaselineNRef.current ||
        p.integrator !== prevIntegratorEnergyRef.current
      ) {
        energyBaselineRef.current = E_total;
        energyBaselineNRef.current = s.N;
        prevIntegratorEnergyRef.current = p.integrator;
        energyHistLenRef.current = 0;
        energyHistHeadRef.current = 0;
        driftAbsEmaRef.current = 0;
        driftSqEmaRef.current = 0;
      }
      const baseline = energyBaselineRef.current ?? E_total;
      const drift = E_total - baseline;
      // Relative drift |Δ|/|E₀| — the actually meaningful stability metric.
      const relDrift = Math.abs(baseline) > 1e-9 ? drift / Math.abs(baseline) : 0;
      // EMA constants chosen for ~1s smoothing at 60fps.
      driftAbsEmaRef.current = driftAbsEmaRef.current * 0.94 + Math.abs(drift) * 0.06;
      driftSqEmaRef.current  = driftSqEmaRef.current  * 0.94 + drift * drift * 0.06;
      const driftRms = Math.sqrt(driftSqEmaRef.current);
      // Push into circular history.
      {
        const head = energyHistHeadRef.current;
        energyHistRef.current[head] = E_total;
        driftHistRef.current[head]  = drift;
        energyHistHeadRef.current = (head + 1) % ENERGY_HIST_CAP;
        if (energyHistLenRef.current < ENERGY_HIST_CAP) energyHistLenRef.current++;
      }

      const fmt = (n: number) => {
        const a = Math.abs(n);
        if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
        if (a >= 1e3) return (n / 1e3).toFixed(2) + "k";
        if (a >= 1)   return n.toFixed(2);
        return n.toExponential(1);
      };
      // FPS — exponential moving average so the readout is stable enough
      // to read at a glance while still tracking real frame-time changes.
      const instFps = dt > 1e-6 ? 1 / dt : 60;
      fpsEmaRef.current = fpsEmaRef.current * 0.92 + instFps * 0.08;
      const fps = fpsEmaRef.current;
      const subStepsEff = lastSubStepsRef.current;

      // ── Probabilistic diagnostics ───────────────────────────────
      // Reduce ensemble offsets to a per-particle isotropic σ
      //   σᵢ² = (1/K) Σ_k (δxᵢ,k² + δyᵢ,k²) / 2     (mean is zero by construction)
      // Aggregate to a scene-wide σ̄ (RMS over particles), and convert the
      // user-set probabilistic edge tolerance into a satisfaction probability:
      //   P(|edge_stretch|/rest < tol) ≈ 2·Φ(tol·rest / σ_edge) − 1
      // where σ_edge ≈ √2·σ̄ from the variance sum of two independent endpoints.
      let sigMean = 0;
      let pConstraint = 1;
      if (s.K > 0) {
        let sumVar = 0;
        for (let i = 0; i < s.N; i++) {
          let acc = 0;
          for (let kk = 0; kk < s.K; kk++) {
            const o = kk * s.N * 2 + i * 2;
            const dx = s.ensX[o], dy = s.ensX[o + 1];
            acc += dx * dx + dy * dy;
          }
          sumVar += acc / (2 * s.K);
        }
        sigMean = Math.sqrt(sumVar / Math.max(1, s.N));
        if (s.E > 0 && p.constraintTol > 0) {
          const restMean = p.restLength;
          const sigEdge = Math.SQRT2 * sigMean;
          const z = (p.constraintTol * restMean) / Math.max(1e-6, sigEdge);
          pConstraint = 2 * normCdf(z) - 1;
        }
      }

      const fmtPct = (n: number) => (n * 100).toFixed(2) + "%";
      // Stability classification — only meaningful for relative drift.
      // Symplectic methods should keep |Δ|/|E₀| bounded (≲ 1%); explicit
      // Euler typically grows monotonically and crosses these thresholds.
      const absRel = Math.abs(relDrift);
      const stabLabel =
        absRel < 0.01 ? "stable"   :
        absRel < 0.05 ? "drifting" :
        absRel < 0.20 ? "unstable" : "diverging";
      const lines = [
        `diagnostics · ${p.integrator}`,
        `KE        ${fmt(KE)}`,
        `PE grav   ${fmt(PE_grav)}`,
        `PE spring ${fmt(PE_spring)}`,
        `PE field  ${fmt(PE_field)}`,
        `── total  ${fmt(E_total)}`,
        `Δ since   ${drift >= 0 ? "+" : ""}${fmt(drift)}`,
        `Δ/E₀      ${(relDrift >= 0 ? "+" : "")}${(relDrift * 100).toFixed(3)}%`,
        `|Δ| ema   ${fmt(driftAbsEmaRef.current)}`,
        `Δ rms     ${fmt(driftRms)}`,
        `stability ${stabLabel}`,
        `c·err max ${fmtPct(cMax)}`,
        `c·err rms ${fmtPct(cRms)}`,
        `subSteps  ${subStepsEff}${p.adaptiveSubSteps ? " (auto)" : ""}`,
        `cIters    ${p.constraintIters | 0}`,
        `fps       ${fps.toFixed(1)}`,
        `MC K      ${s.K}`,
        `σ̄ (px)    ${s.K > 0 ? sigMean.toFixed(2) : "—"}`,
        `P(c≤${(p.constraintTol*100).toFixed(1)}%)  ${s.K > 0 ? (pConstraint*100).toFixed(1)+"%" : "—"}`,
        `twin M    ${p.twinEnabled ? twinSensorsRef.current.length : 0}`,
        `res EMA   ${p.twinEnabled ? twinResidualEmaRef.current.toFixed(1)+"px" : "—"}`,
        `anomalies ${p.twinEnabled ? twinAnomalyCountRef.current : "—"}`,
      ];
      const padX = 10, padY = 8, lineH = 14;
      const panelW = 200;
      // Sparkline plotted under the text lines: shows Δ vs. baseline over
      // the recent history window, with a zero reference line. Auto-scaled
      // to peak |Δ| in the window so both stable & diverging look right.
      const sparkH = 42;
      const sparkPadTop = 6;
      const panelH = padY * 2 + lineH * lines.length + sparkPadTop + sparkH;
      const panelX = w - panelW - 12;
      const panelY = 12;
      ctx.fillStyle = "oklch(0.16 0.02 260 / 0.82)";
      ctx.fillRect(panelX, panelY, panelW, panelH);
      ctx.strokeStyle = "oklch(0.4 0.05 260 / 0.6)";
      ctx.lineWidth = 1;
      ctx.strokeRect(panelX + 0.5, panelY + 0.5, panelW - 1, panelH - 1);
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textBaseline = "top";
      // Color thresholds:
      //   constraint error:  green  < 1%, amber 1-5%, red > 5%
      //   fps:               green ≥ 50, amber 30-50, red < 30
      //   stability:         green stable, amber drifting, red unstable+
      const cColor = (v: number) =>
        v < 0.01 ? "oklch(0.82 0.18 150)" : v < 0.05 ? "oklch(0.84 0.16 85)" : "oklch(0.78 0.20 35)";
      const fpsColor = fps >= 50 ? "oklch(0.82 0.18 150)" : fps >= 30 ? "oklch(0.84 0.16 85)" : "oklch(0.78 0.20 35)";
      const stabColor =
        absRel < 0.01 ? "oklch(0.82 0.18 150)" :
        absRel < 0.05 ? "oklch(0.84 0.16 85)"  : "oklch(0.78 0.20 35)";
      for (let li = 0; li < lines.length; li++) {
        if (li === 0)       ctx.fillStyle = "oklch(0.78 0.14 230)";
        else if (li === 5)  ctx.fillStyle = "oklch(0.94 0.04 230)";
        else if (li === 6)  ctx.fillStyle = drift >= 0 ? "oklch(0.78 0.18 35)" : "oklch(0.78 0.18 150)";
        else if (li === 7)  ctx.fillStyle = stabColor;
        else if (li === 8)  ctx.fillStyle = "oklch(0.84 0.10 230 / 0.9)";
        else if (li === 9)  ctx.fillStyle = "oklch(0.84 0.10 230 / 0.9)";
        else if (li === 10) ctx.fillStyle = stabColor;
        else if (li === 11) ctx.fillStyle = cColor(cMax);
        else if (li === 12) ctx.fillStyle = cColor(cRms);
        else if (li === 15) ctx.fillStyle = fpsColor;
        else                ctx.fillStyle = "oklch(0.78 0.04 230 / 0.85)";
        ctx.fillText(lines[li], panelX + padX, panelY + padY + li * lineH);
      }

      // ── Energy-drift sparkline ───────────────────────────────────
      // X-axis: oldest sample on the left → newest on the right.
      // Y-axis: signed Δ, centered on zero, scaled to ±max(|Δ|) in window.
      {
        const sx = panelX + padX;
        const sy = panelY + padY + lines.length * lineH + sparkPadTop;
        const sw = panelW - padX * 2;
        const sh = sparkH;
        // Background + zero line
        ctx.fillStyle = "oklch(0.20 0.02 260 / 0.6)";
        ctx.fillRect(sx, sy, sw, sh);
        ctx.strokeStyle = "oklch(0.5 0.03 260 / 0.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sx, sy + sh / 2 + 0.5);
        ctx.lineTo(sx + sw, sy + sh / 2 + 0.5);
        ctx.stroke();

        const len = energyHistLenRef.current;
        if (len >= 2) {
          // Read oldest→newest by walking from (head - len) mod cap.
          const cap = ENERGY_HIST_CAP;
          const start = (energyHistHeadRef.current - len + cap) % cap;
          let dMax = 1e-12;
          for (let i = 0; i < len; i++) {
            const v = Math.abs(driftHistRef.current[(start + i) % cap]);
            if (v > dMax) dMax = v;
          }
          ctx.strokeStyle = stabColor;
          ctx.lineWidth = 1.25;
          ctx.beginPath();
          for (let i = 0; i < len; i++) {
            const d = driftHistRef.current[(start + i) % cap];
            const px = sx + (i / (len - 1)) * sw;
            const py = sy + sh / 2 - (d / dMax) * (sh / 2 - 2);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.stroke();
          // Peak-|Δ| label (top-right of sparkline)
          ctx.fillStyle = "oklch(0.78 0.04 230 / 0.7)";
          ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
          ctx.fillText(`±${fmt(dMax)}`, sx + 4, sy + 2);
          ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
        }
      }

      if (p.showEdges && s.E > 0) {
        ctx.lineWidth = 0.6;
        ctx.beginPath();
        for (let e = 0; e < s.E; e++) {
          const i = s.edges[e * 2];
          const j = s.edges[e * 2 + 1];
          ctx.moveTo(s.x[i * 2], s.x[i * 2 + 1]);
          ctx.lineTo(s.x[j * 2], s.x[j * 2 + 1]);
        }
        ctx.strokeStyle = "oklch(0.78 0.16 280 / 0.25)";
        ctx.stroke();
      }

      // Render nodes — tinted by partition when showPartitions is on,
      // or by force magnitude when forceViz === "heatmap".
      // s.f still holds the last sub-step's accumulated force (sim doesn't
      // zero it after step → great free signal for visualization).
      const W = Math.max(1, Math.min(p.workers | 0, s.N));

      // Compute force magnitudes once if we need them
      let fmagMax = 1;
      if (p.forceViz !== "off") {
        for (let i = 0; i < s.N; i++) {
          const fm = Math.hypot(s.f[i * 2], s.f[i * 2 + 1]);
          if (fm > fmagMax) fmagMax = fm;
        }
      }

      for (let i = 0; i < s.N; i++) {
        const sp = Math.hypot(s.v[i * 2], s.v[i * 2 + 1]);
        const radius = 1.5 + s.m[i] * 1.6;
        const q = Math.min(W - 1, Math.floor((i * W) / s.N));

        let hueDeg: number;
        let chroma = 0.18;
        if (p.forceViz === "heatmap") {
          // viridis-ish: low force = deep blue/purple, high = bright yellow
          const t = Math.min(1, Math.hypot(s.f[i * 2], s.f[i * 2 + 1]) / fmagMax);
          hueDeg = 280 - 200 * t;
          chroma = 0.10 + 0.18 * t;
        } else if (p.showPartitions) {
          hueDeg = (q * 360) / Math.max(1, W);
          chroma = 0.22;
        } else {
          hueDeg = (s.hue[i] * 80 + 140) % 360;
        }
        const light = Math.min(0.92, 0.55 + sp / 600);
        ctx.beginPath();
        ctx.arc(s.x[i * 2], s.x[i * 2 + 1], radius, 0, Math.PI * 2);
        ctx.fillStyle = `oklch(${light} ${chroma} ${hueDeg})`;
        ctx.fill();
      }

      // Force vectors — one short arrow per particle, length ∝ |f|/fmax.
      // Drawn after the dots so the tails stay visible.
      if (p.forceViz === "vectors" && fmagMax > 0) {
        const VEC_PX = 22; // max arrow length in pixels
        ctx.strokeStyle = "oklch(0.88 0.18 95 / 0.7)";
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        for (let i = 0; i < s.N; i++) {
          const fxv = s.f[i * 2], fyv = s.f[i * 2 + 1];
          const fm = Math.hypot(fxv, fyv);
          if (fm < 1e-3) continue;
          const k2 = (VEC_PX * fm / fmagMax) / fm;
          const x0 = s.x[i * 2], y0 = s.x[i * 2 + 1];
          const x1 = x0 + fxv * k2, y1 = y0 + fyv * k2;
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          // arrowhead
          const ang = Math.atan2(y1 - y0, x1 - x0);
          const ah = 3;
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ah * Math.cos(ang - 0.4), y1 - ah * Math.sin(ang - 0.4));
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ah * Math.cos(ang + 0.4), y1 - ah * Math.sin(ang + 0.4));
        }
        ctx.stroke();
      }

      // Field-direction arrows: −∇Φ sampled at each particle, independent of
      // the resultant force shown by forceViz=="vectors".
      if (p.showFieldArrows && p.field !== "none") {
        const tNow = (now - tStartRef.current) / 1000;
        const samp = resolveSampling(p.fieldSampling, p.boundary);
        const eps = 0.5 * (Math.max(w, h) / 800);
        // First pass: compute gradients & find max magnitude for normalization
        const gx = new Float32Array(s.N);
        const gy = new Float32Array(s.N);
        let gmax = 1e-6;
        for (let i = 0; i < s.N; i++) {
          const [px, py] = sampleCoords(samp, s.x[i * 2], s.x[i * 2 + 1], w, h);
          let dx: number, dy: number;
          const ga = fieldGradAnalytic(p.field, px, py, w, h);
          if (ga) {
            dx = ga[0]; dy = ga[1];
          } else {
            dx = (fieldPotential(p.field, px + eps, py, w, h, customFnRef.current, tNow) - fieldPotential(p.field, px - eps, py, w, h, customFnRef.current, tNow)) / (2 * eps);
            dy = (fieldPotential(p.field, px, py + eps, w, h, customFnRef.current, tNow) - fieldPotential(p.field, px, py - eps, w, h, customFnRef.current, tNow)) / (2 * eps);
          }
          // arrow points along −∇Φ (descent direction = force direction)
          gx[i] = -dx;
          gy[i] = -dy;
          const m = Math.hypot(gx[i], gy[i]);
          if (m > gmax) gmax = m;
        }
        const VEC_PX = 18;
        ctx.strokeStyle = "oklch(0.82 0.20 200 / 0.75)";
        ctx.lineWidth = 0.7;
        ctx.beginPath();
        for (let i = 0; i < s.N; i++) {
          const m = Math.hypot(gx[i], gy[i]);
          if (m < 1e-6) continue;
          const k2 = (VEC_PX * (m / gmax)) / m;
          const x0 = s.x[i * 2], y0 = s.x[i * 2 + 1];
          const x1 = x0 + gx[i] * k2, y1 = y0 + gy[i] * k2;
          ctx.moveTo(x0, y0);
          ctx.lineTo(x1, y1);
          const ang = Math.atan2(y1 - y0, x1 - x0);
          const ah = 3;
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ah * Math.cos(ang - 0.4), y1 - ah * Math.sin(ang - 0.4));
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - ah * Math.cos(ang + 0.4), y1 - ah * Math.sin(ang + 0.4));
        }
        ctx.stroke();
      }

      // ── Confidence ellipses (zσ contour of the per-particle MC cloud) ──
      // Reduce δx_k → 2×2 covariance, eigen-decompose closed-form, draw
      // an ellipse with semi-axes z·√λ. Skipped at K<2 (no variance).
      if (p.showConfidence && s.K >= 2) {
        const z = Math.max(0.1, p.confidenceZ);
        ctx.lineWidth = 0.8;
        ctx.strokeStyle = "oklch(0.86 0.16 200 / 0.55)";
        for (let i = 0; i < s.N; i++) {
          let sxx = 0, syy = 0, sxy = 0;
          for (let kk = 0; kk < s.K; kk++) {
            const o = kk * s.N * 2 + i * 2;
            const dx = s.ensX[o], dy = s.ensX[o + 1];
            sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
          }
          const invK = 1 / s.K;
          sxx *= invK; syy *= invK; sxy *= invK;
          // closed-form eigenvalues of [[sxx, sxy],[sxy, syy]]
          const tr = sxx + syy;
          const det = sxx * syy - sxy * sxy;
          const disc = Math.max(0, tr * tr * 0.25 - det);
          const root = Math.sqrt(disc);
          const l1 = tr * 0.5 + root;
          const l2 = Math.max(0, tr * 0.5 - root);
          if (l1 < 1e-4) continue;
          const a = z * Math.sqrt(l1);
          const b = z * Math.sqrt(l2);
          // angle of dominant eigenvector
          const ang = Math.abs(sxy) < 1e-9 && Math.abs(sxx - syy) < 1e-9
            ? 0
            : Math.atan2(2 * sxy, sxx - syy) * 0.5;
          ctx.beginPath();
          ctx.ellipse(s.x[i * 2], s.x[i * 2 + 1], a, b, ang, 0, Math.PI * 2);
          ctx.stroke();
        }
      }

      // ── Twin overlay: sensor crosses, residual lines, forecast trails ──
      if (p.showTwin && p.twinEnabled && twinSensorsRef.current.length > 0) {
        const sensors = twinSensorsRef.current;
        // 1) residual line (sim → sensor) — color = anomaly state
        ctx.lineWidth = 0.8;
        for (let i = 0; i < sensors.length; i++) {
          const sn = sensors[i];
          if (sn.bound >= s.N) continue;
          const x0 = s.x[sn.bound * 2], y0 = s.x[sn.bound * 2 + 1];
          const ok = sn.z <= p.twinAnomalyZ;
          ctx.strokeStyle = ok ? "oklch(0.78 0.12 200 / 0.55)" : "oklch(0.72 0.22 30 / 0.85)";
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.lineTo(sn.px, sn.py);
          ctx.stroke();
          // 2) sensor cross-hair
          ctx.strokeStyle = ok ? "oklch(0.86 0.16 200 / 0.9)" : "oklch(0.78 0.22 30)";
          ctx.lineWidth = 1.1;
          ctx.beginPath();
          ctx.moveTo(sn.px - 5, sn.py); ctx.lineTo(sn.px + 5, sn.py);
          ctx.moveTo(sn.px, sn.py - 5); ctx.lineTo(sn.px, sn.py + 5);
          ctx.stroke();
          // 3) noise circle (1σ)
          ctx.strokeStyle = "oklch(0.72 0.10 200 / 0.35)";
          ctx.lineWidth = 0.5;
          ctx.beginPath();
          ctx.arc(sn.px, sn.py, p.twinSensorNoise, 0, Math.PI * 2);
          ctx.stroke();
        }
        // 4) ballistic forecast for each bound particle (linear: x + v·dt·k)
        const F = Math.max(0, Math.min(60, p.twinForecastSteps | 0));
        if (F > 0) {
          const fdt = dt > 1e-6 ? dt : 1 / 60;
          ctx.lineWidth = 0.6;
          for (let i = 0; i < sensors.length; i++) {
            const sn = sensors[i];
            if (sn.bound >= s.N) continue;
            const i2 = sn.bound * 2;
            const x0 = s.x[i2], y0 = s.x[i2 + 1];
            const vx = s.v[i2], vy = s.v[i2 + 1];
            for (let k = 1; k <= F; k++) {
              const a = 0.5 * (1 - k / F);
              ctx.strokeStyle = `oklch(0.82 0.14 95 / ${a.toFixed(3)})`;
              ctx.beginPath();
              ctx.moveTo(x0 + vx * fdt * (k - 1), y0 + vy * fdt * (k - 1));
              ctx.lineTo(x0 + vx * fdt * k, y0 + vy * fdt * k);
              ctx.stroke();
            }
          }
        }
      }

      // ── Debug force overlay ───────────────────────────────────────────
      // Independent of forceViz/showFieldArrows. Draws TWO arrows per
      // particle: gravity (red, body force only) and net force (yellow,
      // includes everything in s.f). Both are normalized to the per-frame
      // max-net-force so their relative magnitudes are directly readable.
      // A HUD in the top-left shows |F| stats and a numeric readout is
      // rendered next to a sparse sample of particles.
      if (p.debugForces) {
        const gMode = p.gravityMode ?? "uniform";
        let gx = 0, gy = 0;
        if (gMode !== "zero" && p.gravity !== 0) {
          if (gMode === "directional") {
            const ang = ((p.gravityAngle ?? 90) * Math.PI) / 180;
            gx = Math.cos(ang) * p.gravity;
            gy = Math.sin(ang) * p.gravity;
          } else {
            gy = p.gravity;
          }
        }
        // stats over net force
        let fMax = 1e-6, fSum = 0, fMin = Infinity;
        for (let i = 0; i < s.N; i++) {
          const fm = Math.hypot(s.f[i * 2], s.f[i * 2 + 1]);
          if (fm > fMax) fMax = fm;
          if (fm < fMin) fMin = fm;
          fSum += fm;
        }
        const fMean = fSum / Math.max(1, s.N);
        const NET_PX = 26;
        const GRAV_PX = 18;
        const gMag = Math.hypot(gx, gy);
        // Net-force arrows (yellow)
        ctx.strokeStyle = "oklch(0.88 0.18 95 / 0.85)";
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        for (let i = 0; i < s.N; i++) {
          const fxv = s.f[i * 2], fyv = s.f[i * 2 + 1];
          const fm = Math.hypot(fxv, fyv);
          if (fm < 1e-3) continue;
          const k2 = (NET_PX * fm / fMax) / fm;
          const x0 = s.x[i * 2], y0 = s.x[i * 2 + 1];
          const x1 = x0 + fxv * k2, y1 = y0 + fyv * k2;
          ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
          const ang = Math.atan2(y1 - y0, x1 - x0);
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - 3 * Math.cos(ang - 0.4), y1 - 3 * Math.sin(ang - 0.4));
          ctx.moveTo(x1, y1);
          ctx.lineTo(x1 - 3 * Math.cos(ang + 0.4), y1 - 3 * Math.sin(ang + 0.4));
        }
        ctx.stroke();
        // Gravity arrows (red), per-particle: gᵢ = g · mᵢ
        if (gMag > 1e-6) {
          ctx.strokeStyle = "oklch(0.70 0.22 25 / 0.85)";
          ctx.lineWidth = 0.7;
          ctx.beginPath();
          for (let i = 0; i < s.N; i++) {
            const m = s.m[i];
            const gxi = gx * m, gyi = gy * m;
            const gm = Math.hypot(gxi, gyi);
            if (gm < 1e-3) continue;
            const k2 = (GRAV_PX * gm / fMax) / gm;
            const x0 = s.x[i * 2], y0 = s.x[i * 2 + 1];
            const x1 = x0 + gxi * k2, y1 = y0 + gyi * k2;
            ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
            const ang = Math.atan2(y1 - y0, x1 - x0);
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1 - 2.5 * Math.cos(ang - 0.4), y1 - 2.5 * Math.sin(ang - 0.4));
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1 - 2.5 * Math.cos(ang + 0.4), y1 - 2.5 * Math.sin(ang + 0.4));
          }
          ctx.stroke();
        }
        // Per-particle magnitude labels (sparse: every Nth particle)
        const labelStride = Math.max(1, Math.ceil(s.N / 24));
        ctx.fillStyle = "oklch(0.95 0.02 95 / 0.85)";
        ctx.font = "9px ui-monospace, monospace";
        for (let i = 0; i < s.N; i += labelStride) {
          const fm = Math.hypot(s.f[i * 2], s.f[i * 2 + 1]);
          ctx.fillText(fm.toFixed(0), s.x[i * 2] + 4, s.x[i * 2 + 1] - 4);
        }
        // HUD
        const hudW = 196, hudH = 64;
        ctx.fillStyle = "oklch(0.16 0.02 260 / 0.78)";
        ctx.fillRect(8, 8, hudW, hudH);
        ctx.strokeStyle = "oklch(0.88 0.18 95 / 0.5)";
        ctx.lineWidth = 0.6;
        ctx.strokeRect(8, 8, hudW, hudH);
        ctx.fillStyle = "oklch(0.95 0.02 95 / 0.95)";
        ctx.font = "10px ui-monospace, monospace";
        ctx.fillText("DEBUG · forces", 16, 22);
        ctx.fillStyle = "oklch(0.88 0.18 95 / 0.95)";
        ctx.fillText(`|F| min ${fMin === Infinity ? 0 : fMin.toFixed(1)}  mean ${fMean.toFixed(1)}  max ${fMax.toFixed(1)}`, 16, 38);
        ctx.fillStyle = "oklch(0.70 0.22 25 / 0.95)";
        ctx.fillText(`|g·m̄| ≈ ${(gMag * 1.2).toFixed(1)}  mode=${gMode}`, 16, 54);
        ctx.fillStyle = "oklch(0.95 0.02 95 / 0.6)";
        ctx.font = "9px ui-monospace, monospace";
        ctx.fillText("yellow = net F   red = gravity", 16, 68);
      }


      if (pointerRef.current.active) {
        const sign = pointerRef.current.mode;
        ctx.strokeStyle = sign > 0 ? "oklch(0.82 0.18 165 / 0.8)" : "oklch(0.72 0.20 35 / 0.8)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(pointerRef.current.x, pointerRef.current.y, 28, 0, Math.PI * 2);
        ctx.stroke();
      }

      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [pointerRef]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 h-full w-full"
      onPointerMove={(e) => {
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        pointerRef.current.x = e.clientX - r.left;
        pointerRef.current.y = e.clientY - r.top;
      }}
      onPointerDown={(e) => {
        const r = (e.target as HTMLCanvasElement).getBoundingClientRect();
        pointerRef.current.x = e.clientX - r.left;
        pointerRef.current.y = e.clientY - r.top;
        pointerRef.current.active = true;
        pointerRef.current.mode = e.button === 2 ? -1 : 1;
      }}
      onPointerUp={() => { pointerRef.current.active = false; }}
      onPointerLeave={() => { pointerRef.current.active = false; }}
      onContextMenu={(e) => e.preventDefault()}
    />
  );
}
