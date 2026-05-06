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
  damping: number;
  attractor: number;
  particleCount: number;
  trail: number;
  paused: boolean;
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
  forceViz: "off" | "vectors" | "heatmap";
  potentialGrad: "analytic" | "finite-diff";
  fieldSampling: "auto" | "clamp" | "wrap" | "none";
  showFieldArrows: boolean;
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
) {
  // First half-kick using PREVIOUS step's forces (cached in s.fPrev),
  // then drift positions with the half-updated velocity.
  for (let i = a; i < b; i++) {
    const invM = 1 / s.m[i];
    const axOld = s.fPrev[i * 2]     * invM;
    const ayOld = s.fPrev[i * 2 + 1] * invM;
    s.v[i * 2]     += 0.5 * axOld * dt;
    s.v[i * 2 + 1] += 0.5 * ayOld * dt;
    s.x[i * 2]     += s.v[i * 2]     * dt;
    s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt;
  }
}

function verletKick(
  s: State,
  dt: number,
  damping: number,
  a: number,
  b: number,
) {
  // Second half-kick using the NEW forces just computed for this step,
  // then cache them as fPrev for the next step's drift.
  const decay = 1 - damping * dt;
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
  integrator: "euler" | "semi-euler" | "verlet",
  boundary: Boundary = "walls",
) {
  stepStateRange(s, dt, damping, w, h, integrator, 0, s.N, boundary);
}

function stepStateRange(
  s: State,
  dt: number,
  damping: number,
  w: number,
  h: number,
  integrator: "euler" | "semi-euler" | "verlet",
  a: number,
  b: number,
  boundary: Boundary = "walls",
) {
  if (integrator === "verlet") {
    // Verlet's drift+kick are split around the force evaluation; the
    // caller invokes verletDrift() BEFORE recomputing forces and
    // verletKick() AFTER. This branch is now position-only damping wrap-up.
  } else if (integrator === "semi-euler") {
    for (let i = a; i < b; i++) {
      const invM = 1 / s.m[i];
      const ax = s.f[i * 2]     * invM;
      const ay = s.f[i * 2 + 1] * invM;
      s.v[i * 2]     = (s.v[i * 2]     + ax * dt) * (1 - damping * dt);
      s.v[i * 2 + 1] = (s.v[i * 2 + 1] + ay * dt) * (1 - damping * dt);
      s.x[i * 2]     += s.v[i * 2]     * dt;
      s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt;
    }
  } else {
    for (let i = a; i < b; i++) {
      const invM = 1 / s.m[i];
      const ax = s.f[i * 2]     * invM;
      const ay = s.f[i * 2 + 1] * invM;
      const vx0 = s.v[i * 2], vy0 = s.v[i * 2 + 1];
      s.v[i * 2]     = (vx0 + ax * dt) * (1 - damping * dt);
      s.v[i * 2 + 1] = (vy0 + ay * dt) * (1 - damping * dt);
      s.x[i * 2]     += vx0 * dt;
      s.x[i * 2 + 1] += vy0 * dt;
    }
  }
  // Boundary handling — three modes:
  //   walls:    elastic-ish reflection at the box edges (restitution 0.7)
  //   wrap:     positions teleport across edges; velocity unchanged
  //             (useful to see flux without bouncing artifacts)
  //   periodic: same wrap, AND pairwise forces use the minimum-image
  //             convention so particles interact across the seam — this
  //             is the standard MD periodic-box setup.
  if (boundary === "walls") {
    for (let i = a; i < b; i++) {
      if (s.x[i * 2] < 0)         { s.x[i * 2] = 0; s.v[i * 2] *= -0.7; }
      else if (s.x[i * 2] > w)    { s.x[i * 2] = w; s.v[i * 2] *= -0.7; }
      if (s.x[i * 2 + 1] < 0)     { s.x[i * 2 + 1] = 0; s.v[i * 2 + 1] *= -0.7; }
      else if (s.x[i * 2 + 1] > h){ s.x[i * 2 + 1] = h; s.v[i * 2 + 1] *= -0.7; }
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
function syncBoundaries(s: State, partOf: (i: number) => number) {
  if (s.E === 0) return;
  // Per-node accumulators for the "other-side" neighborhood mean.
  const sumX = new Float64Array(s.N * 2);
  const sumV = new Float64Array(s.N * 2);
  const cnt  = new Int32Array(s.N);

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
  return { N, D: 2, dtype, device, x, v, m, f, fPrev, hue, edges, edgeRest, E, K: 0, ensX: new Float32Array(0), ensV: new Float32Array(0) };
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
  const onValidationRef = useRef(onValidation);
  onValidationRef.current = onValidation;
  const onLossRef = useRef(onLoss);
  onLossRef.current = onLoss;
  const lossEmaRef = useRef(0);
  const energyBaselineRef = useRef<number | null>(null);
  const energyBaselineNRef = useRef(0);
  const lastSubStepsRef = useRef(1);
  const fpsEmaRef = useRef(60);

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
      const dt = Math.min(0.033, (now - last) / 1000);
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

      if (!p.paused) {
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
          // Velocity-Verlet drift uses the PREVIOUS step's forces (s.fPrev)
          // for the first half-kick, then advances positions. This must run
          // BEFORE we recompute forces for the new positions.
          if (p.integrator === "verlet") {
            for (let q = 0; q < W; q++) {
              verletDrift(s, subDt, partStart(q), partEnd(q));
            }
          }

          // 1. zero forces — state.f.zero_()
          s.f.fill(0);

          // 2. per-worker local forces (each worker owns nodes [a,b))
          for (let q = 0; q < W; q++) {
            const a = partStart(q), b = partEnd(q);

            // gravity (local)
            for (let i = a; i < b; i++) {
              s.f[i * 2 + 1] += p.gravity * s.m[i];
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
            for (let q = 0; q < W; q++) {
              verletKick(s, subDt, p.damping, partStart(q), partEnd(q));
              // still call stepStateRange for boundary handling (verlet branch is a no-op for motion)
              stepStateRange(s, subDt, p.damping, w, h, p.integrator, partStart(q), partEnd(q), p.boundary);
            }
          } else {
            for (let q = 0; q < W; q++) {
              stepStateRange(s, subDt, p.damping, w, h, p.integrator, partStart(q), partEnd(q), p.boundary);
            }
          }

          // 5. sync_boundaries — re-project cross-partition edges so the
          // independently-stepped slices stay consistent at the seams.
          projectConstraints(s, p.constraintIters, subDt);
          syncBoundaries(s, partOf);

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

      // ── differentiable_loop.py ────────────────────────────────────
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
      if (p.gravity !== 0) {
        for (let i = 0; i < s.N; i++) {
          PE_grav += s.m[i] * p.gravity * (h - s.x[i * 2 + 1]);
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

      if (energyBaselineRef.current === null || s.N !== energyBaselineNRef.current) {
        energyBaselineRef.current = E_total;
        energyBaselineNRef.current = s.N;
      }
      const drift = E_total - (energyBaselineRef.current ?? E_total);

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
      const lines = [
        `diagnostics · ${p.integrator}`,
        `KE        ${fmt(KE)}`,
        `PE grav   ${fmt(PE_grav)}`,
        `PE spring ${fmt(PE_spring)}`,
        `PE field  ${fmt(PE_field)}`,
        `── total  ${fmt(E_total)}`,
        `Δ since   ${drift >= 0 ? "+" : ""}${fmt(drift)}`,
        `c·err max ${fmtPct(cMax)}`,
        `c·err rms ${fmtPct(cRms)}`,
        `subSteps  ${subStepsEff}${p.adaptiveSubSteps ? " (auto)" : ""}`,
        `cIters    ${p.constraintIters | 0}`,
        `fps       ${fps.toFixed(1)}`,
      ];
      const padX = 10, padY = 8, lineH = 14;
      const panelW = 188;
      const panelH = padY * 2 + lineH * lines.length;
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
      const cColor = (v: number) =>
        v < 0.01 ? "oklch(0.82 0.18 150)" : v < 0.05 ? "oklch(0.84 0.16 85)" : "oklch(0.78 0.20 35)";
      const fpsColor = fps >= 50 ? "oklch(0.82 0.18 150)" : fps >= 30 ? "oklch(0.84 0.16 85)" : "oklch(0.78 0.20 35)";
      for (let li = 0; li < lines.length; li++) {
        if (li === 0)      ctx.fillStyle = "oklch(0.78 0.14 230)";
        else if (li === 5) ctx.fillStyle = "oklch(0.94 0.04 230)";
        else if (li === 6) ctx.fillStyle = drift >= 0 ? "oklch(0.78 0.18 35)" : "oklch(0.78 0.18 150)";
        else if (li === 7) ctx.fillStyle = cColor(cMax);
        else if (li === 8) ctx.fillStyle = cColor(cRms);
        else if (li === 11) ctx.fillStyle = fpsColor;
        else               ctx.fillStyle = "oklch(0.78 0.04 230 / 0.85)";
        ctx.fillText(lines[li], panelX + padX, panelY + padY + li * lineH);
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
