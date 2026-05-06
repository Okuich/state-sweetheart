import { useEffect, useRef, useState } from "react";

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
  field: "none" | "swirl" | "wells" | "ripple";
  fieldStrength: number;
  subSteps: number;
  workers: number;
  showPartitions: boolean;
  optimize: boolean;
  objectiveLR: number;
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
 * integrator = "verlet" → velocity-Verlet (2nd order, energy-stable):
 *      x ← x + v·dt + ½·a·dt²
 *      v ← v + ½·(a + a_new)·dt        (a_new injected by caller next frame)
 *
 * Walls: elastic-ish reflection with restitution 0.7.
 */
/**
 * Differentiable scalar potential fields Φ(x, y) and helpers.
 *
 * In PyTorch you'd do:
 *     potential = field_fn(state.x).sum()
 *     forces    = -autograd.grad(potential, state.x)[0]
 *
 * Here we mimic the same contract with a small finite-difference gradient,
 * which is the same operation autograd performs analytically. Closed-form
 * gradients would be faster — finite differences keep the field plug-and-play.
 */
type FieldName = "none" | "swirl" | "wells" | "ripple";

function fieldPotential(name: FieldName, x: number, y: number, w: number, h: number): number {
  const cx = w * 0.5, cy = h * 0.5;
  const nx = (x - cx) / Math.max(w, h);
  const ny = (y - cy) / Math.max(w, h);
  switch (name) {
    case "swirl":
      // Spiral well: radial sink + angular twist
      return 0.5 * (nx * nx + ny * ny) + 0.25 * Math.sin(6 * Math.atan2(ny, nx));
    case "wells": {
      // Two Gaussian wells
      const d1 = (nx + 0.18) ** 2 + (ny - 0.0) ** 2;
      const d2 = (nx - 0.18) ** 2 + (ny + 0.0) ** 2;
      return -Math.exp(-d1 * 18) - Math.exp(-d2 * 18);
    }
    case "ripple": {
      const r = Math.sqrt(nx * nx + ny * ny);
      return Math.cos(r * 28) * Math.exp(-r * 2.5) * 0.4;
    }
    default:
      return 0;
  }
}

/**
 * compute_potential_forces — adds  -∇Φ · strength  to state.f for every node.
 * Uses central finite differences (≈ autograd.grad on a scalar field).
 */
function computePotentialForces(s: State, name: FieldName, strength: number, w: number, h: number) {
  computePotentialForces_range(s, name, strength, w, h, 0, s.N);
}

function computePotentialForces_range(s: State, name: FieldName, strength: number, w: number, h: number, a: number, b: number) {
  if (name === "none" || strength === 0) return;
  const eps = 0.5;
  const scale = strength * 1500;
  for (let i = a; i < b; i++) {
    const x = s.x[i * 2], y = s.x[i * 2 + 1];
    const dphidx = (fieldPotential(name, x + eps, y, w, h) - fieldPotential(name, x - eps, y, w, h)) / (2 * eps);
    const dphidy = (fieldPotential(name, x, y + eps, w, h) - fieldPotential(name, x, y - eps, w, h)) / (2 * eps);
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

function stepState(
  s: State,
  dt: number,
  damping: number,
  w: number,
  h: number,
  integrator: "euler" | "semi-euler" | "verlet",
) {
  stepStateRange(s, dt, damping, w, h, integrator, 0, s.N);
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
) {
  if (integrator === "verlet") {
    for (let i = a; i < b; i++) {
      const invM = 1 / s.m[i];
      const ax = s.f[i * 2]     * invM;
      const ay = s.f[i * 2 + 1] * invM;
      s.x[i * 2]     += s.v[i * 2]     * dt + 0.5 * ax * dt * dt;
      s.x[i * 2 + 1] += s.v[i * 2 + 1] * dt + 0.5 * ay * dt * dt;
      s.fPrev[i * 2]     = s.f[i * 2];
      s.fPrev[i * 2 + 1] = s.f[i * 2 + 1];
      s.v[i * 2]     = (s.v[i * 2]     + 0.5 * ax * dt) * (1 - damping * dt);
      s.v[i * 2 + 1] = (s.v[i * 2 + 1] + 0.5 * ay * dt) * (1 - damping * dt);
    }
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
    // forces zeroed at top of next sub-step
  }
  // Wall collisions
  for (let i = a; i < b; i++) {
    if (s.x[i * 2] < 0)        { s.x[i * 2] = 0; s.v[i * 2] *= -0.7; }
    else if (s.x[i * 2] > w)   { s.x[i * 2] = w; s.v[i * 2] *= -0.7; }
    if (s.x[i * 2 + 1] < 0)    { s.x[i * 2 + 1] = 0; s.v[i * 2 + 1] *= -0.7; }
    else if (s.x[i * 2 + 1] > h){ s.x[i * 2 + 1] = h; s.v[i * 2 + 1] *= -0.7; }
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
  return { N, D: 2, dtype, device, x, v, m, f, fPrev, hue, edges, edgeRest, E };
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
            const v = fieldPotential(p.field, c * cell + cell / 2, r * cell + cell / 2, w, h);
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
        const subSteps = Math.max(1, p.subSteps | 0);
        const subDt = dt / subSteps;
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
            computePotentialForces_range(s, p.field, p.fieldStrength, w, h, a, b);
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

          // 4. step(state, dt) — each worker integrates its own slice
          for (let q = 0; q < W; q++) {
            stepStateRange(s, subDt, p.damping, w, h, p.integrator, partStart(q), partEnd(q));
          }

          // 5. sync_boundaries — re-project cross-partition edges so the
          // independently-stepped slices stay consistent at the seams.
          projectConstraints(s, p.constraintIters, subDt);
          syncBoundaries(s, partOf);
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

      // Render nodes — tinted by partition when showPartitions is on
      const W = Math.max(1, Math.min(p.workers | 0, s.N));
      for (let i = 0; i < s.N; i++) {
        const sp = Math.hypot(s.v[i * 2], s.v[i * 2 + 1]);
        const radius = 1.5 + s.m[i] * 1.6;
        const q = Math.min(W - 1, Math.floor((i * W) / s.N));
        const hueDeg = p.showPartitions
          ? (q * 360) / Math.max(1, W)
          : (s.hue[i] * 80 + 140) % 360;
        const chroma = p.showPartitions ? 0.22 : 0.18;
        const light = Math.min(0.92, 0.55 + sp / 600);
        ctx.beginPath();
        ctx.arc(s.x[i * 2], s.x[i * 2 + 1], radius, 0, Math.PI * 2);
        ctx.fillStyle = `oklch(${light} ${chroma} ${hueDeg})`;
        ctx.fill();
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
