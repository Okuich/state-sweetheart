// Unified Goal-Driven Optimization Engine
// ─────────────────────────────────────────────────────────────
// Apps declare *objectives* over SimParams; the engine searches
// the parameter space using one of:
//   - "gradient"  : differentiable (finite-difference surrogate)
//   - "bayesian"  : GP-lite acquisition (UCB) over a small candidate set
//   - "evolution" : (μ+λ) ES with mutation / crossover
// All engines respect bounds + inequality constraints (g(x) ≤ 0)
// and support multi-objective via Chebyshev scalarization w/ weights.
//
// This is a *self-contained* synthetic optimizer — it evaluates
// closed-form proxy objectives derived from SimParams (no live sim
// in the hot loop) so it stays smooth and deterministic.

import type { SimParams } from "@/components/PhysicsCanvas";

// ── Knob space (subset of SimParams we let the optimizer touch) ──
export type KnobKey =
  | "gravity" | "damping" | "attractor" | "springK" | "restLength"
  | "pairwiseStrength" | "pairwiseRadius" | "fieldStrength"
  | "subSteps" | "constraintIters" | "particleCount";

export interface KnobSpec {
  key: KnobKey;
  min: number;
  max: number;
  integer?: boolean;
}

export const DEFAULT_KNOBS: KnobSpec[] = [
  { key: "gravity",          min: -100, max: 300 },
  { key: "damping",          min: 0,    max: 0.95 },
  { key: "attractor",        min: 0,    max: 4 },
  { key: "springK",          min: 0,    max: 300 },
  { key: "restLength",       min: 10,   max: 120 },
  { key: "pairwiseStrength", min: -300, max: 800 },
  { key: "pairwiseRadius",   min: 0,    max: 150 },
  { key: "fieldStrength",    min: -1.5, max: 1.5 },
  { key: "subSteps",         min: 1,    max: 6, integer: true },
  { key: "constraintIters",  min: 0,    max: 12, integer: true },
  { key: "particleCount",    min: 100,  max: 4000, integer: true },
];

// ── Objectives ──
export type ObjectiveKey =
  | "energy"          // minimize kinetic-proxy
  | "stability"       // minimize stiffness/dt instability
  | "fidelity"        // maximize sub-steps & constraint iters (negated)
  | "throughput"      // maximize particles/ms (negated to minimize)
  | "yield"           // fabrication: minimize defect proxy
  | "cost"            // minimize compute proxy
  | "wear";           // predictive maintenance: minimize cyclic stress

export interface ObjectiveSpec {
  key: ObjectiveKey;
  weight: number;        // ≥ 0
  target?: number;       // optional reference; defaults 0
}

// ── Constraints (g(x) ≤ 0) ──
export interface ConstraintSpec {
  key: "max_compute" | "max_energy" | "min_stability";
  bound: number;
}

// ── Application presets ──
export type AppKey = "fabrication" | "operational" | "maintenance" | "custom";

export const APP_PRESETS: Record<AppKey, {
  label: string;
  blurb: string;
  objectives: ObjectiveSpec[];
  constraints: ConstraintSpec[];
}> = {
  fabrication: {
    label: "Fabrication",
    blurb: "Maximize yield, minimize defects under stiffness budget.",
    objectives: [
      { key: "yield",     weight: 0.55 },
      { key: "fidelity",  weight: 0.25 },
      { key: "cost",      weight: 0.20 },
    ],
    constraints: [{ key: "max_compute", bound: 1.0 }],
  },
  operational: {
    label: "Operational efficiency",
    blurb: "Throughput-first under energy / stability budgets.",
    objectives: [
      { key: "throughput", weight: 0.5 },
      { key: "energy",     weight: 0.3 },
      { key: "cost",       weight: 0.2 },
    ],
    constraints: [
      { key: "max_energy",   bound: 0.8 },
      { key: "min_stability", bound: 0.2 }, // -stab + bound ≤ 0  ⇒ stab ≥ 0.2
    ],
  },
  maintenance: {
    label: "Predictive maintenance",
    blurb: "Minimize cyclic wear & energy spikes.",
    objectives: [
      { key: "wear",      weight: 0.6 },
      { key: "energy",    weight: 0.25 },
      { key: "stability", weight: 0.15 },
    ],
    constraints: [{ key: "max_energy", bound: 0.7 }],
  },
  custom: {
    label: "Custom",
    blurb: "Bring your own objective/constraint mix.",
    objectives: [{ key: "energy", weight: 1 }],
    constraints: [],
  },
};

// ── Closed-form proxy evaluator ──
// All proxies are normalized to ~[0..1] so weights are interpretable.
function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }

export interface EvalResult {
  objectives: Record<ObjectiveKey, number>;
  scalar: number;
  constraints: { key: ConstraintSpec["key"]; slack: number }[];
  feasible: boolean;
}

export function evaluate(
  params: SimParams,
  objs: ObjectiveSpec[],
  cons: ConstraintSpec[],
): EvalResult {
  const N = params.particleCount;
  const k = params.springK;
  const dt = 1 / 60;
  const omega = Math.sqrt(Math.max(k, 1e-3));
  const stiffPenalty = clamp01(omega * dt * params.subSteps - 0.6);
  const energy   = clamp01(0.4 * Math.abs(params.gravity) / 300 + 0.6 * Math.abs(params.attractor) / 4);
  const stability= clamp01(1 - stiffPenalty - clamp01((1 - params.damping) * 0.4));
  const fidelity = clamp01(1 - (params.subSteps - 1) / 5 * 0.6 - params.constraintIters / 12 * 0.4);
  const throughput = clamp01(1 - N / 4000 * (1 + (params.subSteps - 1) * 0.25));
  const yieldProxy = clamp01(0.5 * (1 - stability) + 0.3 * energy + 0.2 * (k / 300));
  const cost     = clamp01(N / 4000 * 0.6 + (params.subSteps - 1) / 5 * 0.25 + params.constraintIters / 12 * 0.15);
  const wear     = clamp01(0.45 * (k / 300) + 0.35 * energy + 0.20 * (1 - params.damping));

  const O: Record<ObjectiveKey, number> = {
    energy, stability, fidelity, throughput, yield: yieldProxy, cost, wear,
  };

  // Chebyshev scalarization with weights (lower is better).
  // For maximize-style ones (stability, throughput) we already inverted.
  let scalar = 0;
  let wsum = 0;
  for (const o of objs) {
    const v = O[o.key];
    const t = o.target ?? 0;
    scalar += o.weight * Math.abs(v - t);
    wsum += o.weight;
  }
  scalar = wsum > 0 ? scalar / wsum : scalar;

  const constraints = cons.map((c) => {
    let val = 0;
    if (c.key === "max_compute")    val = cost - c.bound;
    if (c.key === "max_energy")     val = energy - c.bound;
    if (c.key === "min_stability")  val = c.bound - stability;
    return { key: c.key, slack: val };
  });
  const feasible = constraints.every((c) => c.slack <= 1e-6);
  // Soft-penalize infeasibility so optimizer escapes.
  const penalty = constraints.reduce((s, c) => s + Math.max(0, c.slack) * 2, 0);

  return { objectives: O, scalar: scalar + penalty, constraints, feasible };
}

// ── Helpers ──
function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}
function snap(v: number, k: KnobSpec) {
  const c = Math.max(k.min, Math.min(k.max, v));
  return k.integer ? Math.round(c) : c;
}
function applyKnobs(base: SimParams, x: Record<KnobKey, number>): SimParams {
  return { ...base, ...x } as SimParams;
}
function knobsFromParams(base: SimParams, knobs: KnobSpec[]): Record<KnobKey, number> {
  const out: Partial<Record<KnobKey, number>> = {};
  for (const k of knobs) out[k.key] = base[k.key] as number;
  return out as Record<KnobKey, number>;
}

// ── Engines ──
export type EngineKey = "gradient" | "bayesian" | "evolution";

export interface OptStep {
  iter: number;
  scalar: number;
  feasible: boolean;
  best: number;
}

export interface OptResult {
  best: SimParams;
  bestEval: EvalResult;
  history: OptStep[];
  engine: EngineKey;
  evals: number;
}

export interface OptConfig {
  engine: EngineKey;
  iterations: number;
  knobs: KnobSpec[];
  objectives: ObjectiveSpec[];
  constraints: ConstraintSpec[];
  seed?: number;
}

export function optimize(base: SimParams, cfg: OptConfig): OptResult {
  const rand = rng(cfg.seed ?? 42);
  const history: OptStep[] = [];
  let evals = 0;

  const evalAt = (x: Record<KnobKey, number>) => {
    evals++;
    return evaluate(applyKnobs(base, x), cfg.objectives, cfg.constraints);
  };

  let bestX = knobsFromParams(base, cfg.knobs);
  let bestE = evalAt(bestX);
  const record = (i: number, e: EvalResult) => {
    if (e.scalar < bestE.scalar) bestE = e;
    history.push({ iter: i, scalar: e.scalar, feasible: e.feasible, best: bestE.scalar });
  };

  if (cfg.engine === "gradient") {
    // Finite-difference projected descent
    let x = { ...bestX };
    const lr0 = 0.15;
    for (let i = 0; i < cfg.iterations; i++) {
      const grad: Partial<Record<KnobKey, number>> = {};
      const fx = evalAt(x).scalar;
      for (const k of cfg.knobs) {
        const span = k.max - k.min;
        const h = Math.max(span * 0.01, 1e-3);
        const xp = { ...x, [k.key]: snap(x[k.key] + h, k) };
        grad[k.key] = (evalAt(xp).scalar - fx) / h;
      }
      const lr = lr0 / (1 + i * 0.05);
      const nx: Record<KnobKey, number> = { ...x };
      for (const k of cfg.knobs) {
        const span = k.max - k.min;
        nx[k.key] = snap(x[k.key] - lr * (grad[k.key] ?? 0) * span, k);
      }
      x = nx;
      const e = evalAt(x);
      if (e.scalar < bestE.scalar) bestX = { ...x };
      record(i, e);
    }
  } else if (cfg.engine === "bayesian") {
    // GP-lite: sample N candidates, score by UCB over RBF kernel of past evals.
    const seen: { x: Record<KnobKey, number>; y: number }[] = [{ x: bestX, y: bestE.scalar }];
    for (let i = 0; i < cfg.iterations; i++) {
      const candidates: Record<KnobKey, number>[] = [];
      const C = 24;
      for (let c = 0; c < C; c++) {
        const cand: Partial<Record<KnobKey, number>> = {};
        for (const k of cfg.knobs) cand[k.key] = snap(k.min + rand() * (k.max - k.min), k);
        candidates.push(cand as Record<KnobKey, number>);
      }
      // Score each candidate via inverse-distance-weighted mean + exploration bonus.
      let pickIdx = 0;
      let pickScore = Infinity;
      for (let c = 0; c < candidates.length; c++) {
        const cand = candidates[c];
        let num = 0, den = 0, minD = Infinity;
        for (const s of seen) {
          let d2 = 0;
          for (const k of cfg.knobs) {
            const span = k.max - k.min || 1;
            const d = (cand[k.key] - s.x[k.key]) / span;
            d2 += d * d;
          }
          const w = Math.exp(-d2 * 4);
          num += w * s.y; den += w;
          if (d2 < minD) minD = d2;
        }
        const mean = den > 0 ? num / den : 0;
        const explore = -0.4 * Math.sqrt(minD); // bigger distance ⇒ lower (better) score
        const ucb = mean + explore;
        if (ucb < pickScore) { pickScore = ucb; pickIdx = c; }
      }
      const x = candidates[pickIdx];
      const e = evalAt(x);
      seen.push({ x, y: e.scalar });
      if (e.scalar < bestE.scalar) bestX = { ...x };
      record(i, e);
    }
  } else {
    // (μ+λ) Evolution Strategy with bounded Gaussian mutation
    const μ = 6, λ = 12;
    const pop: { x: Record<KnobKey, number>; y: number }[] = [];
    for (let p = 0; p < μ; p++) {
      const x: Partial<Record<KnobKey, number>> = {};
      for (const k of cfg.knobs) x[k.key] = snap(k.min + rand() * (k.max - k.min), k);
      const xx = x as Record<KnobKey, number>;
      pop.push({ x: xx, y: evalAt(xx).scalar });
    }
    pop.sort((a, b) => a.y - b.y);
    for (let i = 0; i < cfg.iterations; i++) {
      const offspring: { x: Record<KnobKey, number>; y: number }[] = [];
      for (let c = 0; c < λ; c++) {
        // crossover two parents
        const a = pop[Math.floor(rand() * μ)];
        const b = pop[Math.floor(rand() * μ)];
        const child: Partial<Record<KnobKey, number>> = {};
        const sigma = 0.15 * Math.exp(-i / Math.max(1, cfg.iterations) * 1.2);
        for (const k of cfg.knobs) {
          const cross = rand() < 0.5 ? a.x[k.key] : b.x[k.key];
          const span = k.max - k.min;
          // Box-Muller
          const u1 = Math.max(1e-9, rand()), u2 = rand();
          const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
          child[k.key] = snap(cross + z * sigma * span, k);
        }
        const cx = child as Record<KnobKey, number>;
        offspring.push({ x: cx, y: evalAt(cx).scalar });
      }
      const merged = pop.concat(offspring).sort((a, b) => a.y - b.y).slice(0, μ);
      pop.splice(0, pop.length, ...merged);
      if (pop[0].y < bestE.scalar) bestX = { ...pop[0].x };
      record(i, evaluate(applyKnobs(base, pop[0].x), cfg.objectives, cfg.constraints));
    }
  }

  return {
    best: applyKnobs(base, bestX),
    bestEval: evaluate(applyKnobs(base, bestX), cfg.objectives, cfg.constraints),
    history,
    engine: cfg.engine,
    evals,
  };
}

// Pareto front over a set of evaluations (for multi-obj reporting).
export function paretoFront(points: { x: Record<KnobKey, number>; o: Record<ObjectiveKey, number> }[],
  keys: ObjectiveKey[]) {
  const dominated = new Set<number>();
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < points.length; j++) {
      if (i === j || dominated.has(i)) continue;
      const a = points[i].o, b = points[j].o;
      let better = true, strict = false;
      for (const k of keys) {
        if (b[k] > a[k]) { better = false; break; }
        if (b[k] < a[k]) strict = true;
      }
      if (better && strict) { dominated.add(i); break; }
    }
  }
  return points.filter((_, i) => !dominated.has(i));
}
