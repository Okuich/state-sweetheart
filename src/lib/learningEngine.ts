// Physics Learning Engine
// ─────────────────────────────────────────────────────────────
// Learns reusable physical priors from synthetic (geometry +
// simulation + fabrication + inspection) examples. Three small
// online models share the feature vector:
//
//   • deformation prior   → ridge regression  (predicts max disp)
//   • stress risk         → logistic classifier (P[stress fail])
//   • manufacturability   → logistic classifier (P[fab success])
//
// Plus an "optimization heuristic" head that ranks design tweaks
// by expected reduction in (stress_risk − manufacturability).
//
// All training is pure TypeScript — SGD with momentum, no deps.

export type Sample = {
  // Geometry embedding (compressed) + sim/fab/qa scalars.
  // 12-dim feature vector — kept small so the math is legible.
  x: number[];
  // Targets:
  yDeform: number;      // measured max displacement, normalized
  yStressFail: 0 | 1;   // QA flagged stress failure
  yFabOk: 0 | 1;        // fabrication telemetry: clean run
};

export type Heads = {
  deform: { w: number[]; b: number };
  stress: { w: number[]; b: number };
  fab:    { w: number[]; b: number };
};

export type TrainStats = {
  epoch: number;
  mseDeform: number;
  bceStress: number;
  bceFab:    number;
  accStress: number;
  accFab:    number;
  loss:      number; // weighted sum
};

export const FEATURE_NAMES = [
  "thickness",        // 0  geometry
  "aspectRatio",      // 1
  "minRadius",        // 2  curvature
  "holeDensity",      // 3  fab feature
  "filletCount",      // 4
  "volume",           // 5
  "surfaceArea",      // 6
  "simEnergy",        // 7  simulation summary
  "simMaxStress",     // 8
  "fabSpindleLoad",   // 9  telemetry
  "fabVibration",     // 10
  "qaSurfaceDefect",  // 11 inspection
] as const;

export const D = FEATURE_NAMES.length;

// ─── deterministic PRNG ──────────────────────────────────────
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

// ─── synthetic dataset generator ─────────────────────────────
// Ground-truth physical priors used to generate plausible labels.
// The learner doesn't see these; it must recover them.
export function makeDataset(n: number, seed = 7): Sample[] {
  const rng = mulberry32(seed);
  const out: Sample[] = [];
  for (let i = 0; i < n; i++) {
    const thickness   = 0.5 + 4.5 * rng();          // mm
    const aspect      = 1 + 9 * rng();
    const minR        = 0.2 + 4 * rng();            // mm
    const holes       = Math.floor(rng() * 12);
    const fillets     = Math.floor(rng() * 8);
    const vol         = 50 + 950 * rng();
    const area        = 80 + 1200 * rng();
    const simE        = 0.1 + rng() * 5;
    const simStress   = 20 + rng() * 380;           // MPa
    const spindle     = 0.3 + rng() * 0.7;
    const vib         = rng() * 1.5;
    const defect      = rng();

    // standardize-ish: divide by rough scale, no leakage.
    const x = [
      thickness / 5,
      aspect / 10,
      minR / 5,
      holes / 12,
      fillets / 8,
      vol / 1000,
      area / 1300,
      simE / 5,
      simStress / 400,
      spindle,
      vib / 1.5,
      defect,
    ];

    // Hidden ground truth.
    // Deformation grows with stress, falls with thickness & minR.
    const deformTrue =
      0.8 * x[8] + 0.4 * x[1] - 0.6 * x[0] - 0.3 * x[2] + 0.2 * x[3];
    const yDeform = Math.max(0, deformTrue + gauss(rng, 0, 0.05));

    // Stress failure logit: thin + high stress + tight radii + sharp.
    const sLogit =
      -1.2 + 3.0 * x[8] - 2.5 * x[0] - 1.6 * x[2] + 1.4 * x[1] + 0.6 * x[10];
    const pS = 1 / (1 + Math.exp(-sLogit));
    const yStressFail: 0 | 1 = rng() < pS ? 1 : 0;

    // Fab success: penalized by holes, vibration, spindle load, defects.
    const fLogit =
      2.0 - 2.6 * x[3] - 2.2 * x[10] - 1.4 * x[9] - 2.8 * x[11] + 0.4 * x[2];
    const pF = 1 / (1 + Math.exp(-fLogit));
    const yFabOk: 0 | 1 = rng() < pF ? 1 : 0;

    out.push({ x, yDeform, yStressFail, yFabOk });
  }
  return out;
}

// ─── model init ──────────────────────────────────────────────
export function initHeads(seed = 1): Heads {
  const rng = mulberry32(seed);
  const init = () => Array.from({ length: D }, () => gauss(rng, 0, 0.1));
  return {
    deform: { w: init(), b: 0 },
    stress: { w: init(), b: 0 },
    fab:    { w: init(), b: 0 },
  };
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const dot = (a: number[], b: number[]) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export function predict(h: Heads, x: number[]) {
  return {
    deform: dot(h.deform.w, x) + h.deform.b,
    stressP: sigmoid(dot(h.stress.w, x) + h.stress.b),
    fabP:    sigmoid(dot(h.fab.w, x)    + h.fab.b),
  };
}

// ─── training loop (mini-batch SGD + momentum + L2) ──────────
export type TrainOpts = {
  epochs: number;
  lr: number;
  batch: number;
  l2: number;
  momentum: number;
};

export function train(
  heads: Heads,
  data: Sample[],
  opts: TrainOpts,
  onEpoch?: (s: TrainStats) => void
): { heads: Heads; history: TrainStats[] } {
  const rng = mulberry32(42);
  const v: Heads = JSON.parse(JSON.stringify({
    deform: { w: new Array(D).fill(0), b: 0 },
    stress: { w: new Array(D).fill(0), b: 0 },
    fab:    { w: new Array(D).fill(0), b: 0 },
  }));
  const history: TrainStats[] = [];

  for (let ep = 0; ep < opts.epochs; ep++) {
    // shuffle
    const idx = data.map((_, i) => i);
    for (let i = idx.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }

    for (let b = 0; b < idx.length; b += opts.batch) {
      const slice = idx.slice(b, b + opts.batch);
      const m = slice.length;
      // grads
      const gD = { w: new Array(D).fill(0), b: 0 };
      const gS = { w: new Array(D).fill(0), b: 0 };
      const gF = { w: new Array(D).fill(0), b: 0 };

      for (const k of slice) {
        const s = data[k];
        const p = predict(heads, s.x);
        const eD = p.deform - s.yDeform;          // MSE
        const eS = p.stressP - s.yStressFail;     // BCE grad simplifies to (p − y)
        const eF = p.fabP    - s.yFabOk;
        for (let i = 0; i < D; i++) {
          gD.w[i] += eD * s.x[i];
          gS.w[i] += eS * s.x[i];
          gF.w[i] += eF * s.x[i];
        }
        gD.b += eD;
        gS.b += eS;
        gF.b += eF;
      }

      const stepHead = (
        h: { w: number[]; b: number },
        g: { w: number[]; b: number },
        vh: { w: number[]; b: number },
      ) => {
        for (let i = 0; i < D; i++) {
          const grad = g.w[i] / m + opts.l2 * h.w[i];
          vh.w[i] = opts.momentum * vh.w[i] - opts.lr * grad;
          h.w[i] += vh.w[i];
        }
        const gb = g.b / m;
        vh.b = opts.momentum * vh.b - opts.lr * gb;
        h.b += vh.b;
      };
      stepHead(heads.deform, gD, v.deform);
      stepHead(heads.stress, gS, v.stress);
      stepHead(heads.fab,    gF, v.fab);
    }

    history.push(evaluate(heads, data, ep));
    onEpoch?.(history[history.length - 1]);
  }
  return { heads, history };
}

export function evaluate(heads: Heads, data: Sample[], epoch = 0): TrainStats {
  let mse = 0, bceS = 0, bceF = 0, accS = 0, accF = 0;
  for (const s of data) {
    const p = predict(heads, s.x);
    mse += (p.deform - s.yDeform) ** 2;
    const pS = Math.min(1 - 1e-6, Math.max(1e-6, p.stressP));
    const pF = Math.min(1 - 1e-6, Math.max(1e-6, p.fabP));
    bceS += -(s.yStressFail * Math.log(pS) + (1 - s.yStressFail) * Math.log(1 - pS));
    bceF += -(s.yFabOk      * Math.log(pF) + (1 - s.yFabOk)      * Math.log(1 - pF));
    if ((pS >= 0.5 ? 1 : 0) === s.yStressFail) accS++;
    if ((pF >= 0.5 ? 1 : 0) === s.yFabOk) accF++;
  }
  const n = data.length;
  return {
    epoch,
    mseDeform: mse / n,
    bceStress: bceS / n,
    bceFab:    bceF / n,
    accStress: accS / n,
    accFab:    accF / n,
    loss: mse / n + bceS / n + bceF / n,
  };
}

// ─── optimization heuristic ──────────────────────────────────
// Suggest single-feature tweaks that reduce risk and raise
// manufacturability. Uses learned weights as a local linear
// surrogate around the candidate.
export type Suggestion = {
  feature: string;
  index: number;
  delta: number;   // change to apply (in normalized units)
  gain: number;    // expected drop in (stress_risk − fab_ok)
  rationale: string;
};

export function suggestTweaks(heads: Heads, x: number[], k = 5): Suggestion[] {
  const out: Suggestion[] = [];
  // marginal effect on objective J = stressP − fabP at current x
  const p = predict(heads, x);
  const dS = p.stressP * (1 - p.stressP);
  const dF = p.fabP    * (1 - p.fabP);
  for (let i = 0; i < D; i++) {
    const dJ_dxi = dS * heads.stress.w[i] - dF * heads.fab.w[i];
    // pick step sign that decreases J, magnitude bounded by [0,1] feasibility
    const sign = dJ_dxi > 0 ? -1 : 1;
    const room = sign > 0 ? 1 - x[i] : x[i];
    const delta = sign * Math.min(0.2, room);
    const gain = -dJ_dxi * delta; // expected reduction
    out.push({
      feature: FEATURE_NAMES[i],
      index: i,
      delta,
      gain,
      rationale:
        Math.abs(heads.stress.w[i]) > Math.abs(heads.fab.w[i])
          ? "stress-dominant"
          : "fab-dominant",
    });
  }
  return out
    .filter((s) => s.gain > 1e-5 && Math.abs(s.delta) > 1e-3)
    .sort((a, b) => b.gain - a.gain)
    .slice(0, k);
}

// ─── feature importance (|w| across heads, normalized) ───────
export function featureImportance(heads: Heads) {
  const imp = FEATURE_NAMES.map((name, i) => ({
    name,
    deform: Math.abs(heads.deform.w[i]),
    stress: Math.abs(heads.stress.w[i]),
    fab:    Math.abs(heads.fab.w[i]),
    total:  Math.abs(heads.deform.w[i])
          + Math.abs(heads.stress.w[i])
          + Math.abs(heads.fab.w[i]),
  }));
  const max = Math.max(...imp.map((r) => r.total)) || 1;
  return imp.map((r) => ({ ...r, normalized: r.total / max }));
}
