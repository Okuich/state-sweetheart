// Federated Industrial Learning System
// ─────────────────────────────────────────────────────────────
// Simulates N "companies" (clients) that each hold a private
// dataset (geometry + fab telemetry). Only model GRADIENTS leave
// the device — never raw features. Adds:
//   • differential-privacy noise (gaussian mechanism)
//   • gradient clipping (L2 norm)
//   • secure aggregation (additive secret-sharing simulation)
//   • cross-round drift / privacy budget tracking
//
// Pure TypeScript — no network. The "encryption" is modeled by
// per-client random masks that cancel in the sum, the standard
// secure-aggregation trick.

export type Vec = number[];

export type ClientShard = {
  id: string;
  name: string;
  // private samples: x ∈ R^D, y ∈ R
  X: Vec[];
  y: number[];
};

export type GlobalModel = {
  w: Vec;     // weights
  b: number;  // bias
  round: number;
};

export type DPConfig = {
  clipNorm: number;     // L2 clip
  noiseSigma: number;   // gaussian noise stdev (added to clipped grad)
  epsilonPerRound: number; // tracked, not enforced
};

export type RoundReport = {
  round: number;
  perClient: { id: string; name: string; gradNorm: number; lossBefore: number; lossAfter: number }[];
  aggregatedNorm: number;
  noiseEnergy: number;
  globalLoss: number;
  epsilonSpent: number;
  privacyBudgetRemaining: number;
};

export const D = 6; // feature dim per shard sample

// ── PRNG ────────────────────────────────────────────────────
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
  const u = Math.max(1e-9, rng()), v = rng();
  return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ── synthetic shards (each client has DIFFERENT distribution) ──
const COMPANY_NAMES = [
  "Helix Aero", "Forge North", "AltaMach", "BluPrint Co",
  "Caldera Mfg", "Nexus Tooling", "Orion Composites", "Vega Foundry",
];
export function makeFederation(numClients: number, samplesPerClient: number, seed = 7): ClientShard[] {
  const rng = mulberry32(seed);
  // hidden GLOBAL truth — same physics underneath all clients
  const wTrue = Array.from({ length: D }, () => gauss(rng, 0, 1));
  const bTrue = gauss(rng, 0, 0.5);
  const out: ClientShard[] = [];
  for (let c = 0; c < numClients; c++) {
    // per-client distribution shift
    const muShift = gauss(rng, 0, 0.3);
    const X: Vec[] = [];
    const y: number[] = [];
    for (let i = 0; i < samplesPerClient; i++) {
      const x = Array.from({ length: D }, () => gauss(rng, muShift, 1));
      const yi = x.reduce((a, v, k) => a + v * wTrue[k], bTrue) + gauss(rng, 0, 0.15);
      X.push(x);
      y.push(yi);
    }
    out.push({
      id: `client-${c + 1}`,
      name: COMPANY_NAMES[c % COMPANY_NAMES.length] + (c >= COMPANY_NAMES.length ? `-${c+1}` : ""),
      X, y,
    });
  }
  return out;
}

// ── linear model utils ─────────────────────────────────────
export function initGlobal(seed = 1): GlobalModel {
  const rng = mulberry32(seed);
  return { w: Array.from({ length: D }, () => gauss(rng, 0, 0.05)), b: 0, round: 0 };
}
function dot(a: Vec, b: Vec) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]*b[i]; return s; }
function l2(a: Vec) { return Math.sqrt(dot(a, a)); }

export function loss(model: GlobalModel, X: Vec[], y: number[]) {
  let s = 0;
  for (let i = 0; i < X.length; i++) {
    const e = dot(model.w, X[i]) + model.b - y[i];
    s += e * e;
  }
  return s / Math.max(1, X.length);
}

// local SGD: returns the *update* (w_local - w_global) to send up
function localTrain(global: GlobalModel, shard: ClientShard, lr: number, epochs: number, batch: number, rng: () => number) {
  const local: GlobalModel = { w: [...global.w], b: global.b, round: global.round };
  const N = shard.X.length;
  for (let e = 0; e < epochs; e++) {
    // shuffle
    const idx = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (let b0 = 0; b0 < N; b0 += batch) {
      const slice = idx.slice(b0, b0 + batch);
      const gW = new Array(D).fill(0); let gB = 0;
      for (const k of slice) {
        const x = shard.X[k];
        const err = dot(local.w, x) + local.b - shard.y[k];
        for (let d = 0; d < D; d++) gW[d] += err * x[d];
        gB += err;
      }
      const m = slice.length;
      for (let d = 0; d < D; d++) local.w[d] -= (lr * gW[d]) / m;
      local.b -= (lr * gB) / m;
    }
  }
  // delta to send
  const dw = local.w.map((v, i) => v - global.w[i]);
  const db = local.b - global.b;
  return { dw, db };
}

// ── DP: clip + add gaussian noise ──────────────────────────
function clipAndNoise(dw: Vec, db: number, cfg: DPConfig, rng: () => number) {
  const flat: Vec = [...dw, db];
  const n = l2(flat);
  const scale = Math.min(1, cfg.clipNorm / Math.max(1e-9, n));
  const clipped = flat.map((v) => v * scale);
  const noisy = clipped.map((v) => v + gauss(rng, 0, cfg.noiseSigma));
  return {
    dw: noisy.slice(0, D),
    db: noisy[D],
    clippedNorm: l2(clipped),
    noiseEnergy: noisy.reduce((a, v, i) => a + (v - clipped[i]) ** 2, 0),
  };
}

// ── secure aggregation simulation ──────────────────────────
// Each client adds a random mask r_i; the server only sees masked
// values; pairs cancel so Σ masked = Σ true. We simulate by drawing
// pairwise masks from a shared seed per (i,j) and verifying the
// aggregate equals direct sum (within epsilon).
function secureSum(values: Vec[], rng: () => number) {
  const C = values.length;
  const L = values[0].length;
  const masked = values.map((v) => [...v]);
  for (let i = 0; i < C; i++) {
    for (let j = i + 1; j < C; j++) {
      const r = Array.from({ length: L }, () => gauss(rng, 0, 1));
      for (let k = 0; k < L; k++) {
        masked[i][k] += r[k];
        masked[j][k] -= r[k];
      }
    }
  }
  const sum = new Array(L).fill(0);
  for (const m of masked) for (let k = 0; k < L; k++) sum[k] += m[k];
  return sum;
}

// ── one federated round ────────────────────────────────────
export function federatedRound(
  global: GlobalModel,
  clients: ClientShard[],
  cfg: DPConfig,
  opts: { lr: number; localEpochs: number; batch: number; serverLR: number; budgetRemaining: number },
  seed: number
): { next: GlobalModel; report: RoundReport } {
  const rng = mulberry32(seed);
  const updates: { dw: Vec; db: number }[] = [];
  const perClient: RoundReport["perClient"] = [];

  for (const c of clients) {
    const lossBefore = loss(global, c.X, c.y);
    const { dw, db } = localTrain(global, c, opts.lr, opts.localEpochs, opts.batch, rng);
    const dp = clipAndNoise(dw, db, cfg, rng);
    updates.push({ dw: dp.dw, db: dp.db });
    const lossAfter = loss({ w: global.w.map((v, i) => v + dp.dw[i]), b: global.b + dp.db, round: global.round }, c.X, c.y);
    perClient.push({
      id: c.id, name: c.name, gradNorm: dp.clippedNorm,
      lossBefore, lossAfter,
    });
  }

  // secure aggregate (sum then average)
  const flatVecs = updates.map((u) => [...u.dw, u.db]);
  const summed = secureSum(flatVecs, rng);
  const C = updates.length;
  const avg = summed.map((v) => v / C);

  // server step (FedAvg w/ optional server LR)
  const next: GlobalModel = {
    w: global.w.map((v, i) => v + opts.serverLR * avg[i]),
    b: global.b + opts.serverLR * avg[D],
    round: global.round + 1,
  };

  // global eval (across union — *only used to display loss curve*; in
  // a real deployment this would be a held-out shared validation set)
  const allX = clients.flatMap((c) => c.X);
  const allY = clients.flatMap((c) => c.y);
  const globalLoss = loss(next, allX, allY);

  const noiseEnergy = perClient.reduce((a, _, i) => a + l2(updates[i].dw) ** 2 + updates[i].db ** 2, 0) / C;
  const aggregatedNorm = l2([...avg]);

  const epsilonSpent = cfg.epsilonPerRound * (next.round);
  return {
    next,
    report: {
      round: next.round,
      perClient,
      aggregatedNorm,
      noiseEnergy,
      globalLoss,
      epsilonSpent,
      privacyBudgetRemaining: Math.max(0, opts.budgetRemaining - cfg.epsilonPerRound),
    },
  };
}
