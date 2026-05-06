/**
 * phaseFieldFracture.ts
 * 2D phase-field fracture model (AT2 functional).
 *
 *   E[u, d] = ∫ (1−d)² · ψ⁺(ε(u)) + G_c · ( d² / (4ℓ) + ℓ · |∇d|² )  dΩ
 *
 * Damage d ∈ [0,1] (0 = pristine, 1 = fully cracked) lives on a regular
 * grid. Each step we:
 *   1. Compute the elastic driving energy ψ⁺ from a prescribed strain
 *      field (here: a tension load growing with `loadFactor`, with a
 *      pre-notch concentrating stress).
 *   2. Update history H = max(H_prev, ψ⁺)  (irreversibility).
 *   3. Solve one Jacobi sweep of the AT2 damage equation:
 *        d = (2H + G_c·ℓ·Δd_neighbors / h²) / (2H + G_c/(2ℓ) + 4·G_c·ℓ/h²)
 *      then clamp to [d_prev, 1] (no healing).
 *   4. Track the crack set { d > dThreshold } and its growth over time.
 *
 * Pure / deterministic: same seed/load history → same field.
 */

export interface PhaseFieldOptions {
  /** Grid width in cells. Default 96. */
  W?: number;
  /** Grid height in cells. Default 96. */
  H?: number;
  /** Cell size h. Default 1/W. */
  h?: number;
  /** Regularization length ℓ (smear width of the crack). Default 4·h. */
  ell?: number;
  /** Fracture energy G_c. Default 1. */
  Gc?: number;
  /** Threshold for "is cracked" in the crack-mask. Default 0.95. */
  dThreshold?: number;
  /** Pre-notch as [x0,y0,x1,y1] in normalized [0,1] coords. */
  notch?: [number, number, number, number];
  /** Strain amplitude per unit loadFactor. Default 1. */
  strainAmp?: number;
}

export interface PhaseFieldState {
  W: number; H: number; h: number; ell: number; Gc: number;
  dThreshold: number; strainAmp: number;
  /** Damage field (length W*H). */
  d: Float32Array;
  /** Previous damage (for irreversibility / monotonic growth). */
  dPrev: Float32Array;
  /** History field H (max ψ⁺). */
  hist: Float32Array;
  /** Per-cell ψ⁺ snapshot (for visualization). */
  psi: Float32Array;
  /** Stress concentration factor map (1 + spike near notch). */
  concentration: Float32Array;
  /** Time-series of crack-set size. */
  crackSizeHistory: number[];
  /** Time-series of mean damage. */
  meanDamageHistory: number[];
  /** Step counter. */
  step: number;
  /** Current applied load factor. */
  loadFactor: number;
}

export function initPhaseField(opts: PhaseFieldOptions = {}): PhaseFieldState {
  const W = opts.W ?? 96;
  const H = opts.H ?? 96;
  const h = opts.h ?? 1 / W;
  const ell = opts.ell ?? 4 * h;
  const Gc = opts.Gc ?? 1;
  const dThreshold = opts.dThreshold ?? 0.95;
  const strainAmp = opts.strainAmp ?? 1;
  const notch = opts.notch ?? [0.0, 0.5, 0.25, 0.5];

  const N = W * H;
  const concentration = new Float32Array(N);
  // Distance-falloff stress concentration around the notch line segment.
  const x0 = notch[0] * W, y0 = notch[1] * H;
  const x1 = notch[2] * W, y1 = notch[3] * H;
  const lx = x1 - x0, ly = y1 - y0;
  const len2 = Math.max(1e-9, lx * lx + ly * ly);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      // Closest point on segment.
      const t = Math.max(0, Math.min(1, ((i - x0) * lx + (j - y0) * ly) / len2));
      const cx = x0 + t * lx, cy = y0 + t * ly;
      const dx = i - cx, dy = j - cy;
      const r = Math.sqrt(dx * dx + dy * dy);
      const tipBoost = t > 0.95 ? 1.2 : 0;
      // 1 baseline + spike near the notch tip and along its length.
      const c = 1 + 1.5 * Math.exp(-r * r / 8) + tipBoost * Math.exp(-r * r / 4);
      concentration[j * W + i] = c;
    }
  }

  return {
    W, H, h, ell, Gc, dThreshold, strainAmp,
    d:           new Float32Array(N),
    dPrev:       new Float32Array(N),
    hist:        new Float32Array(N),
    psi:         new Float32Array(N),
    concentration,
    crackSizeHistory: [],
    meanDamageHistory: [],
    step: 0,
    loadFactor: 0,
  };
}

/** One simulation step: increase load by `dLoad` and run one Jacobi sweep. */
export function stepPhaseField(s: PhaseFieldState, dLoad: number): void {
  s.loadFactor += dLoad;
  const eps = s.strainAmp * s.loadFactor;
  const eps2_half = 0.5 * eps * eps;

  // 1. ψ⁺ from concentration map.
  for (let i = 0; i < s.psi.length; i++) {
    const c = s.concentration[i];
    s.psi[i] = eps2_half * c * c;
    if (s.psi[i] > s.hist[i]) s.hist[i] = s.psi[i];
  }

  // 2. Snapshot current d as previous.
  s.dPrev.set(s.d);

  // 3. Jacobi sweep on damage equation.
  const W = s.W, H = s.H, h = s.h, ell = s.ell, Gc = s.Gc;
  const denomConst = Gc / (2 * ell) + 4 * Gc * ell / (h * h);
  const lapCoef = Gc * ell / (h * h);
  const dOld = s.dPrev;
  const d = s.d;
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const im = i > 0       ? k - 1 : k;
      const ip = i < W - 1   ? k + 1 : k;
      const jm = j > 0       ? k - W : k;
      const jp = j < H - 1   ? k + W : k;
      const lap = dOld[im] + dOld[ip] + dOld[jm] + dOld[jp];
      const num = 2 * s.hist[k] + lapCoef * lap;
      const denom = 2 * s.hist[k] + denomConst;
      let dNew = num / denom;
      if (dNew < dOld[k]) dNew = dOld[k]; // irreversibility
      if (dNew > 1) dNew = 1;
      d[k] = dNew;
    }
  }

  // 4. Track diagnostics.
  let cracked = 0, sum = 0;
  for (let i = 0; i < d.length; i++) {
    sum += d[i];
    if (d[i] > s.dThreshold) cracked++;
  }
  s.crackSizeHistory.push(cracked);
  s.meanDamageHistory.push(sum / d.length);
  s.step++;
}

/** Reset to pristine state but keep notch / params. */
export function resetPhaseField(s: PhaseFieldState): void {
  s.d.fill(0); s.dPrev.fill(0); s.hist.fill(0); s.psi.fill(0);
  s.crackSizeHistory.length = 0;
  s.meanDamageHistory.length = 0;
  s.step = 0;
  s.loadFactor = 0;
}
