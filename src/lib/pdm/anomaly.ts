/**
 * Anomaly + drift primitives used by the PDM engine.
 *
 *  - RunningMoments: incremental mean/variance per latent dim (Welford).
 *  - kNN nearest distance.
 *  - Mahalanobis-style σ-score using running per-dim variance (diagonal
 *    approximation — cheap and stable for high-dim latents).
 *  - Adaptive threshold via EWMA of |score| plus a configurable k·σ cap.
 *  - Trajectory distance using last-w windowed L2.
 */

export class RunningMoments {
  n = 0;
  mean: Float64Array;
  m2: Float64Array;
  constructor(public d: number) {
    this.mean = new Float64Array(d);
    this.m2 = new Float64Array(d);
  }
  update(v: Float64Array): void {
    this.n++;
    for (let i = 0; i < this.d; i++) {
      const delta = v[i] - this.mean[i];
      this.mean[i] += delta / this.n;
      this.m2[i] += delta * (v[i] - this.mean[i]);
    }
  }
  variance(i: number): number { return this.n > 1 ? this.m2[i] / (this.n - 1) : 1; }
}

/** σ-distance from running mean (diagonal Mahalanobis). */
export function sigmaDistance(rm: RunningMoments, v: Float64Array): number {
  if (rm.n < 2) return 0;
  let s = 0;
  for (let i = 0; i < rm.d; i++) {
    const dv = v[i] - rm.mean[i];
    s += (dv * dv) / Math.max(1e-9, rm.variance(i));
  }
  return Math.sqrt(s / rm.d);
}

/** Nearest-neighbor distance + index (L2). Returns +∞ if `set` is empty. */
export function nearest(set: Float64Array[], q: Float64Array): { dist: number; idx: number } {
  let best = Infinity, bi = -1;
  for (let i = 0; i < set.length; i++) {
    const x = set[i];
    let s = 0;
    for (let j = 0; j < q.length; j++) { const d = x[j] - q[j]; s += d * d; }
    if (s < best) { best = s; bi = i; }
  }
  return { dist: Math.sqrt(best), idx: bi };
}

export class AdaptiveThreshold {
  ewma = 0;
  ewmsd = 1;
  init = false;
  constructor(public alpha = 0.05, public k = 3) {}
  /** Update with a fresh score and return current threshold. */
  update(score: number): number {
    const s = Math.abs(score);
    if (!this.init) { this.ewma = s; this.ewmsd = Math.max(1e-6, s * 0.5); this.init = true; }
    else {
      const diff = s - this.ewma;
      this.ewma += this.alpha * diff;
      this.ewmsd = (1 - this.alpha) * (this.ewmsd + this.alpha * diff * diff);
    }
    return this.threshold;
  }
  get threshold(): number { return this.ewma + this.k * Math.sqrt(this.ewmsd); }
}

/** Mean L2 between two equal-length latent trajectories. */
export function trajectoryDistance(a: Float64Array[], b: Float64Array[]): number {
  const n = Math.min(a.length, b.length);
  if (n === 0) return Infinity;
  let s = 0;
  for (let i = 0; i < n; i++) {
    const ai = a[a.length - n + i], bi = b[b.length - n + i];
    let d2 = 0;
    for (let j = 0; j < ai.length; j++) { const dv = ai[j] - bi[j]; d2 += dv * dv; }
    s += Math.sqrt(d2);
  }
  return s / n;
}

/** k-means++ style centroid init + Lloyd's iterations. */
export function kmeans(points: Float64Array[], k: number, iters = 20): {
  centroids: Float64Array[]; assign: number[];
} {
  const n = points.length;
  if (n === 0) return { centroids: [], assign: [] };
  const d = points[0].length;
  const centroids: Float64Array[] = [new Float64Array(points[Math.floor(seedRand() * n)])];
  while (centroids.length < Math.min(k, n)) {
    const dists = points.map((p) => Math.min(...centroids.map((c) => l2sq(p, c))));
    const sum = dists.reduce((a, b) => a + b, 0) || 1;
    let r = seedRand() * sum, idx = 0;
    for (; idx < n - 1; idx++) { r -= dists[idx]; if (r <= 0) break; }
    centroids.push(new Float64Array(points[idx]));
  }
  const assign = new Array(n).fill(0);
  for (let it = 0; it < iters; it++) {
    let moved = false;
    for (let i = 0; i < n; i++) {
      let best = 0, bestD = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const dd = l2sq(points[i], centroids[c]);
        if (dd < bestD) { bestD = dd; best = c; }
      }
      if (assign[i] !== best) { assign[i] = best; moved = true; }
    }
    const sums = centroids.map(() => new Float64Array(d));
    const cnts = new Array(centroids.length).fill(0);
    for (let i = 0; i < n; i++) {
      const c = assign[i];
      cnts[c]++;
      for (let j = 0; j < d; j++) sums[c][j] += points[i][j];
    }
    for (let c = 0; c < centroids.length; c++) {
      if (cnts[c] === 0) continue;
      for (let j = 0; j < d; j++) centroids[c][j] = sums[c][j] / cnts[c];
    }
    if (!moved) break;
  }
  return { centroids, assign };
}

function l2sq(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return s;
}

// Deterministic LCG so tests + engine are reproducible.
let _seed = 0x12345;
function seedRand(): number { _seed = (_seed * 1664525 + 1013904223) >>> 0; return _seed / 0x100000000; }
export function resetSeed(s = 0x12345): void { _seed = s >>> 0; }
