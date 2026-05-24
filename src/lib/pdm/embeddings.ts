/**
 * Manifold embedding + latent state compression for the PDM engine.
 *
 * Implementation: PCA via covariance eigendecomposition (Jacobi). We use
 * the method-of-snapshots trick when the number of samples is smaller
 * than the channel count — same approach as `src/lib/pde/rom.ts` so the
 * project stays consistent. Output is a low-rank projection matrix used
 * to compress raw state vectors into a stable latent space for kNN +
 * trajectory work.
 */

import { CHANNEL_ORDER, type SensorChannel, type SensorReading, type StateVector } from "./types";

export interface EmbeddingModel {
  /** Channel-wise mean (subtracted before projection). */
  mean: Float64Array;
  /** Channel-wise std (divides centered input). */
  std: Float64Array;
  /** k × d projection matrix (row-major). */
  basis: Float64Array;
  d: number;
  k: number;
  /** Singular values (sqrt eigenvalues of covariance). */
  sigma: Float64Array;
}

/** Pack a sparse SensorReading into a dense state vector (with imputation). */
export function packReading(r: SensorReading, mean?: Float64Array): StateVector {
  const v = new Float64Array(CHANNEL_ORDER.length);
  for (let i = 0; i < CHANNEL_ORDER.length; i++) {
    const ch = CHANNEL_ORDER[i] as SensorChannel;
    const x = r.channels[ch];
    v[i] = (x !== undefined && Number.isFinite(x)) ? x : (mean ? mean[i] : 0);
  }
  return v;
}

/** Build a PCA embedding from a sample matrix (rows = samples). */
export function fitEmbedding(samples: Float64Array[], k = 4): EmbeddingModel {
  const n = samples.length;
  if (n === 0) throw new Error("fitEmbedding: empty sample set");
  const d = samples[0].length;
  const kk = Math.max(1, Math.min(k, Math.min(n, d)));

  // Channel stats
  const mean = new Float64Array(d);
  for (const s of samples) for (let j = 0; j < d; j++) mean[j] += s[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  const std = new Float64Array(d);
  for (const s of samples) {
    for (let j = 0; j < d; j++) {
      const dv = s[j] - mean[j];
      std[j] += dv * dv;
    }
  }
  for (let j = 0; j < d; j++) std[j] = Math.sqrt(std[j] / Math.max(1, n - 1)) || 1;

  // Centered/scaled matrix X (n × d), row-major
  const X = new Float64Array(n * d);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < d; j++) X[i * d + j] = (samples[i][j] - mean[j]) / std[j];
  }

  // Covariance C = (1/(n-1)) Xᵀ X  (d × d), or method-of-snapshots if n < d
  let basis: Float64Array;
  let sigma: Float64Array;

  if (d <= n) {
    const C = new Float64Array(d * d);
    const inv = 1 / Math.max(1, n - 1);
    for (let a = 0; a < d; a++) {
      for (let b = a; b < d; b++) {
        let s = 0;
        for (let i = 0; i < n; i++) s += X[i * d + a] * X[i * d + b];
        const v = s * inv;
        C[a * d + b] = v;
        C[b * d + a] = v;
      }
    }
    const { values, vectors } = jacobiEig(C, d);
    const order = argSortDesc(values);
    basis = new Float64Array(kk * d);
    sigma = new Float64Array(kk);
    for (let i = 0; i < kk; i++) {
      const idx = order[i];
      sigma[i] = Math.sqrt(Math.max(0, values[idx]));
      for (let j = 0; j < d; j++) basis[i * d + j] = vectors[j * d + idx];
    }
  } else {
    // n < d: build n×n Gram and lift
    const G = new Float64Array(n * n);
    const inv = 1 / Math.max(1, n - 1);
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        let s = 0;
        for (let p = 0; p < d; p++) s += X[i * d + p] * X[j * d + p];
        const v = s * inv;
        G[i * n + j] = v;
        G[j * n + i] = v;
      }
    }
    const { values, vectors } = jacobiEig(G, n);
    const order = argSortDesc(values);
    basis = new Float64Array(kk * d);
    sigma = new Float64Array(kk);
    for (let i = 0; i < kk; i++) {
      const idx = order[i];
      const lam = Math.max(0, values[idx]);
      sigma[i] = Math.sqrt(lam);
      const denom = sigma[i] > 1e-12 ? sigma[i] * Math.sqrt(Math.max(1, n - 1)) : 1;
      for (let p = 0; p < d; p++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += vectors[j * n + idx] * X[j * d + p];
        basis[i * d + p] = s / denom;
      }
    }
  }

  return { mean, std, basis, d, k: kk, sigma };
}

/** Project a raw state vector into the latent space. */
export function project(model: EmbeddingModel, v: StateVector): Float64Array {
  const out = new Float64Array(model.k);
  for (let i = 0; i < model.k; i++) {
    let s = 0;
    for (let j = 0; j < model.d; j++) {
      s += model.basis[i * model.d + j] * ((v[j] - model.mean[j]) / model.std[j]);
    }
    out[i] = s;
  }
  return out;
}

/** Build a temporal embedding by stacking the last `w` latent vectors. */
export function temporalEmbed(latents: Float64Array[], w: number): Float64Array {
  const k = latents[0]?.length ?? 0;
  const out = new Float64Array(k * w);
  const n = latents.length;
  for (let i = 0; i < w; i++) {
    const src = latents[Math.max(0, n - w + i)] ?? new Float64Array(k);
    out.set(src, i * k);
  }
  return out;
}

// ---------------- helpers ----------------

function argSortDesc(a: Float64Array): number[] {
  const idx = Array.from({ length: a.length }, (_, i) => i);
  idx.sort((i, j) => a[j] - a[i]);
  return idx;
}

/** Cyclic Jacobi eigendecomposition for symmetric matrices (row-major). */
function jacobiEig(Ain: Float64Array, n: number): { values: Float64Array; vectors: Float64Array } {
  const A = new Float64Array(Ain);
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  const maxSweeps = 64;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) off += A[p * n + q] * A[p * n + q];
    }
    if (off < 1e-22) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = A[p * n + p];
        const aqq = A[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        A[p * n + p] = app - t * apq;
        A[q * n + q] = aqq + t * apq;
        A[p * n + q] = 0; A[q * n + p] = 0;
        for (let r = 0; r < n; r++) {
          if (r !== p && r !== q) {
            const arp = A[r * n + p], arq = A[r * n + q];
            A[r * n + p] = c * arp - s * arq; A[p * n + r] = A[r * n + p];
            A[r * n + q] = s * arp + c * arq; A[q * n + r] = A[r * n + q];
          }
          const vrp = V[r * n + p], vrq = V[r * n + q];
          V[r * n + p] = c * vrp - s * vrq;
          V[r * n + q] = s * vrp + c * vrq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = A[i * n + i];
  return { values, vectors: V };
}
