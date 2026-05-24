/**
 * Simulation Optimization — feature embeddings.
 *
 * Packs a SimDescriptor into a fixed-order numeric feature vector,
 * then projects it into a low-dim latent via online PCA built up
 * from observed simulations.
 *
 * Featurization is deterministic and stable so the same descriptor
 * always maps to the same latent — enabling content-addressed reuse.
 */

import type { SimDescriptor } from "./types";

/** Number of base features extracted before PCA projection. */
export const RAW_FEATURE_DIM = 24;

const DOMAINS: SimDescriptor["domain"][] = ["fem", "cfd", "thermal", "structural", "fluid"];

/** Stable raw feature packing. */
export function packDescriptor(d: SimDescriptor): Float64Array {
  const v = new Float64Array(RAW_FEATURE_DIM);
  // log-scaled size to keep things in a sensible range.
  v[0] = Math.log10(Math.max(1, d.ndof));
  v[1] = clamp01(d.density);
  v[2] = d.logKappa;
  v[3] = d.spd ? 1 : 0;
  v[4] = clamp01(d.nonlinearity);

  // Mesh quality (already in reasonable ranges).
  const m = d.mesh;
  v[5]  = Math.log10(Math.max(1, m.nElems));
  v[6]  = Math.log10(Math.max(1, m.nNodes));
  v[7]  = m.aspectMax;
  v[8]  = m.aspectMean;
  v[9]  = m.minAngle;
  v[10] = m.maxAngle;
  v[11] = m.jacobianMin;
  v[12] = clamp01(m.skewness);

  // Domain one-hot.
  for (let i = 0; i < DOMAINS.length; i++) {
    v[13 + i] = d.domain === DOMAINS[i] ? 1 : 0;
  }
  // 13..17 used.
  // BC fingerprint — fold into 6 slots via deterministic mixing.
  const fp = d.bcFingerprint ?? [];
  for (let i = 0; i < 6; i++) v[18 + i] = fp[i] ?? 0;
  return v;
}

export interface EmbeddingModel {
  mean: Float64Array;
  /** Principal directions, rows are components, length = RAW_FEATURE_DIM. */
  pcs: Float64Array[];
  /** Per-component standard deviations for whitening. */
  scale: Float64Array;
}

/** Fit a small PCA from a snapshot batch. */
export function fitEmbedding(snapshots: Float64Array[], dim: number): EmbeddingModel {
  const N = snapshots.length;
  const D = RAW_FEATURE_DIM;
  const mean = new Float64Array(D);
  for (const s of snapshots) for (let i = 0; i < D; i++) mean[i] += s[i];
  for (let i = 0; i < D; i++) mean[i] /= Math.max(1, N);

  // Centered matrix X (N×D) → covariance C (D×D).
  const C = new Float64Array(D * D);
  for (const s of snapshots) {
    for (let i = 0; i < D; i++) {
      const xi = s[i] - mean[i];
      for (let j = 0; j < D; j++) {
        C[i * D + j] += xi * (s[j] - mean[j]);
      }
    }
  }
  const inv = 1 / Math.max(1, N - 1);
  for (let i = 0; i < D * D; i++) C[i] *= inv;

  // Cyclic Jacobi eigendecomposition (D is small, ≤ ~32).
  const { eigvecs, eigvals } = jacobiEig(C, D);

  // Sort by descending eigenvalue.
  const order = [...Array(D).keys()].sort((a, b) => eigvals[b] - eigvals[a]);
  const k = Math.max(1, Math.min(dim, D));
  const pcs: Float64Array[] = [];
  for (let r = 0; r < k; r++) {
    const idx = order[r];
    const row = new Float64Array(D);
    for (let i = 0; i < D; i++) row[i] = eigvecs[i * D + idx];
    pcs.push(row);
  }

  // Per-axis scale = sqrt(eigval) for whitening; floor to avoid 1/0.
  const scale = new Float64Array(k);
  for (let r = 0; r < k; r++) {
    scale[r] = Math.max(1e-6, Math.sqrt(Math.max(0, eigvals[order[r]])));
  }

  return { mean, pcs, scale };
}

export function project(model: EmbeddingModel, raw: Float64Array): Float64Array {
  const k = model.pcs.length;
  const out = new Float64Array(k);
  for (let r = 0; r < k; r++) {
    let s = 0;
    const pc = model.pcs[r];
    for (let i = 0; i < raw.length; i++) s += pc[i] * (raw[i] - model.mean[i]);
    out[r] = s / model.scale[r];
  }
  return out;
}

function jacobiEig(A: Float64Array, n: number): { eigvecs: Float64Array; eigvals: Float64Array } {
  const a = new Float64Array(A);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  const maxSweeps = 60;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) off += a[p * n + q] * a[p * n + q];
    }
    if (off < 1e-18) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-20) continue;
        const app = a[p * n + p], aqq = a[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0; a[q * n + p] = 0;
        for (let i = 0; i < n; i++) {
          if (i !== p && i !== q) {
            const aip = a[i * n + p], aiq = a[i * n + q];
            a[i * n + p] = c * aip - s * aiq;
            a[p * n + i] = a[i * n + p];
            a[i * n + q] = s * aip + c * aiq;
            a[q * n + i] = a[i * n + q];
          }
          const vip = v[i * n + p], viq = v[i * n + q];
          v[i * n + p] = c * vip - s * viq;
          v[i * n + q] = s * vip + c * viq;
        }
      }
    }
  }
  const eigvals = new Float64Array(n);
  for (let i = 0; i < n; i++) eigvals[i] = a[i * n + i];
  return { eigvecs: v, eigvals };
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
