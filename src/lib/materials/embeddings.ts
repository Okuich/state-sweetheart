/**
 * Material Metric Intelligence — feature embeddings.
 *
 * Packs a MaterialRecord into a normalized, fixed-order feature vector
 * suitable for L2 / cosine similarity search. Each axis is min-max
 * scaled against the corpus to remove unit bias (a Pa is not a $).
 *
 *   axes:
 *     0  log10(yield strength)
 *     1  log10(ultimate strength)
 *     2  log10(youngs modulus)
 *     3  density
 *     4  log10(fatigue limit)
 *     5  log10(thermal conductivity)
 *     6  thermal expansion
 *     7  log10(max service temp)
 *     8  log10(cost per kg)
 *     9  log10(embodied CO2)
 *    10  corrosion resistance
 *    11  weather resistance
 */

import type { MaterialRecord } from "./types";

export const FEATURE_DIM = 12;

export interface FeatureScaler {
  min: Float64Array;
  max: Float64Array;
}

function safeLog10(x: number): number {
  return Math.log10(Math.max(1e-6, x));
}

export function rawFeatures(m: MaterialRecord): Float64Array {
  const v = new Float64Array(FEATURE_DIM);
  v[0]  = safeLog10(m.yieldStrength);
  v[1]  = safeLog10(m.ultimateStrength);
  v[2]  = safeLog10(m.youngsModulus);
  v[3]  = m.density;
  v[4]  = safeLog10(m.fatigueLimit);
  v[5]  = safeLog10(m.thermalConductivity);
  v[6]  = m.thermalExpansion;
  v[7]  = safeLog10(m.maxServiceTempC);
  v[8]  = safeLog10(m.costPerKg);
  v[9]  = safeLog10(m.embodiedCO2);
  v[10] = m.corrosionResistance;
  v[11] = m.weatherResistance;
  return v;
}

export function fitScaler(materials: MaterialRecord[]): FeatureScaler {
  const min = new Float64Array(FEATURE_DIM);
  const max = new Float64Array(FEATURE_DIM);
  for (let i = 0; i < FEATURE_DIM; i++) { min[i] = Infinity; max[i] = -Infinity; }
  for (const m of materials) {
    const f = rawFeatures(m);
    for (let i = 0; i < FEATURE_DIM; i++) {
      if (f[i] < min[i]) min[i] = f[i];
      if (f[i] > max[i]) max[i] = f[i];
    }
  }
  return { min, max };
}

export function embed(m: MaterialRecord, scaler: FeatureScaler): Float64Array {
  const f = rawFeatures(m);
  const out = new Float64Array(FEATURE_DIM);
  for (let i = 0; i < FEATURE_DIM; i++) {
    const d = scaler.max[i] - scaler.min[i];
    out[i] = d > 1e-9 ? (f[i] - scaler.min[i]) / d : 0;
  }
  return out;
}

export function l2(a: Float64Array, b: Float64Array): number {
  const n = Math.min(a.length, b.length);
  let s = 0;
  for (let i = 0; i < n; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}

export function cosine(a: Float64Array, b: Float64Array): number {
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i];
    na  += a[i] * a[i];
    nb  += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom > 1e-12 ? dot / denom : 0;
}
