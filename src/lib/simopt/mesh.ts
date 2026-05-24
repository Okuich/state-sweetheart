/**
 * Mesh quality metric system.
 *
 * Aggregates per-element quality into a scalar "quality score" in
 * [0,1] (1 = ideal mesh) and an "instability proxy" in [0,1]
 * (1 = highly likely to cause solver instability). These are used
 * by the router to refuse cache hits on degenerate meshes and to
 * pick a more robust solver (gmres > cg) when conditioning is bad.
 */

import type { MeshStats } from "./types";

export interface MeshQuality {
  /** Overall quality score in [0,1]. */
  score: number;
  /** Probability that the mesh will cause solver instability. */
  instability: number;
  /** Distortion score — composite of skewness + aspect outliers. */
  distortion: number;
  /** Effective conditioning penalty added to logKappa. */
  conditioningPenalty: number;
}

/**
 * Compute mesh quality. All sub-scores fold into a single
 * "score" with conservative weights:
 *
 *   - skewness   weight 0.30  (PDE residual blow-up risk)
 *   - aspect     weight 0.25  (anisotropy → poor preconditioner)
 *   - angles     weight 0.20  (tangent ill-conditioning)
 *   - jacobian   weight 0.25  (inverted elements → divergence)
 */
export function evalMeshQuality(m: MeshStats): MeshQuality {
  // Skewness: lower is better.
  const skew = clamp01(1 - m.skewness);

  // Aspect: 1 = ideal, large values penalized. Treat 8 as saturation.
  const aspect = clamp01(1 - clamp01((Math.max(1, m.aspectMean) - 1) / 7));
  const aspectMaxPenalty = clamp01(1 - clamp01((Math.max(1, m.aspectMax) - 1) / 15));

  // Angles: ideal ranges depend on element type; assume tris/tets where
  // ideal min ≈ π/3 (60°) and ideal max ≈ π/3..2π/3.
  const minAngleScore = clamp01(m.minAngle / (Math.PI / 3));
  const maxAngleScore = clamp01(1 - Math.max(0, m.maxAngle - 2 * Math.PI / 3) / (Math.PI / 3));
  const angleScore = 0.5 * (minAngleScore + maxAngleScore);

  // Jacobian: clamp to [0,1]; ≤ 0 means inverted → instability.
  const jac = clamp01(m.jacobianMin);

  const score = 0.30 * skew + 0.25 * (0.5 * aspect + 0.5 * aspectMaxPenalty)
              + 0.20 * angleScore + 0.25 * jac;

  // Instability heuristic: weighted complement plus hard signals.
  const hardInverted = m.jacobianMin <= 0 ? 1 : 0;
  const instability = clamp01(0.6 * (1 - score) + 0.4 * hardInverted);

  const distortion = clamp01(m.skewness + Math.max(0, m.aspectMax - 4) / 16);

  // Conditioning penalty (added to logKappa proxy): bad meshes raise
  // condition number empirically by ~1 decade per 0.4 quality lost.
  const conditioningPenalty = Math.max(0, (1 - score) * 2.5);

  return { score, instability, distortion, conditioningPenalty };
}

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }
