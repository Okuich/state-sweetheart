/**
 * Structural embeddings.
 *
 * Deterministic, low-dimensional descriptors derived from the topology graph
 * plus feature/manufacturability stats. Designed for retrieval, prior lookup,
 * and anomaly detection — not learned, but learnable downstream (the same
 * vector layout works as input to a contrastive encoder).
 *
 *   curvature[12]    — discrete curvature histogram (face-deficit 0..6 × hi/lo)
 *   features[8]      — feature-class fractions
 *   structural[12]   — moments + valence stats (size-invariant)
 *   manuf[8]         — manufacturability + driver penalties
 *   ----
 *   total = 40 dims
 */

import type { TopologyGraph } from "./graph";
import type { FeatureReport } from "./features";
import type { ManufacturabilityScores } from "./manufacturability";
import type { AABB, FeatureClass } from "./types";

const FEATURE_ORDER: FeatureClass[] = [
  "bulk", "boundary", "thin_wall", "overhang",
  "cavity", "stress_concentrator", "thermal_bottleneck", "symmetry_seed",
];

export interface StructuralEmbedding {
  /** 40-d vector. */
  vector: Float32Array;
  /** Slice indices (for inspection / labelling). */
  slices: { curvature: [number, number]; features: [number, number]; structural: [number, number]; manuf: [number, number] };
}

export function buildStructuralEmbedding(
  graph: TopologyGraph,
  feats: FeatureReport,
  manuf: ManufacturabilityScores,
  bbox: AABB,
): StructuralEmbedding {
  const N = Math.max(1, graph.nodes.length);
  const vec = new Float32Array(40);

  // Curvature histogram: 7 bins of face-deficit (0..6), each split high/low density → 12 dims (0..5 deficit).
  for (const n of graph.nodes) {
    const c = Math.min(5, n.curvature);
    const hi = n.density >= 0.4 ? 1 : 0;
    vec[c * 2 + hi]++;
  }
  for (let i = 0; i < 12; i++) vec[i] /= N;

  // Features fractions (8 dims).
  for (let i = 0; i < FEATURE_ORDER.length; i++) {
    vec[12 + i] = feats.counts[FEATURE_ORDER[i]] / N;
  }

  // Structural (12 dims): centroid offsets, second moments, valence stats.
  const cx = (bbox.min[0] + bbox.max[0]) / 2;
  const cy = (bbox.min[1] + bbox.max[1]) / 2;
  const cz = (bbox.min[2] + bbox.max[2]) / 2;
  const ext = Math.max(1e-6, Math.max(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2]));
  let mx = 0, my = 0, mz = 0;
  let m2x = 0, m2y = 0, m2z = 0;
  let valSum = 0, valMax = 0, edgeAreaSum = 0;
  for (let i = 0; i < N; i++) {
    const n = graph.nodes[i];
    const dx = (n.center[0] - cx) / ext;
    const dy = (n.center[1] - cy) / ext;
    const dz = (n.center[2] - cz) / ext;
    mx += dx; my += dy; mz += dz;
    m2x += dx * dx; m2y += dy * dy; m2z += dz * dz;
    const v = graph.neighborOffsets[i + 1] - graph.neighborOffsets[i];
    valSum += v;
    if (v > valMax) valMax = v;
  }
  for (const e of graph.edges) edgeAreaSum += e.shared;
  vec[20] = mx / N;
  vec[21] = my / N;
  vec[22] = mz / N;
  vec[23] = Math.sqrt(m2x / N);
  vec[24] = Math.sqrt(m2y / N);
  vec[25] = Math.sqrt(m2z / N);
  vec[26] = valSum / N / 6;
  vec[27] = valMax / 6;
  vec[28] = graph.edges.length / N;
  vec[29] = edgeAreaSum / Math.max(1, graph.edges.length) / (ext * ext);
  vec[30] = feats.symmetryScore;
  vec[31] = Math.log10(N + 1) / 6;

  // Manufacturability (8 dims).
  vec[32] = manuf.feasibility;
  vec[33] = manuf.supportFraction;
  vec[34] = manuf.machiningAccess;
  vec[35] = manuf.thermalDistortionRisk;
  vec[36] = manuf.assemblyComplexity;
  vec[37] = manuf.drivers.overhangPenalty;
  vec[38] = manuf.drivers.cavityPenalty;
  vec[39] = manuf.drivers.thinWallPenalty;

  return {
    vector: vec,
    slices: {
      curvature: [0, 12],
      features: [12, 20],
      structural: [20, 32],
      manuf: [32, 40],
    },
  };
}

/** Cosine similarity for retrieval. */
export function cosine(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) throw new Error("vector length mismatch");
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const n = Math.sqrt(na) * Math.sqrt(nb);
  return n > 0 ? dot / n : 0;
}
