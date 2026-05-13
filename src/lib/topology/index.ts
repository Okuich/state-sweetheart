/**
 * Topology Intelligence Engine — top-level entry.
 *
 *   octree mesh  →  buildTopologyGraph  →  classifyFeatures
 *                                       →  scoreManufacturability
 *                                       →  buildStructuralEmbedding
 *                                       →  partitionTopology  (distributed)
 *
 * Returns physics priors for adaptive meshing + deformation heuristics so
 * Physics OS and the meshing layer can consume them directly.
 */

import { buildTopologyGraph, type TopologyGraph } from "./graph";
import { classifyFeatures, type FeatureReport } from "./features";
import { scoreManufacturability, type ManufacturabilityScores } from "./manufacturability";
import { buildStructuralEmbedding, cosine, type StructuralEmbedding } from "./embeddings";
import { partitionTopology, type TopologyPartitionPlan } from "./partition";
import type { OctreeMesh } from "../meshing/octree";
import type { AABB, FeatureClass } from "./types";

export type { TopologyGraph } from "./graph";
export type { FeatureReport } from "./features";
export type { ManufacturabilityScores } from "./manufacturability";
export type { StructuralEmbedding } from "./embeddings";
export type { TopologyPartitionPlan } from "./partition";
export type { TopoNode, TopoEdge, FeatureClass } from "./types";
export { FEATURE_LABELS } from "./types";
export { cosine, buildTopologyGraph, classifyFeatures, scoreManufacturability, buildStructuralEmbedding, partitionTopology };

export interface TopologyResult {
  graph: TopologyGraph;
  features: FeatureReport;
  manufacturability: ManufacturabilityScores;
  embedding: StructuralEmbedding;
  partition: TopologyPartitionPlan;
  /** Physics OS priors. */
  priors: PhysicsPriors;
  totalMs: number;
}

export interface PhysicsPriors {
  /** Per-feature-class refinement multiplier in [0,2]. */
  refinementHints: Record<FeatureClass, number>;
  /** Time-step scale recommendation in [0.1,1] — small under stress concentrators. */
  timestepScale: number;
  /** Damping recommendation in [0,1] — high near thin walls / overhangs. */
  damping: number;
  /** Boundary stabilization weight in [0,1] — high near cavities. */
  contactStiffness: number;
}

export interface AnalyzeOptions {
  partitionCount?: number;
}

export function analyzeTopology(mesh: OctreeMesh, opts: AnalyzeOptions = {}): TopologyResult {
  const t0 = Date.now();
  const graph = buildTopologyGraph(mesh);
  const features = classifyFeatures(graph, mesh.bbox);
  const manufacturability = scoreManufacturability(graph, features, mesh.bbox);
  const embedding = buildStructuralEmbedding(graph, features, manufacturability, mesh.bbox);
  const partition = partitionTopology(graph, opts.partitionCount ?? 4);
  const priors = derivePriors(features, manufacturability);
  return { graph, features, manufacturability, embedding, partition, priors, totalMs: Date.now() - t0 };
}

function derivePriors(feats: FeatureReport, manuf: ManufacturabilityScores): PhysicsPriors {
  return {
    refinementHints: {
      bulk: 0.5,
      boundary: 1.0,
      thin_wall: 1.6,
      overhang: 1.4,
      cavity: 1.5,
      stress_concentrator: 2.0,
      thermal_bottleneck: 1.7,
      symmetry_seed: 0.8,
    },
    timestepScale: clamp(0.3, 1, 1 - manuf.drivers.stressPenalty * 0.7),
    damping: clamp(0, 1, 0.1 + manuf.drivers.thinWallPenalty * 0.4 + manuf.drivers.overhangPenalty * 0.4),
    contactStiffness: clamp(0, 1, 0.4 + manuf.drivers.cavityPenalty * 0.6),
  };
  function clamp(lo: number, hi: number, v: number) { return Math.max(lo, Math.min(hi, v)); }
}

/** Convenience: nearest-neighbor retrieval over a corpus of embeddings. */
export function retrieveSimilar(
  query: StructuralEmbedding,
  corpus: { id: string; embedding: StructuralEmbedding }[],
  k = 5,
): { id: string; similarity: number }[] {
  const scored = corpus.map((c) => ({ id: c.id, similarity: cosine(query.vector, c.embedding.vector) }));
  scored.sort((a, b) => b.similarity - a.similarity);
  return scored.slice(0, k);
}

export type { AABB };
