/**
 * Signed Distance Field infrastructure for Geometry OS / Physics OS.
 *
 *   primitives  →  buildSparseSDF  →  queries / gradients / collision
 *                                  →  partitionSDF  (multi-GPU halo)
 *                                  →  buildEmbedding  (retrieval / priors)
 */

export { evalPrim, evalScene } from "./primitives";
export {
  buildSparseSDF, sampleSDF, BRICK, DEFAULT_OPTS,
  type SparseSDF, type SDFOptions, type SDFStats,
} from "./sparseField";
export {
  distance, gradient, nearestSurface, penetration, sphereCollide,
} from "./queries";
export { partitionSDF, type SDFPartitionPlan } from "./partition";
export { buildEmbedding, interiorCentroid, type SDFEmbedding } from "./embeddings";
export type { SDFPrim, AdaptiveHint, AdaptiveHintKind, AABB, Vec3 } from "./types";
