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
export {
  dPrim, dScene, dField,
  type PrimGradients, type DiffSceneResult,
} from "./differentiable";
export { partitionSDF, type SDFPartitionPlan } from "./partition";
export { buildEmbedding, interiorCentroid, type SDFEmbedding } from "./embeddings";
export {
  createGpuSdfBackend,
  type GpuMode, type GpuBackend, type GpuUnavailable,
} from "./gpu";
export type { SDFPrim, AdaptiveHint, AdaptiveHintKind, AABB, Vec3 } from "./types";
