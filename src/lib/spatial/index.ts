export { morton3D, mortonCodes, mortonOrder, expandBits, type Vec3 } from "./morton";
export { buildBVH, bvhQueryAABB, bvhAllPairs, type BVH, type AABB } from "./bvh";
export { buildKDTree, knn, radiusQuery, type KDTree } from "./kdtree";
export { buildSpatialHash, broadphasePairs, type SpatialHash } from "./spatialHash";
export { rayAABB, rayQuery, pointInSolid, contactCandidates, type RayHit } from "./queries";
export { analyzeFabrication, FAB_CLASS_LABELS, type FabIntelReport } from "./fabIntel";

import type { OctreeMesh } from "../meshing/octree";
import { buildBVH, type BVH } from "./bvh";

/** Build a BVH directly from an octree leaf list (each leaf = one primitive). */
export function bvhFromOctree(mesh: OctreeMesh): BVH {
  const L = mesh.leaves.length;
  const lo = new Float32Array(L * 3);
  const hi = new Float32Array(L * 3);
  for (let i = 0; i < L; i++) {
    const n = mesh.nodes[mesh.leaves[i]];
    lo[i * 3]     = n.bbox.min[0]; lo[i * 3 + 1] = n.bbox.min[1]; lo[i * 3 + 2] = n.bbox.min[2];
    hi[i * 3]     = n.bbox.max[0]; hi[i * 3 + 1] = n.bbox.max[1]; hi[i * 3 + 2] = n.bbox.max[2];
  }
  return buildBVH(lo, hi);
}
