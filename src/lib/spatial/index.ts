export * from "./morton";
export * from "./bvh";
export * from "./kdtree";
export * from "./spatialHash";
export * from "./queries";
export * from "./fabIntel";

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
