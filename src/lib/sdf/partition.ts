/**
 * Distributed SDF partitioning with halo regions.
 *
 * Slabs the bbox along its longest axis into `partitionCount` partitions,
 * assigns each allocated brick to the partition containing its centroid, and
 * computes a per-partition halo (bricks within `haloBricks` of the partition
 * boundary) that must be replicated for safe distance/gradient queries near
 * boundaries.
 *
 * The output mirrors the meshing partitioner so dashboards can render both
 * with the same comm-matrix UI.
 */

import type { SparseSDF } from "./sparseField";
import { BRICK } from "./sparseField";

export interface SDFPartitionPlan {
  partitionCount: number;
  axis: 0 | 1 | 2;
  /** Per-partition resident brick indices. */
  resident: number[][];
  /** Per-partition halo brick indices (subset of other partitions). */
  halos: number[][];
  /** Flattened P×P brick-flow matrix (number of halo bricks pulled). */
  commMatrix: number[];
  imbalance: number;
  meanResident: number;
  totalHalo: number;
}

export function partitionSDF(sdf: SparseSDF, partitionCount: number, haloBricks = 1): SDFPartitionPlan {
  const P = Math.max(1, partitionCount);
  // Pick longest axis.
  const ext = [
    sdf.bbox.max[0] - sdf.bbox.min[0],
    sdf.bbox.max[1] - sdf.bbox.min[1],
    sdf.bbox.max[2] - sdf.bbox.min[2],
  ];
  let axis: 0 | 1 | 2 = 0;
  if (ext[1] > ext[axis]) axis = 1;
  if (ext[2] > ext[axis]) axis = 2;
  const span = ext[axis];
  const slab = span / P;
  const owners = new Int32Array(sdf.bricks.length);

  for (let i = 0; i < sdf.bricks.length; i++) {
    const b = sdf.bricks[i];
    const c = b.origin[axis] + (b.voxelSize * BRICK) / 2;
    const t = (c - sdf.bbox.min[axis]) / span;
    let p = Math.floor(t * P);
    if (p < 0) p = 0;
    if (p >= P) p = P - 1;
    owners[i] = p;
  }

  const resident: number[][] = Array.from({ length: P }, () => []);
  for (let i = 0; i < owners.length; i++) resident[owners[i]].push(i);

  // Halo: bricks whose centroid lies within haloBricks * brickWorld of an adjacent slab.
  const halos: number[][] = Array.from({ length: P }, () => []);
  const commMatrix = new Array(P * P).fill(0);
  for (let i = 0; i < sdf.bricks.length; i++) {
    const b = sdf.bricks[i];
    const owner = owners[i];
    const c = b.origin[axis] + (b.voxelSize * BRICK) / 2;
    const haloDist = haloBricks * b.voxelSize * BRICK;
    // Distance to lower neighbor boundary.
    if (owner > 0) {
      const boundary = sdf.bbox.min[axis] + owner * slab;
      if (c - boundary < haloDist) {
        halos[owner - 1].push(i);
        commMatrix[(owner - 1) * P + owner] += 1;
      }
    }
    if (owner < P - 1) {
      const boundary = sdf.bbox.min[axis] + (owner + 1) * slab;
      if (boundary - c < haloDist) {
        halos[owner + 1].push(i);
        commMatrix[(owner + 1) * P + owner] += 1;
      }
    }
  }

  const sizes = resident.map((r) => r.length);
  const mean = sizes.reduce((a, b) => a + b, 0) / Math.max(1, P);
  const max = Math.max(...sizes, 0);
  const imbalance = mean > 0 ? (max - mean) / mean : 0;

  return {
    partitionCount: P,
    axis,
    resident,
    halos,
    commMatrix,
    imbalance,
    meanResident: mean,
    totalHalo: halos.reduce((a, b) => a + b.length, 0),
  };
}
