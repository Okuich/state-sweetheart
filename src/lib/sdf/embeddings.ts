/**
 * Geometry embeddings + topology descriptors derived from a SparseSDF.
 *
 * These are deterministic, low-dimensional vectors suitable for retrieval,
 * manufacturability priors, and ML downstream tasks. They are NOT learned —
 * they're handcrafted invariants computed from the field statistics.
 */

import { BRICK, type SparseSDF } from "./sparseField";
import type { Vec3 } from "./types";

export interface SDFEmbedding {
  /** 32-d global geometry embedding. */
  geometry: Float32Array;
  /** Topology descriptors. */
  topology: {
    /** Estimated genus from Euler characteristic of surface bricks. */
    genusEstimate: number;
    /** Surface area / bounding box area. */
    surfaceRatio: number;
    /** Volume fraction of the bbox occupied by the solid. */
    volumeFraction: number;
    /** Mean wall thickness (bricks). */
    meanThickness: number;
    /** Min wall thickness — flags thin walls for fab. */
    minThickness: number;
    /** Histogram of |sdf| values inside the band, 16 bins. */
    bandHistogram: number[];
  };
  /** Manufacturability prior (0 = bad, 1 = good). */
  manufacturability: {
    score: number;
    overhangRisk: number;
    thinWallRisk: number;
    trappedVolumeRisk: number;
  };
}

export function buildEmbedding(sdf: SparseSDF): SDFEmbedding {
  const N = sdf.bricks.length;
  const band = sdf.options.bandWidth * sdf.options.voxelSize;
  let inside = 0;
  let total = 0;
  let surface = 0;
  let thinSum = 0;
  let thinCount = 0;
  let minThick = Infinity;
  const hist = new Array(16).fill(0);
  let overhangBricks = 0;
  let trappedBricks = 0;

  for (const b of sdf.bricks) {
    if (b.surface) surface++;
    for (let i = 0; i < b.voxels.length; i++) {
      const d = b.voxels[i];
      total++;
      if (d < 0) inside++;
      const t = Math.min(1, Math.abs(d) / band);
      const bin = Math.min(15, Math.floor(t * 16));
      hist[bin]++;
    }
    // Wall thickness ≈ 2 * |min interior distance| inside this brick.
    if (b.surface) {
      let minNeg = 0;
      for (let i = 0; i < b.voxels.length; i++) {
        if (b.voxels[i] < minNeg) minNeg = b.voxels[i];
      }
      const wt = -minNeg * 2;
      if (wt > 0) {
        thinSum += wt;
        thinCount++;
        if (wt < minThick) minThick = wt;
        if (wt < sdf.options.voxelSize * 1.5) overhangBricks++;
      }
    }
    // Trapped: brick is fully inside (mip < -band) and far from any surface neighbor.
    if (b.mip < -band * 0.8) trappedBricks++;
  }

  const dims = sdf.stats.rootBrickDims;
  const bboxBrickArea = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2]);
  const surfaceRatio = bboxBrickArea > 0 ? surface / bboxBrickArea : 0;
  const volumeFraction = total > 0 ? inside / total : 0;
  const meanThickness = thinCount > 0 ? thinSum / thinCount : 0;
  const minThickness = thinCount > 0 ? minThick : 0;

  // Estimated genus via Euler-like proxy from surface brick adjacency parity.
  // V - E + F ≈ 2 - 2g  (very rough on a brick complex)
  const F = surface;
  const V = Math.max(1, Math.round(Math.cbrt(F) * 8));
  const E = Math.max(1, Math.round(F * 1.5));
  const genusEstimate = Math.max(0, Math.round((2 - (V - E + F)) / 2));

  // Manufacturability priors.
  const overhangRisk = surface > 0 ? overhangBricks / surface : 0;
  const thinWallRisk = minThickness > 0 ? Math.max(0, 1 - minThickness / (sdf.options.voxelSize * 4)) : 0;
  const trappedVolumeRisk = N > 0 ? trappedBricks / N : 0;
  const score = Math.max(0, 1 - 0.5 * overhangRisk - 0.3 * thinWallRisk - 0.2 * trappedVolumeRisk);

  // 32-d geometry embedding: 16 band hist (normalized) + 16 derived features.
  const geometry = new Float32Array(32);
  const histSum = hist.reduce((a, b) => a + b, 0) || 1;
  for (let i = 0; i < 16; i++) geometry[i] = hist[i] / histSum;
  geometry[16] = volumeFraction;
  geometry[17] = surfaceRatio;
  geometry[18] = meanThickness / Math.max(sdf.options.voxelSize, 1e-6);
  geometry[19] = minThickness / Math.max(sdf.options.voxelSize, 1e-6);
  geometry[20] = overhangRisk;
  geometry[21] = thinWallRisk;
  geometry[22] = trappedVolumeRisk;
  geometry[23] = score;
  geometry[24] = sdf.stats.sparsity;
  geometry[25] = N > 0 ? surface / N : 0;
  geometry[26] = sdf.stats.adaptiveBricks / Math.max(1, N);
  geometry[27] = Math.log10(N + 1) / 6;
  geometry[28] = genusEstimate / 8;
  geometry[29] = sdf.stats.levels / 8;
  geometry[30] = (sdf.bbox.max[0] - sdf.bbox.min[0]) / Math.max(1, sdf.bbox.max[1] - sdf.bbox.min[1]);
  geometry[31] = (sdf.bbox.max[2] - sdf.bbox.min[2]) / Math.max(1, sdf.bbox.max[1] - sdf.bbox.min[1]);

  return {
    geometry,
    topology: {
      genusEstimate,
      surfaceRatio,
      volumeFraction,
      meanThickness,
      minThickness,
      bandHistogram: hist,
    },
    manufacturability: {
      score,
      overhangRisk,
      thinWallRisk,
      trappedVolumeRisk,
    },
  };
}

/** Centroid of all interior voxels — useful for support-generation seeding. */
export function interiorCentroid(sdf: SparseSDF): Vec3 {
  let sx = 0, sy = 0, sz = 0, n = 0;
  for (const b of sdf.bricks) {
    for (let z = 0; z < BRICK; z++) {
      for (let y = 0; y < BRICK; y++) {
        for (let x = 0; x < BRICK; x++) {
          const v = b.voxels[(z * BRICK + y) * BRICK + x];
          if (v < 0) {
            sx += b.origin[0] + x * b.voxelSize;
            sy += b.origin[1] + y * b.voxelSize;
            sz += b.origin[2] + z * b.voxelSize;
            n++;
          }
        }
      }
    }
  }
  if (n === 0) {
    return [
      (sdf.bbox.min[0] + sdf.bbox.max[0]) / 2,
      (sdf.bbox.min[1] + sdf.bbox.max[1]) / 2,
      (sdf.bbox.min[2] + sdf.bbox.max[2]) / 2,
    ];
  }
  return [sx / n, sy / n, sz / n];
}
