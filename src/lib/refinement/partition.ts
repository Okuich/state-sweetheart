/**
 * Partition-aware adaptive refinement support.
 *
 * Given a base partition plan + a refinement pass, compute:
 *   - per-partition leaf delta (load drift)
 *   - halo growth from new leaves crossing partition boundaries
 *   - a comm-minimizing repartition hint when imbalance exceeds tolerance
 *
 * Repartitioning here is greedy (move worst-loaded leaves toward
 * neighbors). A real run would call ParMETIS / Zoltan; this is the
 * communication-aware shim Physics OS speaks to.
 */

import type { OctreeMesh, Vec3 } from "../meshing/octree";
import type { PartitionPlan } from "../meshing/partition";

/** Map each leaf to its dominant partition by majority of its 6 tets. */
function leafPartition(mesh: OctreeMesh, plan: PartitionPlan): Uint16Array {
  const N = mesh.leaves.length;
  const out = new Uint16Array(N);
  const counts = new Map<number, number[]>();
  for (let t = 0; t < mesh.tetLeaf.length; t++) {
    const li = mesh.tetLeaf[t];
    const p = plan.tetPart[t] ?? 0;
    let c = counts.get(li);
    if (!c) { c = new Array(plan.partitionCount).fill(0); counts.set(li, c); }
    c[p]++;
  }
  for (let li = 0; li < N; li++) {
    const c = counts.get(li);
    if (!c) { out[li] = 0; continue; }
    let best = 0, bestI = 0;
    for (let p = 0; p < c.length; p++) if (c[p] > best) { best = c[p]; bestI = p; }
    out[li] = bestI;
  }
  return out;
}

export interface RepartitionHint {
  partitionCount: number;
  /** New per-partition counts after greedy rebalance. */
  newSizes: number[];
  /** New per-partition halo size estimates. */
  newHaloSizes: number[];
  /** Estimated cross-partition messages avoided. */
  commSaved: number;
  /** Imbalance ratio after rebalance (max/mean). */
  imbalance: number;
}

function leafCenter(mesh: OctreeMesh, li: number): Vec3 {
  const b = mesh.nodes[mesh.leaves[li]].bbox;
  return [
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  ];
}

/**
 * Estimate halo growth: leaves whose neighbor in the *previous* partition
 * lives in another partition are counted as halo entries.
 */
export function estimateHaloDelta(
  baseMesh: OctreeMesh,
  basePartition: PartitionPlan,
  refinedMesh: OctreeMesh,
  splitLeafIdx: Uint32Array,
): { addedHalo: number; perPartition: number[] } {
  const P = basePartition.partitionCount;
  const perPartition = new Array<number>(P).fill(0);
  let addedHalo = 0;

  // Each split leaf becomes ~8 child leaves; assume face-adjacent neighbors
  // in 6 directions. Estimate how many neighbors cross partition lines via
  // proximity in the refined mesh's bbox layout.
  for (const li of splitLeafIdx) {
    const part = basePartition.assignment[li] ?? 0;
    // Crude heuristic: assume 30% of new face neighbors cross boundaries.
    const newHalo = Math.round(8 * 0.3);
    perPartition[part] += newHalo;
    addedHalo += newHalo;
  }
  void baseMesh; void refinedMesh; // shape-aware version would re-map by Morton
  return { addedHalo, perPartition };
}

export function planRepartition(
  basePartition: PartitionPlan,
  refinedMesh: OctreeMesh,
  splitLeafIdx: Uint32Array,
  imbalanceTol = 0.15,
): RepartitionHint {
  const P = basePartition.partitionCount;
  const sizes = Array.from(basePartition.sizes);

  // Add the splits to whichever partition currently owns the original leaf.
  for (const li of splitLeafIdx) {
    const part = basePartition.assignment[li] ?? 0;
    sizes[part] += 7; // split = +7 leaves
  }

  const total = sizes.reduce((a, b) => a + b, 0);
  const target = total / P;
  const before = Math.max(...sizes) / Math.max(1, target);

  // Greedy: move from heaviest to lightest, capped at 5% of leaves per move.
  const newSizes = sizes.slice();
  const moveCap = Math.max(1, Math.round(total * 0.05));
  for (let iter = 0; iter < P * 2; iter++) {
    const hi = newSizes.indexOf(Math.max(...newSizes));
    const lo = newSizes.indexOf(Math.min(...newSizes));
    if (hi === lo) break;
    const delta = Math.min(moveCap, Math.floor((newSizes[hi] - newSizes[lo]) / 2));
    if (delta <= 0) break;
    newSizes[hi] -= delta;
    newSizes[lo] += delta;
  }

  const after = Math.max(...newSizes) / Math.max(1, target);
  const newHaloSizes = newSizes.map((s, i) => {
    const baseHalo = basePartition.halos[i]?.length ?? 0;
    return Math.round(baseHalo * (s / Math.max(1, sizes[i] || 1)));
  });

  const commSaved = Math.max(0, Math.round((before - after) * total * 0.1));
  void refinedMesh; void leafCenter;

  return {
    partitionCount: P,
    newSizes,
    newHaloSizes,
    commSaved,
    imbalance: after,
  };
}

export function shouldRepartition(hint: RepartitionHint, tol = 1.15): boolean {
  return hint.imbalance > tol;
}
