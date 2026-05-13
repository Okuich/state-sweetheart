/**
 * MPI-style mesh partitioner with halo regions.
 * Greedy region-growing seeded by deterministic stride; produces:
 *   - per-tet partition id
 *   - halo (1-ring) tet sets per partition
 *   - inter-partition communication graph (edge-cut weights)
 * Output is solver-friendly and balanced within ~10% of mean partition size.
 */

import type { OctreeMesh } from "./octree";
import type { AdjacencyTensors } from "./adjacency";

export interface PartitionPlan {
  partitionCount: number;
  /** partition id per tet, length T. */
  tetPart: Uint16Array;
  /** size per partition. */
  sizes: Uint32Array;
  /** halo tets per partition (Uint32Array of tet ids). */
  halos: Uint32Array[];
  /** edge cut count between partitions (full P×P symmetric matrix, flat). */
  commMatrix: Uint32Array;
  /** Total number of cut faces. */
  edgeCut: number;
  /** Imbalance factor max(size)/mean(size). */
  imbalance: number;
  buildMs: number;
}

export function partitionMesh(
  mesh: OctreeMesh,
  adj: AdjacencyTensors,
  partitionCount: number,
): PartitionPlan {
  const t0 = Date.now();
  const T = mesh.tets.length / 4;
  const P = Math.max(1, Math.min(partitionCount, T));
  const tetPart = new Uint16Array(T);
  tetPart.fill(0xffff);

  // Seed: stride seeds across tets for spatial spread.
  const seeds: number[] = [];
  const stride = Math.max(1, Math.floor(T / P));
  for (let p = 0; p < P; p++) seeds.push(Math.min(T - 1, p * stride));

  // Multi-source BFS via round-robin queues.
  const frontiers: number[][] = seeds.map((s, p) => {
    tetPart[s] = p;
    return [s];
  });
  const sizes = new Uint32Array(P);
  for (let p = 0; p < P; p++) sizes[p] = 1;
  const cap = Math.ceil((T / P) * 1.1);

  let assigned = P;
  while (assigned < T) {
    let progressed = false;
    for (let p = 0; p < P; p++) {
      if (sizes[p] >= cap) continue;
      const f = frontiers[p];
      if (!f.length) continue;
      const next: number[] = [];
      for (const t of f) {
        const start = adj.ttRowPtr[t];
        const end = adj.ttRowPtr[t + 1];
        for (let k = start; k < end; k++) {
          const n = adj.ttCol[k];
          if (tetPart[n] === 0xffff) {
            tetPart[n] = p;
            sizes[p]++;
            assigned++;
            next.push(n);
            if (sizes[p] >= cap) break;
          }
        }
        if (sizes[p] >= cap) break;
      }
      frontiers[p] = next;
      if (next.length) progressed = true;
    }
    if (!progressed) break;
  }
  // Mop up any unassigned (e.g. detached components) into smallest partition.
  for (let t = 0; t < T; t++) {
    if (tetPart[t] === 0xffff) {
      let smallest = 0;
      for (let p = 1; p < P; p++) if (sizes[p] < sizes[smallest]) smallest = p;
      tetPart[t] = smallest;
      sizes[smallest]++;
    }
  }

  // Halo + communication matrix.
  const haloSets: Set<number>[] = Array.from({ length: P }, () => new Set<number>());
  const commMatrix = new Uint32Array(P * P);
  let edgeCut = 0;
  for (let t = 0; t < T; t++) {
    const pt = tetPart[t];
    const start = adj.ttRowPtr[t];
    const end = adj.ttRowPtr[t + 1];
    for (let k = start; k < end; k++) {
      const n = adj.ttCol[k];
      const pn = tetPart[n];
      if (pn !== pt) {
        haloSets[pt].add(n);
        commMatrix[pt * P + pn]++;
        edgeCut++;
      }
    }
  }

  const halos = haloSets.map((s) => new Uint32Array(Array.from(s).sort((a, b) => a - b)));
  const meanSize = T / P;
  let maxSize = 0;
  for (let p = 0; p < P; p++) if (sizes[p] > maxSize) maxSize = sizes[p];

  return {
    partitionCount: P,
    tetPart,
    sizes,
    halos,
    commMatrix,
    edgeCut: edgeCut / 2,
    imbalance: meanSize ? maxSize / meanSize : 1,
    buildMs: Date.now() - t0,
  };
}
