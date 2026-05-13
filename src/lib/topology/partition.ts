/**
 * Distributed topology partitioning.
 *
 * Greedy multi-source BFS over the leaf graph that:
 *   1. Seeds K well-spread nodes via farthest-point sampling.
 *   2. Grows partitions via shared-face-area-weighted BFS — preserves locality
 *      better than naive node-count splits and minimizes inter-partition edge
 *      cut, which directly reduces halo-region traffic in distributed solvers.
 *   3. Computes halos: any node 1 hop into a neighboring partition.
 *   4. Returns a compact comm-matrix mirroring the meshing partitioner so
 *      dashboards can render both with the same visualization.
 */

import type { TopologyGraph } from "./graph";

export interface TopologyPartitionPlan {
  partitionCount: number;
  /** owner[node] = partition. */
  owners: Int32Array;
  /** Per-partition resident node indices. */
  resident: number[][];
  /** Per-partition halo node indices (subset of other partitions). */
  halos: number[][];
  /** Flattened P×P comm matrix (cross-edge counts). */
  commMatrix: number[];
  /** Total edges crossing partitions. */
  edgeCut: number;
  /** Imbalance = (max-mean)/mean. */
  imbalance: number;
}

export function partitionTopology(graph: TopologyGraph, partitionCount: number): TopologyPartitionPlan {
  const N = graph.nodes.length;
  const P = Math.max(1, partitionCount);
  const owners = new Int32Array(N).fill(-1);
  if (N === 0) {
    return { partitionCount: P, owners, resident: Array.from({ length: P }, () => []), halos: Array.from({ length: P }, () => []), commMatrix: new Array(P * P).fill(0), edgeCut: 0, imbalance: 0 };
  }

  // Farthest-point seeding (graph-distance).
  const seeds: number[] = [0];
  const distTo = new Int32Array(N).fill(Number.MAX_SAFE_INTEGER);
  bfsDist(graph, 0, distTo);
  while (seeds.length < P) {
    let bestI = -1, bestD = -1;
    for (let i = 0; i < N; i++) if (distTo[i] > bestD) { bestD = distTo[i]; bestI = i; }
    if (bestI < 0) break;
    seeds.push(bestI);
    const tmp = new Int32Array(N).fill(Number.MAX_SAFE_INTEGER);
    bfsDist(graph, bestI, tmp);
    for (let i = 0; i < N; i++) if (tmp[i] < distTo[i]) distTo[i] = tmp[i];
  }
  while (seeds.length < P) seeds.push(seeds[seeds.length - 1]); // pathological tiny graphs

  // Multi-source BFS, weighted by shared face area (priority via deterministic ordering).
  const queue: { p: number; i: number; w: number }[] = [];
  for (let p = 0; p < P; p++) {
    owners[seeds[p]] = p;
    queue.push({ p, i: seeds[p], w: 0 });
  }
  // FIFO with tie-break — sufficient + deterministic for our scale.
  const target = Math.ceil(N / P) + 1;
  const sizes = new Int32Array(P);
  for (let p = 0; p < P; p++) sizes[p] = 1;

  while (queue.length) {
    const { p, i } = queue.shift()!;
    const s = graph.neighborOffsets[i];
    const e = graph.neighborOffsets[i + 1];
    // Sort neighbors by shared-face area descending for locality preservation.
    const order: { j: number; w: number }[] = [];
    for (let k = s; k < e; k++) {
      order.push({ j: graph.neighborIdx[k], w: graph.edges[graph.neighborEdge[k]].shared });
    }
    order.sort((a, b) => b.w - a.w || a.j - b.j);
    for (const { j, w } of order) {
      if (owners[j] !== -1) continue;
      if (sizes[p] >= target) continue;
      owners[j] = p;
      sizes[p]++;
      queue.push({ p, i: j, w });
    }
  }
  // Sweep: assign any orphans to neighbor majority or smallest partition.
  for (let i = 0; i < N; i++) {
    if (owners[i] !== -1) continue;
    const tally = new Int32Array(P);
    const s = graph.neighborOffsets[i];
    const e = graph.neighborOffsets[i + 1];
    for (let k = s; k < e; k++) {
      const o = owners[graph.neighborIdx[k]];
      if (o >= 0) tally[o]++;
    }
    let best = 0;
    for (let p = 1; p < P; p++) if (tally[p] > tally[best] || (tally[p] === tally[best] && sizes[p] < sizes[best])) best = p;
    owners[i] = best;
    sizes[best]++;
  }

  const resident: number[][] = Array.from({ length: P }, () => []);
  for (let i = 0; i < N; i++) resident[owners[i]].push(i);

  // Halos + comm matrix from edges.
  const haloSets: Set<number>[] = Array.from({ length: P }, () => new Set());
  const commMatrix = new Array(P * P).fill(0);
  let edgeCut = 0;
  for (const e of graph.edges) {
    const oa = owners[e.a];
    const ob = owners[e.b];
    if (oa === ob) continue;
    edgeCut++;
    haloSets[oa].add(e.b);
    haloSets[ob].add(e.a);
    commMatrix[oa * P + ob]++;
    commMatrix[ob * P + oa]++;
  }
  const halos = haloSets.map((s) => Array.from(s));

  const arrSizes = Array.from(sizes);
  const mean = arrSizes.reduce((a, b) => a + b, 0) / Math.max(1, P);
  const max = Math.max(...arrSizes, 0);
  const imbalance = mean > 0 ? (max - mean) / mean : 0;

  return { partitionCount: P, owners, resident, halos, commMatrix, edgeCut, imbalance };
}

function bfsDist(graph: TopologyGraph, src: number, out: Int32Array) {
  for (let i = 0; i < out.length; i++) out[i] = Number.MAX_SAFE_INTEGER;
  out[src] = 0;
  const queue = [src];
  while (queue.length) {
    const i = queue.shift()!;
    const s = graph.neighborOffsets[i];
    const e = graph.neighborOffsets[i + 1];
    for (let k = s; k < e; k++) {
      const j = graph.neighborIdx[k];
      if (out[j] !== Number.MAX_SAFE_INTEGER) continue;
      out[j] = out[i] + 1;
      queue.push(j);
    }
  }
}
