/**
 * Geometry partitioning algorithms.
 *
 * Each algorithm takes the tet adjacency CSR + per-tet centroids and emits a
 * partition assignment plus statistics. Algorithms:
 *   - regionGrow: BFS from strided seeds (geometric coherence, fast)
 *   - spectral: Fiedler vector of the (lazy) Laplacian via power iteration
 *               on the random-walk normalized adjacency, recursively bisected
 *   - morton: Morton-order strip partition (cache-local, comm-suboptimal)
 *   - kway: Kernighan–Lin style refinement on a region-grow seed
 *
 * All return a `tetPart: Uint16Array` with values in [0, P).
 *
 * Pure TypeScript, deterministic given equal inputs.
 */

import type { OctreeMesh, Vec3 } from "../meshing/octree";
import type { AdjacencyTensors } from "../meshing/adjacency";

export type PartitionAlgorithm = "regionGrow" | "spectral" | "morton" | "kway";

export interface PartitionStats {
  algorithm: PartitionAlgorithm;
  edgeCut: number;
  imbalance: number;
  sizes: Uint32Array;
  buildMs: number;
}

export interface PartitionAssignment {
  tetPart: Uint16Array;
  stats: PartitionStats;
}

function tetCentroid(mesh: OctreeMesh, t: number): Vec3 {
  const i0 = mesh.tets[t * 4 + 0] * 3;
  const i1 = mesh.tets[t * 4 + 1] * 3;
  const i2 = mesh.tets[t * 4 + 2] * 3;
  const i3 = mesh.tets[t * 4 + 3] * 3;
  const v = mesh.vertices;
  return [
    (v[i0] + v[i1] + v[i2] + v[i3]) / 4,
    (v[i0 + 1] + v[i1 + 1] + v[i2 + 1] + v[i3 + 1]) / 4,
    (v[i0 + 2] + v[i1 + 2] + v[i2 + 2] + v[i3 + 2]) / 4,
  ];
}

function computeStats(
  algo: PartitionAlgorithm,
  P: number,
  tetPart: Uint16Array,
  adj: AdjacencyTensors,
  t0: number,
): PartitionStats {
  const T = tetPart.length;
  const sizes = new Uint32Array(P);
  for (let t = 0; t < T; t++) sizes[tetPart[t]]++;
  let cut = 0;
  for (let t = 0; t < T; t++) {
    const pt = tetPart[t];
    const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
    for (let k = s; k < e; k++) if (tetPart[adj.ttCol[k]] !== pt) cut++;
  }
  let max = 0;
  for (let p = 0; p < P; p++) if (sizes[p] > max) max = sizes[p];
  const mean = T / P;
  return {
    algorithm: algo,
    edgeCut: cut / 2,
    imbalance: mean ? max / mean : 1,
    sizes,
    buildMs: Date.now() - t0,
  };
}

export function regionGrowPartition(
  mesh: OctreeMesh,
  adj: AdjacencyTensors,
  P: number,
): PartitionAssignment {
  const t0 = Date.now();
  const T = mesh.tets.length / 4;
  const tetPart = new Uint16Array(T).fill(0xffff);
  const stride = Math.max(1, Math.floor(T / P));
  const sizes = new Uint32Array(P);
  const frontiers: number[][] = [];
  for (let p = 0; p < P; p++) {
    const s = Math.min(T - 1, p * stride);
    tetPart[s] = p;
    sizes[p] = 1;
    frontiers.push([s]);
  }
  const cap = Math.ceil((T / P) * 1.1);
  let assigned = P;
  while (assigned < T) {
    let progressed = false;
    for (let p = 0; p < P; p++) {
      if (sizes[p] >= cap) continue;
      const f = frontiers[p];
      const next: number[] = [];
      for (const t of f) {
        const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
        for (let k = s; k < e; k++) {
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
  for (let t = 0; t < T; t++) {
    if (tetPart[t] === 0xffff) {
      let m = 0;
      for (let p = 1; p < P; p++) if (sizes[p] < sizes[m]) m = p;
      tetPart[t] = m;
      sizes[m]++;
    }
  }
  return { tetPart, stats: computeStats("regionGrow", P, tetPart, adj, t0) };
}

/** Morton key from quantized centroid (21 bits per axis). */
function mortonKey(p: Vec3, lo: Vec3, scale: number): bigint {
  const x = Math.max(0, Math.min(0x1fffff, Math.floor((p[0] - lo[0]) * scale)));
  const y = Math.max(0, Math.min(0x1fffff, Math.floor((p[1] - lo[1]) * scale)));
  const z = Math.max(0, Math.min(0x1fffff, Math.floor((p[2] - lo[2]) * scale)));
  let k = 0n;
  for (let i = 0; i < 21; i++) {
    const b = BigInt(i);
    k |= (BigInt((x >> i) & 1) << (3n * b));
    k |= (BigInt((y >> i) & 1) << (3n * b + 1n));
    k |= (BigInt((z >> i) & 1) << (3n * b + 2n));
  }
  return k;
}

export function mortonPartition(
  mesh: OctreeMesh,
  adj: AdjacencyTensors,
  P: number,
): PartitionAssignment {
  const t0 = Date.now();
  const T = mesh.tets.length / 4;
  const lo: Vec3 = [mesh.bbox.min[0], mesh.bbox.min[1], mesh.bbox.min[2]];
  const ext = Math.max(
    mesh.bbox.max[0] - mesh.bbox.min[0],
    mesh.bbox.max[1] - mesh.bbox.min[1],
    mesh.bbox.max[2] - mesh.bbox.min[2],
    1e-9,
  );
  const scale = (1 << 20) / ext;
  const keys = new Array<{ t: number; k: bigint }>(T);
  for (let t = 0; t < T; t++) keys[t] = { t, k: mortonKey(tetCentroid(mesh, t), lo, scale) };
  keys.sort((a, b) => (a.k < b.k ? -1 : a.k > b.k ? 1 : a.t - b.t));
  const tetPart = new Uint16Array(T);
  const chunk = Math.ceil(T / P);
  for (let i = 0; i < T; i++) tetPart[keys[i].t] = Math.min(P - 1, Math.floor(i / chunk));
  return { tetPart, stats: computeStats("morton", P, tetPart, adj, t0) };
}

/**
 * Spectral bisection via power iteration on the lazy random-walk Laplacian
 * (I - D^-1 A) with the principal eigenvector deflated. We recurse to depth
 * log2(P), assigning sign(Fiedler) at each split.
 */
export function spectralPartition(
  mesh: OctreeMesh,
  adj: AdjacencyTensors,
  P: number,
): PartitionAssignment {
  const t0 = Date.now();
  const T = mesh.tets.length / 4;
  const tetPart = new Uint16Array(T);
  const groups: number[][] = [Array.from({ length: T }, (_, i) => i)];
  let depth = 0;
  const maxDepth = Math.ceil(Math.log2(Math.max(1, P)));

  while (depth < maxDepth && groups.length < P) {
    const next: number[][] = [];
    for (const g of groups) {
      if (g.length < 4 || groups.length + next.length >= P) {
        next.push(g);
        continue;
      }
      const [a, b] = fiedlerBisect(g, adj);
      if (a.length === 0 || b.length === 0) { next.push(g); continue; }
      next.push(a, b);
    }
    groups.length = 0;
    groups.push(...next);
    depth++;
  }

  // Pad to P groups via splitting the largest, or merge smallest if too many.
  while (groups.length < P) {
    let li = 0;
    for (let i = 1; i < groups.length; i++) if (groups[i].length > groups[li].length) li = i;
    const big = groups[li];
    if (big.length < 2) break;
    const half = Math.floor(big.length / 2);
    groups[li] = big.slice(0, half);
    groups.push(big.slice(half));
  }
  while (groups.length > P) {
    groups.sort((x, y) => x.length - y.length);
    const a = groups.shift()!, b = groups.shift()!;
    groups.unshift(a.concat(b));
  }
  groups.forEach((g, p) => { for (const t of g) tetPart[t] = p; });
  return { tetPart, stats: computeStats("spectral", P, tetPart, adj, t0) };
}

function fiedlerBisect(group: number[], adj: AdjacencyTensors): [number[], number[]] {
  const n = group.length;
  const idx = new Map<number, number>();
  group.forEach((t, i) => idx.set(t, i));
  // Power iteration with deflation of the constant vector.
  let v = new Float32Array(n);
  for (let i = 0; i < n; i++) v[i] = ((i * 2654435761) >>> 0) / 0xffffffff - 0.5;
  // Subtract mean.
  const sub = (x: Float32Array) => {
    let m = 0;
    for (let i = 0; i < n; i++) m += x[i];
    m /= n;
    for (let i = 0; i < n; i++) x[i] -= m;
  };
  const norm = (x: Float32Array) => {
    let s = 0;
    for (let i = 0; i < n; i++) s += x[i] * x[i];
    s = Math.sqrt(Math.max(s, 1e-30));
    for (let i = 0; i < n; i++) x[i] /= s;
  };
  const apply = (x: Float32Array, out: Float32Array) => {
    // Lazy walk: y_i = 0.5*x_i + 0.5*mean(x_neighbors_in_subgraph).
    for (let i = 0; i < n; i++) {
      const t = group[i];
      const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
      let sum = 0, deg = 0;
      for (let k = s; k < e; k++) {
        const j = idx.get(adj.ttCol[k]);
        if (j !== undefined) { sum += x[j]; deg++; }
      }
      out[i] = 0.5 * x[i] + 0.5 * (deg ? sum / deg : x[i]);
    }
  };
  sub(v); norm(v);
  const tmp = new Float32Array(n);
  for (let it = 0; it < 24; it++) {
    apply(v, tmp);
    sub(tmp);
    norm(tmp);
    [v, tmp[0]] as unknown; // no-op
    v.set(tmp);
  }
  // Median split for balanced bisection.
  const sorted = Array.from(v).slice().sort((a, b) => a - b);
  const median = sorted[Math.floor(n / 2)];
  const a: number[] = [], b: number[] = [];
  for (let i = 0; i < n; i++) (v[i] <= median ? a : b).push(group[i]);
  return [a, b];
}

/** K-way Kernighan–Lin: swap pairs of boundary tets that reduce edge cut. */
export function kwayRefine(
  base: PartitionAssignment,
  adj: AdjacencyTensors,
  passes = 4,
): PartitionAssignment {
  const t0 = Date.now();
  const tetPart = new Uint16Array(base.tetPart);
  const T = tetPart.length;
  const P = base.stats.sizes.length;
  const sizes = new Uint32Array(base.stats.sizes);
  const target = T / P;

  const externalCount = (t: number, part: number): number => {
    const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
    let c = 0;
    for (let k = s; k < e; k++) if (tetPart[adj.ttCol[k]] !== part) c++;
    return c;
  };
  const bestNeighborPart = (t: number): number => {
    const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
    const counts = new Map<number, number>();
    for (let k = s; k < e; k++) {
      const p = tetPart[adj.ttCol[k]];
      counts.set(p, (counts.get(p) ?? 0) + 1);
    }
    let bestP = tetPart[t], bestC = -1;
    for (const [p, c] of counts) if (c > bestC) { bestC = c; bestP = p; }
    return bestP;
  };

  for (let pass = 0; pass < passes; pass++) {
    let moved = 0;
    for (let t = 0; t < T; t++) {
      const cur = tetPart[t];
      const target2 = bestNeighborPart(t);
      if (target2 === cur) continue;
      const before = externalCount(t, cur);
      const after = externalCount(t, target2);
      if (after >= before) continue;
      // Balance guard.
      if (sizes[target2] + 1 > target * 1.15) continue;
      if (sizes[cur] - 1 < target * 0.85) continue;
      sizes[cur]--; sizes[target2]++;
      tetPart[t] = target2;
      moved++;
    }
    if (moved === 0) break;
  }
  return { tetPart, stats: computeStats("kway", P, tetPart, adj, t0) };
}

export function runAlgorithm(
  algo: PartitionAlgorithm,
  mesh: OctreeMesh,
  adj: AdjacencyTensors,
  P: number,
): PartitionAssignment {
  switch (algo) {
    case "regionGrow": return regionGrowPartition(mesh, adj, P);
    case "morton": return mortonPartition(mesh, adj, P);
    case "spectral": return spectralPartition(mesh, adj, P);
    case "kway": {
      const seed = regionGrowPartition(mesh, adj, P);
      return kwayRefine(seed, adj);
    }
  }
}
