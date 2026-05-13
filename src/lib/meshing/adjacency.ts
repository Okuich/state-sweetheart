/**
 * GPU-coalesced adjacency tensors for tetrahedral meshes.
 * Produces CSR-format vertex-to-vertex and tet-to-tet adjacency suitable
 * for warp-aligned access on GPU compute kernels.
 */

import type { OctreeMesh } from "./octree";

export interface AdjacencyTensors {
  /** CSR row pointer for vertex-vertex adjacency, length V+1. */
  vvRowPtr: Uint32Array;
  vvCol: Uint32Array;
  /** CSR for tet-tet face neighbors, length T+1. */
  ttRowPtr: Uint32Array;
  ttCol: Uint32Array;
  /** Edge list (compacted, src<dst), 2 entries per edge. */
  edges: Uint32Array;
  /** Tetrahedra adjacent across each edge, [tetA, tetB, ...] (-1 sentinel for boundary). */
  edgeStats: { count: number; meanDegree: number; maxDegree: number };
  /** Coalesced layout: per-vertex aligned slot count (for GPU). */
  warpStride: number;
  buildMs: number;
}

const WARP = 32;

function ceilTo(x: number, m: number): number {
  return Math.ceil(x / m) * m;
}

export function buildAdjacency(mesh: OctreeMesh): AdjacencyTensors {
  const t0 = Date.now();
  const tets = mesh.tets;
  const tCount = tets.length / 4;
  const vCount = mesh.vertices.length / 3;

  // 1. vertex-vertex adjacency via tet edges (sets to dedup).
  const vNeighbors: Set<number>[] = Array.from({ length: vCount }, () => new Set<number>());
  for (let t = 0; t < tCount; t++) {
    const i0 = tets[t * 4 + 0];
    const i1 = tets[t * 4 + 1];
    const i2 = tets[t * 4 + 2];
    const i3 = tets[t * 4 + 3];
    const ix = [i0, i1, i2, i3];
    for (let a = 0; a < 4; a++)
      for (let b = a + 1; b < 4; b++) {
        vNeighbors[ix[a]].add(ix[b]);
        vNeighbors[ix[b]].add(ix[a]);
      }
  }

  // CSR build.
  const vvRowPtr = new Uint32Array(vCount + 1);
  let total = 0;
  let maxDeg = 0;
  for (let i = 0; i < vCount; i++) {
    vvRowPtr[i] = total;
    total += vNeighbors[i].size;
    if (vNeighbors[i].size > maxDeg) maxDeg = vNeighbors[i].size;
  }
  vvRowPtr[vCount] = total;
  const vvCol = new Uint32Array(total);
  let cur = 0;
  for (let i = 0; i < vCount; i++) {
    const sorted = Array.from(vNeighbors[i]).sort((a, b) => a - b);
    for (const n of sorted) vvCol[cur++] = n;
  }

  // 2. tet-tet face neighbors.
  const faceMap = new Map<string, number[]>();
  const faceKey = (a: number, b: number, c: number) => {
    const s = [a, b, c].sort((x, y) => x - y);
    return `${s[0]},${s[1]},${s[2]}`;
  };
  for (let t = 0; t < tCount; t++) {
    const i0 = tets[t * 4 + 0];
    const i1 = tets[t * 4 + 1];
    const i2 = tets[t * 4 + 2];
    const i3 = tets[t * 4 + 3];
    for (const [a, b, c] of [
      [i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3],
    ]) {
      const k = faceKey(a, b, c);
      const arr = faceMap.get(k);
      if (arr) arr.push(t); else faceMap.set(k, [t]);
    }
  }
  const tNeighbors: Set<number>[] = Array.from({ length: tCount }, () => new Set<number>());
  faceMap.forEach((ts) => {
    if (ts.length === 2) {
      tNeighbors[ts[0]].add(ts[1]);
      tNeighbors[ts[1]].add(ts[0]);
    }
  });
  const ttRowPtr = new Uint32Array(tCount + 1);
  let tt = 0;
  for (let i = 0; i < tCount; i++) {
    ttRowPtr[i] = tt;
    tt += tNeighbors[i].size;
  }
  ttRowPtr[tCount] = tt;
  const ttCol = new Uint32Array(tt);
  let tcur = 0;
  for (let i = 0; i < tCount; i++) {
    const sorted = Array.from(tNeighbors[i]).sort((a, b) => a - b);
    for (const n of sorted) ttCol[tcur++] = n;
  }

  // 3. edge list (unique, src<dst).
  const edges: number[] = [];
  for (let i = 0; i < vCount; i++) {
    for (const j of vNeighbors[i]) if (j > i) edges.push(i, j);
  }

  return {
    vvRowPtr,
    vvCol,
    ttRowPtr,
    ttCol,
    edges: new Uint32Array(edges),
    edgeStats: {
      count: edges.length / 2,
      meanDegree: vCount ? total / vCount : 0,
      maxDegree: maxDeg,
    },
    warpStride: ceilTo(maxDeg, WARP),
    buildMs: Date.now() - t0,
  };
}
