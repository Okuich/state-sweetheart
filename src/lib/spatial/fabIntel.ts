/**
 * Fabrication intelligence built on the spatial acceleration layer.
 * Inputs are the octree mesh from src/lib/meshing — leaves classified as
 * boundary, interior, or unsupported.
 *
 * Detects:
 *   - trapped volumes (interior leaves disconnected from any boundary)
 *   - unsupported overhangs (boundary leaf with no neighbor below)
 *   - inaccessible regions (interior leaves > N rings from boundary)
 *   - collision-prone clusters (high boundary leaf concentration)
 */

import type { OctreeMesh } from "../meshing/octree";
import type { AdjacencyTensors } from "../meshing/adjacency";

export interface FabIntelReport {
  trappedVolumeLeaves: number;
  trappedClusters: number;
  overhangLeaves: number;
  inaccessibleLeaves: number;
  collisionClusters: number;
  /** Per-leaf classification, length = leaf count. */
  classification: Uint8Array;
  notes: string[];
  ms: number;
}

const CLS_INTERIOR = 0;
const CLS_BOUNDARY = 1;
const CLS_OVERHANG = 2;
const CLS_TRAPPED = 3;
const CLS_INACCESSIBLE = 4;

/** Build per-leaf adjacency by aggregating tet-tet adjacency through tetLeaf. */
function leafAdjacency(mesh: OctreeMesh, adj: AdjacencyTensors): Uint32Array[] {
  const L = mesh.leaves.length;
  const sets: Set<number>[] = Array.from({ length: L }, () => new Set<number>());
  const T = mesh.tets.length / 4;
  for (let t = 0; t < T; t++) {
    const lt = mesh.tetLeaf[t];
    const start = adj.ttRowPtr[t], end = adj.ttRowPtr[t + 1];
    for (let k = start; k < end; k++) {
      const ln = mesh.tetLeaf[adj.ttCol[k]];
      if (ln !== lt) sets[lt].add(ln);
    }
  }
  return sets.map((s) => Uint32Array.from(s));
}

export function analyzeFabrication(mesh: OctreeMesh, adj: AdjacencyTensors): FabIntelReport {
  const t0 = Date.now();
  const L = mesh.leaves.length;
  const cls = new Uint8Array(L);
  const ladj = leafAdjacency(mesh, adj);

  // Initial classification: boundary vs interior from mesh.boundaryLeaf flag.
  for (let i = 0; i < L; i++) cls[i] = mesh.boundaryLeaf[i] ? CLS_BOUNDARY : CLS_INTERIOR;

  // Overhang detection: a boundary leaf with no neighbor in the −Z half-space.
  let overhangs = 0;
  for (let i = 0; i < L; i++) {
    if (cls[i] !== CLS_BOUNDARY) continue;
    const n = mesh.nodes[mesh.leaves[i]];
    const cz = (n.bbox.min[2] + n.bbox.max[2]) / 2;
    let hasSupportBelow = false;
    for (const j of ladj[i]) {
      const nj = mesh.nodes[mesh.leaves[j]];
      const cjz = (nj.bbox.min[2] + nj.bbox.max[2]) / 2;
      if (cjz < cz - 1e-9) { hasSupportBelow = true; break; }
    }
    if (!hasSupportBelow) { cls[i] = CLS_OVERHANG; overhangs++; }
  }

  // Trapped volumes: BFS over interior leaves; any component with no path
  // to a boundary leaf is trapped. Count clusters too.
  const visited = new Uint8Array(L);
  let trappedClusters = 0;
  let trappedLeaves = 0;
  for (let s = 0; s < L; s++) {
    if (visited[s] || cls[s] !== CLS_INTERIOR) continue;
    const queue = [s];
    visited[s] = 1;
    let touchesBoundary = false;
    const component: number[] = [];
    while (queue.length) {
      const v = queue.pop()!;
      component.push(v);
      for (const w of ladj[v]) {
        if (cls[w] === CLS_BOUNDARY || cls[w] === CLS_OVERHANG) { touchesBoundary = true; continue; }
        if (!visited[w] && cls[w] === CLS_INTERIOR) { visited[w] = 1; queue.push(w); }
      }
    }
    if (!touchesBoundary) {
      trappedClusters++;
      for (const v of component) { cls[v] = CLS_TRAPPED; trappedLeaves++; }
    }
  }

  // Inaccessible: interior leaves > 3 hops from any boundary.
  const dist = new Int32Array(L).fill(-1);
  const front: number[] = [];
  for (let i = 0; i < L; i++) {
    if (cls[i] === CLS_BOUNDARY || cls[i] === CLS_OVERHANG) { dist[i] = 0; front.push(i); }
  }
  while (front.length) {
    const v = front.shift()!;
    for (const w of ladj[v]) {
      if (dist[w] === -1) { dist[w] = dist[v] + 1; front.push(w); }
    }
  }
  let inaccessible = 0;
  for (let i = 0; i < L; i++) {
    if (cls[i] === CLS_INTERIOR && (dist[i] === -1 || dist[i] > 3)) {
      cls[i] = CLS_INACCESSIBLE; inaccessible++;
    }
  }

  // Collision-prone clusters: boundary leaves with ≥4 boundary neighbors.
  let collisionClusters = 0;
  for (let i = 0; i < L; i++) {
    if (cls[i] !== CLS_BOUNDARY) continue;
    let bn = 0;
    for (const j of ladj[i]) if (cls[j] === CLS_BOUNDARY || cls[j] === CLS_OVERHANG) bn++;
    if (bn >= 4) collisionClusters++;
  }

  const notes: string[] = [];
  if (trappedClusters) notes.push(`${trappedClusters} trapped pocket(s) — drainage required`);
  if (overhangs) notes.push(`${overhangs} unsupported overhang leaves — add supports or rotate`);
  if (inaccessible) notes.push(`${inaccessible} inaccessible leaves — tool reach concern`);
  if (collisionClusters > 5) notes.push(`${collisionClusters} dense boundary clusters — review fixturing`);
  if (notes.length === 0) notes.push("no fabrication blockers detected");

  return {
    trappedVolumeLeaves: trappedLeaves,
    trappedClusters,
    overhangLeaves: overhangs,
    inaccessibleLeaves: inaccessible,
    collisionClusters,
    classification: cls,
    notes,
    ms: Date.now() - t0,
  };
}

export const FAB_CLASS_LABELS = ["interior", "boundary", "overhang", "trapped", "inaccessible"] as const;
