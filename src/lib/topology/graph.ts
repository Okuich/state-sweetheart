/**
 * Topology graph construction.
 *
 * Walks an OctreeMesh and produces a leaf-adjacency graph (nodes = leaves,
 * edges = face-sharing pairs). Adjacency is detected via face-vertex hashing:
 * leaves whose bounding boxes share at least 3 vertices in the deduped vertex
 * pool also share a face.
 *
 * The graph is the substrate for everything downstream: feature classification
 * (`features.ts`), manufacturability scoring (`manufacturability.ts`),
 * structural embeddings (`embeddings.ts`), and distributed partitioning
 * (`partition.ts`).
 */

import type { OctreeMesh } from "../meshing/octree";
import type { TopoEdge, TopoNode, Vec3 } from "./types";

export interface TopologyGraph {
  nodes: TopoNode[];
  edges: TopoEdge[];
  /** CSR-style neighbor offsets for fast traversal. */
  neighborOffsets: Uint32Array;
  neighborIdx: Uint32Array;
  /** Edge index per (a→b) entry in neighborIdx (parallel array). */
  neighborEdge: Uint32Array;
  buildMs: number;
}

function leafCenter(mesh: OctreeMesh, leafIdx: number): Vec3 {
  const n = mesh.nodes[mesh.leaves[leafIdx]];
  return [
    (n.bbox.min[0] + n.bbox.max[0]) / 2,
    (n.bbox.min[1] + n.bbox.max[1]) / 2,
    (n.bbox.min[2] + n.bbox.max[2]) / 2,
  ];
}

function leafExtent(mesh: OctreeMesh, leafIdx: number): Vec3 {
  const n = mesh.nodes[mesh.leaves[leafIdx]];
  return [
    n.bbox.max[0] - n.bbox.min[0],
    n.bbox.max[1] - n.bbox.min[1],
    n.bbox.max[2] - n.bbox.min[2],
  ];
}

/** Build the leaf adjacency graph. Pure-TS, deterministic. */
export function buildTopologyGraph(mesh: OctreeMesh): TopologyGraph {
  const t0 = Date.now();
  const L = mesh.leaves.length;
  const nodes: TopoNode[] = new Array(L);

  // Construct nodes (skeletons; features are filled by features.ts).
  for (let i = 0; i < L; i++) {
    const n = mesh.nodes[mesh.leaves[i]];
    const ext = leafExtent(mesh, i);
    const radius = 0.5 * Math.hypot(ext[0], ext[1], ext[2]);
    nodes[i] = {
      leaf: i,
      center: leafCenter(mesh, i),
      radius,
      density: n.density,
      tag: n.tag,
      feature: "bulk",
      wallThickness: 0,
      curvature: 0,
      boundary: mesh.boundaryLeaf[i] === 1,
      downward: false,
    };
  }

  // Face-sharing detection via hashed face keys. Each leaf has 6 axis-aligned
  // faces; a face is uniquely keyed by (axis, fixedCoord, lo1, hi1, lo2, hi2).
  // Two leaves share a face iff they share the same key on opposite sides.
  // Hashing uses string concatenation with quantized floats (1e-6 precision).
  const Q = 1_000_000;
  const q = (x: number) => Math.round(x * Q);
  const faceMap = new Map<string, { leaf: number; axis: 0 | 1 | 2 }[]>();
  for (let i = 0; i < L; i++) {
    const n = mesh.nodes[mesh.leaves[i]];
    const { min, max } = n.bbox;
    // Axis 0: faces at min[0] and max[0]
    pushFace(faceMap, `0|${q(min[0])}|${q(min[1])}|${q(max[1])}|${q(min[2])}|${q(max[2])}`, i, 0);
    pushFace(faceMap, `0|${q(max[0])}|${q(min[1])}|${q(max[1])}|${q(min[2])}|${q(max[2])}`, i, 0);
    pushFace(faceMap, `1|${q(min[1])}|${q(min[0])}|${q(max[0])}|${q(min[2])}|${q(max[2])}`, i, 1);
    pushFace(faceMap, `1|${q(max[1])}|${q(min[0])}|${q(max[0])}|${q(min[2])}|${q(max[2])}`, i, 1);
    pushFace(faceMap, `2|${q(min[2])}|${q(min[0])}|${q(max[0])}|${q(min[1])}|${q(max[1])}`, i, 2);
    pushFace(faceMap, `2|${q(max[2])}|${q(min[0])}|${q(max[0])}|${q(min[1])}|${q(max[1])}`, i, 2);
  }

  const edges: TopoEdge[] = [];
  const seenPair = new Set<string>();
  for (const [, list] of faceMap) {
    if (list.length < 2) continue;
    for (let i = 0; i < list.length; i++) {
      for (let j = i + 1; j < list.length; j++) {
        const a = Math.min(list[i].leaf, list[j].leaf);
        const b = Math.max(list[i].leaf, list[j].leaf);
        if (a === b) continue;
        const key = `${a}|${b}`;
        if (seenPair.has(key)) continue;
        seenPair.add(key);
        const axis = list[i].axis;
        // Shared face area: product of the two non-axis extents at the smaller leaf.
        const ea = leafExtent(mesh, a);
        const eb = leafExtent(mesh, b);
        const ax1 = (axis + 1) % 3;
        const ax2 = (axis + 2) % 3;
        const shared = Math.min(ea[ax1], eb[ax1]) * Math.min(ea[ax2], eb[ax2]);
        edges.push({ a, b, shared, axis });
      }
    }
  }

  // Build CSR neighbor index.
  const deg = new Uint32Array(L);
  for (const e of edges) { deg[e.a]++; deg[e.b]++; }
  const neighborOffsets = new Uint32Array(L + 1);
  for (let i = 0; i < L; i++) neighborOffsets[i + 1] = neighborOffsets[i] + deg[i];
  const cursor = new Uint32Array(L);
  const total = neighborOffsets[L];
  const neighborIdx = new Uint32Array(total);
  const neighborEdge = new Uint32Array(total);
  edges.forEach((e, ei) => {
    const pa = neighborOffsets[e.a] + cursor[e.a]++;
    const pb = neighborOffsets[e.b] + cursor[e.b]++;
    neighborIdx[pa] = e.b;
    neighborIdx[pb] = e.a;
    neighborEdge[pa] = ei;
    neighborEdge[pb] = ei;
  });

  // Compute curvature (face-deficit) per node — boundary faces contribute.
  for (let i = 0; i < L; i++) {
    const fn = neighborOffsets[i + 1] - neighborOffsets[i];
    nodes[i].curvature = Math.max(0, 6 - fn);
  }

  return { nodes, edges, neighborOffsets, neighborIdx, neighborEdge, buildMs: Date.now() - t0 };
}

function pushFace(
  m: Map<string, { leaf: number; axis: 0 | 1 | 2 }[]>,
  key: string,
  leaf: number,
  axis: 0 | 1 | 2,
) {
  const e = m.get(key);
  if (e) e.push({ leaf, axis });
  else m.set(key, [{ leaf, axis }]);
}

/** Iterate neighbors of node `i`. */
export function neighborsOf(g: TopologyGraph, i: number): Iterable<number> {
  const start = g.neighborOffsets[i];
  const end = g.neighborOffsets[i + 1];
  return {
    [Symbol.iterator]() {
      let k = start;
      return {
        next(): IteratorResult<number> {
          if (k < end) return { value: g.neighborIdx[k++], done: false };
          return { value: 0, done: true };
        },
      };
    },
  };
}
