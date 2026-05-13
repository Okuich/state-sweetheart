/**
 * Linear BVH (LBVH) over axis-aligned bounding boxes.
 *
 * Build pipeline (GPU-friendly, branch-minimal):
 *   centroids → Morton codes → sort → top-down median split.
 *
 * Stored as flat SoA arrays for warp-coalesced traversal:
 *   nodeMin[3*N], nodeMax[3*N], left[N], right[N], primFirst[N], primCount[N].
 *
 * `traverse()` is iterative (explicit stack), no recursion — mirrors a
 * persistent traversal kernel.
 */

import { mortonCodes, mortonOrder } from "./morton";

export type Vec3 = readonly [number, number, number];
export interface AABB { min: Vec3; max: Vec3 }

export interface BVH {
  primCount: number;
  nodeCount: number;
  nodeMin: Float32Array;   // 3*N
  nodeMax: Float32Array;   // 3*N
  left: Int32Array;        // -1 if leaf
  right: Int32Array;       // -1 if leaf
  primFirst: Uint32Array;  // start index into primOrder (leaf only)
  primCount_: Uint16Array; // 0 if internal
  primOrder: Uint32Array;  // sorted primitive indices
  primMin: Float32Array;   // 3*P
  primMax: Float32Array;   // 3*P
  buildMs: number;
  meanLeafSize: number;
  maxDepth: number;
}

const LEAF_THRESHOLD = 4;

function aabbUnion(out: Float32Array, oi: number, a: Float32Array, ai: number, b: Float32Array, bi: number) {
  out[oi]     = Math.min(a[ai],     b[bi]);
  out[oi + 1] = Math.min(a[ai + 1], b[bi + 1]);
  out[oi + 2] = Math.min(a[ai + 2], b[bi + 2]);
}

export function buildBVH(primMin: Float32Array, primMax: Float32Array): BVH {
  const t0 = Date.now();
  const P = primMin.length / 3;
  if (P === 0) throw new Error("BVH: empty primitive set");

  // Scene bbox + per-prim centroids.
  const sceneMin: [number, number, number] = [Infinity, Infinity, Infinity];
  const sceneMax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
  const centroids = new Float32Array(P * 3);
  for (let i = 0; i < P; i++) {
    for (let k = 0; k < 3; k++) {
      const lo = primMin[i * 3 + k];
      const hi = primMax[i * 3 + k];
      const c = (lo + hi) * 0.5;
      centroids[i * 3 + k] = c;
      if (lo < sceneMin[k]) sceneMin[k] = lo;
      if (hi > sceneMax[k]) sceneMax[k] = hi;
    }
  }

  const codes = mortonCodes(centroids, { min: sceneMin, max: sceneMax });
  const primOrder = mortonOrder(codes);

  // Pre-allocate generously: at most 2P-1 nodes for a balanced tree, but
  // with leaf-threshold > 1 we have fewer; cap at 2P to be safe.
  const cap = Math.max(8, 2 * P);
  const nodeMin = new Float32Array(cap * 3);
  const nodeMax = new Float32Array(cap * 3);
  const left = new Int32Array(cap);
  const right = new Int32Array(cap);
  const primFirst = new Uint32Array(cap);
  const primCount_ = new Uint16Array(cap);
  let nodeCount = 0;
  let maxDepth = 0;

  // Iterative top-down build using explicit task stack.
  type Task = { first: number; count: number; depth: number; nodeId: number };
  const root = nodeCount++;
  const stack: Task[] = [{ first: 0, count: P, depth: 0, nodeId: root }];

  while (stack.length) {
    const { first, count, depth, nodeId } = stack.pop()!;
    if (depth > maxDepth) maxDepth = depth;

    // Compute node bbox over its primitives.
    let lx = Infinity, ly = Infinity, lz = Infinity;
    let hx = -Infinity, hy = -Infinity, hz = -Infinity;
    for (let i = 0; i < count; i++) {
      const p = primOrder[first + i];
      lx = Math.min(lx, primMin[p * 3]);
      ly = Math.min(ly, primMin[p * 3 + 1]);
      lz = Math.min(lz, primMin[p * 3 + 2]);
      hx = Math.max(hx, primMax[p * 3]);
      hy = Math.max(hy, primMax[p * 3 + 1]);
      hz = Math.max(hz, primMax[p * 3 + 2]);
    }
    nodeMin[nodeId * 3] = lx; nodeMin[nodeId * 3 + 1] = ly; nodeMin[nodeId * 3 + 2] = lz;
    nodeMax[nodeId * 3] = hx; nodeMax[nodeId * 3 + 1] = hy; nodeMax[nodeId * 3 + 2] = hz;

    if (count <= LEAF_THRESHOLD || depth >= 30) {
      left[nodeId] = -1;
      right[nodeId] = -1;
      primFirst[nodeId] = first;
      primCount_[nodeId] = count;
      continue;
    }

    // Median split along primitives already Morton-sorted ⇒ spatial coherence.
    const half = count >> 1;
    const lId = nodeCount++;
    const rId = nodeCount++;
    left[nodeId] = lId;
    right[nodeId] = rId;
    primFirst[nodeId] = 0;
    primCount_[nodeId] = 0;
    stack.push({ first: first + half, count: count - half, depth: depth + 1, nodeId: rId });
    stack.push({ first, count: half, depth: depth + 1, nodeId: lId });
  }

  void aabbUnion;

  let leafSum = 0;
  let leafCount = 0;
  for (let i = 0; i < nodeCount; i++) {
    if (left[i] === -1) { leafSum += primCount_[i]; leafCount++; }
  }

  return {
    primCount: P,
    nodeCount,
    nodeMin: nodeMin.subarray(0, nodeCount * 3),
    nodeMax: nodeMax.subarray(0, nodeCount * 3),
    left: left.subarray(0, nodeCount),
    right: right.subarray(0, nodeCount),
    primFirst: primFirst.subarray(0, nodeCount),
    primCount_: primCount_.subarray(0, nodeCount),
    primOrder,
    primMin,
    primMax,
    buildMs: Date.now() - t0,
    meanLeafSize: leafCount ? leafSum / leafCount : 0,
    maxDepth,
  };
}

/** Iterative AABB-overlap traversal. Returns prim ids whose AABB overlaps the query. */
export function bvhQueryAABB(bvh: BVH, qMin: Vec3, qMax: Vec3, out: number[] = []): number[] {
  const stack: number[] = [0];
  while (stack.length) {
    const n = stack.pop()!;
    const i3 = n * 3;
    if (
      bvh.nodeMax[i3]     < qMin[0] || bvh.nodeMin[i3]     > qMax[0] ||
      bvh.nodeMax[i3 + 1] < qMin[1] || bvh.nodeMin[i3 + 1] > qMax[1] ||
      bvh.nodeMax[i3 + 2] < qMin[2] || bvh.nodeMin[i3 + 2] > qMax[2]
    ) continue;
    if (bvh.left[n] === -1) {
      const f = bvh.primFirst[n];
      const c = bvh.primCount_[n];
      for (let k = 0; k < c; k++) {
        const p = bvh.primOrder[f + k];
        const p3 = p * 3;
        if (
          bvh.primMax[p3]     >= qMin[0] && bvh.primMin[p3]     <= qMax[0] &&
          bvh.primMax[p3 + 1] >= qMin[1] && bvh.primMin[p3 + 1] <= qMax[1] &&
          bvh.primMax[p3 + 2] >= qMin[2] && bvh.primMin[p3 + 2] <= qMax[2]
        ) out.push(p);
      }
    } else {
      stack.push(bvh.left[n], bvh.right[n]);
    }
  }
  return out;
}

/** All overlapping pairs (i<j). Naïve self-traversal; fine for ≤ 50k prims. */
export function bvhAllPairs(bvh: BVH, max = 1_000_000): Uint32Array {
  const pairs: number[] = [];
  for (let i = 0; i < bvh.primCount && pairs.length / 2 < max; i++) {
    const i3 = i * 3;
    const qMin: Vec3 = [bvh.primMin[i3], bvh.primMin[i3 + 1], bvh.primMin[i3 + 2]];
    const qMax: Vec3 = [bvh.primMax[i3], bvh.primMax[i3 + 1], bvh.primMax[i3 + 2]];
    const hits: number[] = [];
    bvhQueryAABB(bvh, qMin, qMax, hits);
    for (const j of hits) {
      if (j > i) { pairs.push(i, j); if (pairs.length / 2 >= max) break; }
    }
  }
  return new Uint32Array(pairs);
}
