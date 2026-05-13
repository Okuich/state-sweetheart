/**
 * Median-split KD-tree over points. Used for nearest-neighbor queries
 * (deformation neighborhoods, contact candidates, thermal propagation).
 * Stored as flat arrays — no allocations during queries.
 */

export type Vec3 = readonly [number, number, number];

export interface KDTree {
  count: number;
  /** Sorted point indices (DFS order). */
  order: Uint32Array;
  /** Splitting axis per internal node (0/1/2), -1 if leaf. */
  axis: Int8Array;
  /** Index into `order` of the median (this node's anchor point). */
  anchor: Uint32Array;
  /** Children. -1 means none. */
  left: Int32Array;
  right: Int32Array;
  /** Original point cloud, ref. */
  points: Float32Array;
  buildMs: number;
}

export function buildKDTree(points: Float32Array): KDTree {
  const t0 = Date.now();
  const n = points.length / 3;
  const cap = Math.max(1, n);
  const order = new Uint32Array(cap);
  for (let i = 0; i < cap; i++) order[i] = i;
  const axis = new Int8Array(cap);
  const anchor = new Uint32Array(cap);
  const left = new Int32Array(cap).fill(-1);
  const right = new Int32Array(cap).fill(-1);

  let nextNode = 0;
  type Task = { first: number; count: number; depth: number };
  const stack: Task[] = [{ first: 0, count: n, depth: 0 }];
  const nodeIds: number[] = [0];
  nextNode = 1;

  while (stack.length) {
    const { first, count, depth } = stack.pop()!;
    const nodeId = nodeIds.pop()!;
    if (count <= 0) { axis[nodeId] = -1; continue; }
    const ax = depth % 3;
    // Sort the slice of `order` by chosen axis.
    const slice = Array.from(order.subarray(first, first + count));
    slice.sort((a, b) => points[a * 3 + ax] - points[b * 3 + ax]);
    for (let i = 0; i < count; i++) order[first + i] = slice[i];
    const med = first + (count >> 1);
    axis[nodeId] = ax;
    anchor[nodeId] = med;

    const lCount = (count >> 1);
    const rCount = count - lCount - 1;
    if (lCount > 0) {
      left[nodeId] = nextNode;
      nodeIds.push(nextNode);
      stack.push({ first, count: lCount, depth: depth + 1 });
      nextNode++;
    }
    if (rCount > 0) {
      right[nodeId] = nextNode;
      nodeIds.push(nextNode);
      stack.push({ first: med + 1, count: rCount, depth: depth + 1 });
      nextNode++;
    }
  }

  return {
    count: n,
    order,
    axis: axis.subarray(0, nextNode),
    anchor: anchor.subarray(0, nextNode),
    left: left.subarray(0, nextNode),
    right: right.subarray(0, nextNode),
    points,
    buildMs: Date.now() - t0,
  };
}

function dist2(a: Float32Array, ai: number, q: Vec3): number {
  const dx = a[ai] - q[0];
  const dy = a[ai + 1] - q[1];
  const dz = a[ai + 2] - q[2];
  return dx * dx + dy * dy + dz * dz;
}

/** k-nearest neighbors with iterative bounded-priority traversal. */
export function knn(tree: KDTree, q: Vec3, k: number): { idx: number; d2: number }[] {
  if (tree.count === 0 || k <= 0) return [];
  const heap: { idx: number; d2: number }[] = [];
  const stack: number[] = [0];
  while (stack.length) {
    const node = stack.pop()!;
    if (node < 0 || tree.axis[node] < 0) continue;
    const ax = tree.axis[node];
    const aIdx = tree.order[tree.anchor[node]];
    const d2 = dist2(tree.points, aIdx * 3, q);

    if (heap.length < k) {
      heap.push({ idx: aIdx, d2 });
      heap.sort((a, b) => b.d2 - a.d2);
    } else if (d2 < heap[0].d2) {
      heap[0] = { idx: aIdx, d2 };
      heap.sort((a, b) => b.d2 - a.d2);
    }

    const split = tree.points[aIdx * 3 + ax];
    const diff = q[ax] - split;
    const near = diff < 0 ? tree.left[node] : tree.right[node];
    const far  = diff < 0 ? tree.right[node] : tree.left[node];
    if (near >= 0) stack.push(near);
    if (far >= 0 && (heap.length < k || diff * diff < heap[0].d2)) stack.push(far);
  }
  return heap.sort((a, b) => a.d2 - b.d2);
}

/** All points within radius r (squared distance ≤ r²). */
export function radiusQuery(tree: KDTree, q: Vec3, r: number): number[] {
  const r2 = r * r;
  const out: number[] = [];
  const stack: number[] = [0];
  while (stack.length) {
    const node = stack.pop()!;
    if (node < 0 || tree.axis[node] < 0) continue;
    const ax = tree.axis[node];
    const aIdx = tree.order[tree.anchor[node]];
    if (dist2(tree.points, aIdx * 3, q) <= r2) out.push(aIdx);
    const split = tree.points[aIdx * 3 + ax];
    const diff = q[ax] - split;
    if (tree.left[node] >= 0  && (diff < 0 || diff * diff <= r2)) stack.push(tree.left[node]);
    if (tree.right[node] >= 0 && (diff > 0 || diff * diff <= r2)) stack.push(tree.right[node]);
  }
  return out;
}
