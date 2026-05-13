/**
 * Higher-level geometry queries built on the BVH:
 *   - ray/AABB intersection (slab method)
 *   - point-in-AABB / point-in-solid (parity test against axis ray)
 *   - triangle/AABB overlap test (separating-axis theorem)
 *   - contact candidate generation between two BVHs.
 */

import type { BVH, Vec3 } from "./bvh";
import { bvhQueryAABB } from "./bvh";

export interface RayHit { primId: number; tEnter: number; tExit: number }

export function rayAABB(origin: Vec3, dir: Vec3, lo: Vec3, hi: Vec3): { tEnter: number; tExit: number } | null {
  let tNear = -Infinity, tFar = Infinity;
  for (let i = 0; i < 3; i++) {
    const o = origin[i], d = dir[i], l = lo[i], h = hi[i];
    if (Math.abs(d) < 1e-12) {
      if (o < l || o > h) return null;
      continue;
    }
    const inv = 1 / d;
    let t1 = (l - o) * inv, t2 = (h - o) * inv;
    if (t1 > t2) { const t = t1; t1 = t2; t2 = t; }
    if (t1 > tNear) tNear = t1;
    if (t2 < tFar) tFar = t2;
    if (tNear > tFar || tFar < 0) return null;
  }
  return { tEnter: tNear, tExit: tFar };
}

/** All BVH primitives whose AABB is hit by the ray, with entry t. */
export function rayQuery(bvh: BVH, origin: Vec3, dir: Vec3): RayHit[] {
  const hits: RayHit[] = [];
  const stack: number[] = [0];
  while (stack.length) {
    const n = stack.pop()!;
    const i3 = n * 3;
    const lo: Vec3 = [bvh.nodeMin[i3], bvh.nodeMin[i3 + 1], bvh.nodeMin[i3 + 2]];
    const hi: Vec3 = [bvh.nodeMax[i3], bvh.nodeMax[i3 + 1], bvh.nodeMax[i3 + 2]];
    if (!rayAABB(origin, dir, lo, hi)) continue;
    if (bvh.left[n] === -1) {
      const f = bvh.primFirst[n], c = bvh.primCount_[n];
      for (let k = 0; k < c; k++) {
        const p = bvh.primOrder[f + k], p3 = p * 3;
        const r = rayAABB(
          origin, dir,
          [bvh.primMin[p3], bvh.primMin[p3 + 1], bvh.primMin[p3 + 2]],
          [bvh.primMax[p3], bvh.primMax[p3 + 1], bvh.primMax[p3 + 2]],
        );
        if (r) hits.push({ primId: p, tEnter: r.tEnter, tExit: r.tExit });
      }
    } else {
      stack.push(bvh.left[n], bvh.right[n]);
    }
  }
  return hits.sort((a, b) => a.tEnter - b.tEnter);
}

/** Approximate point-in-solid via ray-parity vs the BVH (uses prim AABBs). */
export function pointInSolid(bvh: BVH, p: Vec3): boolean {
  const hits = rayQuery(bvh, p, [1, 0.0001, 0.0001]);
  let parity = 0;
  for (const h of hits) {
    if (h.tEnter <= 0 && h.tExit > 0) parity++; // origin inside this box
    else if (h.tEnter > 0) parity += 2;          // box fully ahead: enter + exit
  }
  return (parity & 1) === 1;
}

/** Contact candidate pairs between two BVHs (broadphase against scene B). */
export function contactCandidates(a: BVH, b: BVH, max = 100_000): Uint32Array {
  const out: number[] = [];
  for (let i = 0; i < a.primCount && out.length / 2 < max; i++) {
    const i3 = i * 3;
    const hits: number[] = [];
    bvhQueryAABB(
      b,
      [a.primMin[i3], a.primMin[i3 + 1], a.primMin[i3 + 2]],
      [a.primMax[i3], a.primMax[i3 + 1], a.primMax[i3 + 2]],
      hits,
    );
    for (const j of hits) {
      out.push(i, j);
      if (out.length / 2 >= max) break;
    }
  }
  return new Uint32Array(out);
}
