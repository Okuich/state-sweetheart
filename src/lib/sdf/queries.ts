/**
 * GPU-shaped query layer over a SparseSDF.
 *
 * All queries are pure-TS but written so the same memory layout can be
 * uploaded to a WebGPU storage buffer (BRICK^3 contiguous floats per brick).
 */

import { sampleSDF, type SparseSDF } from "./sparseField";
import type { Vec3 } from "./types";

/** Distance to the surface (signed). */
export function distance(sdf: SparseSDF, p: Vec3): number {
  return sampleSDF(sdf, p);
}

/** Central-difference gradient → outward surface normal direction. */
export function gradient(sdf: SparseSDF, p: Vec3, h?: number): Vec3 {
  const eps = h ?? sdf.options.voxelSize * 0.5;
  const dx = sampleSDF(sdf, [p[0] + eps, p[1], p[2]]) - sampleSDF(sdf, [p[0] - eps, p[1], p[2]]);
  const dy = sampleSDF(sdf, [p[0], p[1] + eps, p[2]]) - sampleSDF(sdf, [p[0], p[1] - eps, p[2]]);
  const dz = sampleSDF(sdf, [p[0], p[1], p[2] + eps]) - sampleSDF(sdf, [p[0], p[1], p[2] - eps]);
  const inv = 1 / (2 * eps);
  return [dx * inv, dy * inv, dz * inv];
}

/** Project p to the closest surface point via Newton iteration on the SDF. */
export function nearestSurface(sdf: SparseSDF, p: Vec3, iters = 6): { point: Vec3; distance: number; normal: Vec3 } {
  let q: Vec3 = [p[0], p[1], p[2]];
  let d = sampleSDF(sdf, q);
  let n: Vec3 = [0, 0, 1];
  for (let i = 0; i < iters; i++) {
    n = gradient(sdf, q);
    const ln = Math.hypot(n[0], n[1], n[2]) || 1;
    n = [n[0] / ln, n[1] / ln, n[2] / ln];
    q = [q[0] - d * n[0], q[1] - d * n[1], q[2] - d * n[2]];
    d = sampleSDF(sdf, q);
    if (Math.abs(d) < sdf.options.voxelSize * 0.05) break;
  }
  return { point: q, distance: d, normal: n };
}

/** Penetration depth (positive when p is inside the solid). */
export function penetration(sdf: SparseSDF, p: Vec3): number {
  return -sampleSDF(sdf, p);
}

/**
 * Sphere-vs-SDF collision. Returns penetration > 0 only on contact.
 * Output normal points from surface toward `center` (separation direction).
 */
export function sphereCollide(
  sdf: SparseSDF,
  center: Vec3,
  radius: number,
): { hit: boolean; depth: number; normal: Vec3; contact: Vec3 } {
  const d = sampleSDF(sdf, center);
  const depth = radius - d;
  if (depth <= 0) {
    return { hit: false, depth: 0, normal: [0, 0, 1], contact: center };
  }
  let n = gradient(sdf, center);
  const ln = Math.hypot(n[0], n[1], n[2]) || 1;
  n = [n[0] / ln, n[1] / ln, n[2] / ln];
  const contact: Vec3 = [center[0] - n[0] * d, center[1] - n[1] * d, center[2] - n[2] * d];
  return { hit: true, depth, normal: n, contact };
}
