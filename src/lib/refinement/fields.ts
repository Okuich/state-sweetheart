/**
 * Physics-feedback field synthesis.
 *
 * Mocks the per-leaf scalar fields a Physics OS solver would emit each step:
 *   stress magnitude, thermal gradient, contact pressure, deformation,
 *   curvature error, constraint residual.
 *
 * In production these arrive from the FEM kernel as Float32Array per leaf;
 * here we deterministically synthesize them from the octree + seeds so the
 * engine can be driven end-to-end inside the sandbox.
 */

import type { OctreeMesh, RefinementSeed, Vec3 } from "../meshing/octree";

export type FieldKind =
  | "stress"
  | "thermal"
  | "contact"
  | "deformation"
  | "curvature"
  | "residual";

export interface PhysicsFields {
  /** Per-leaf scalar in [0, +inf), normalized later. */
  stress: Float32Array;
  thermal: Float32Array;
  contact: Float32Array;
  deformation: Float32Array;
  curvature: Float32Array;
  residual: Float32Array;
}

function leafCenter(mesh: OctreeMesh, leafIdx: number): Vec3 {
  const id = mesh.leaves[leafIdx];
  const b = mesh.nodes[id].bbox;
  return [
    (b.min[0] + b.max[0]) / 2,
    (b.min[1] + b.max[1]) / 2,
    (b.min[2] + b.max[2]) / 2,
  ];
}

function seedFalloff(p: Vec3, s: RefinementSeed): number {
  const dx = p[0] - s.center[0], dy = p[1] - s.center[1], dz = p[2] - s.center[2];
  const r2 = dx * dx + dy * dy + dz * dz;
  return Math.exp(-r2 / Math.max(1e-9, s.radius * s.radius));
}

/**
 * Synthesize plausible physics fields per leaf, keyed off seeds.
 * Each seed kind preferentially excites a subset of fields:
 *   sharp/hole  → stress + curvature
 *   hotspot     → thermal + residual
 *   contact     → contact + stress
 *   overhang    → deformation + residual
 *   thin_wall   → stress + deformation
 *   fillet      → curvature
 */
export function synthesizeFields(
  mesh: OctreeMesh,
  seeds: RefinementSeed[],
  step = 0,
): PhysicsFields {
  const N = mesh.leaves.length;
  const f: PhysicsFields = {
    stress: new Float32Array(N),
    thermal: new Float32Array(N),
    contact: new Float32Array(N),
    deformation: new Float32Array(N),
    curvature: new Float32Array(N),
    residual: new Float32Array(N),
  };
  const t = step * 0.05;
  for (let i = 0; i < N; i++) {
    const p = leafCenter(mesh, i);
    for (const s of seeds) {
      const g = seedFalloff(p, s) * s.weight;
      switch (s.kind) {
        case "sharp": f.stress[i] += g * 1.4; f.curvature[i] += g * 1.1; break;
        case "hole": f.stress[i] += g * 1.2; f.curvature[i] += g * 0.9; break;
        case "hotspot": f.thermal[i] += g * 1.5 * (1 + 0.2 * Math.sin(t)); f.residual[i] += g * 0.4; break;
        case "contact": f.contact[i] += g * 1.6; f.stress[i] += g * 0.6; break;
        case "overhang": f.deformation[i] += g * 1.3; f.residual[i] += g * 0.5; break;
        case "thin_wall": f.stress[i] += g * 0.9; f.deformation[i] += g * 1.0; break;
        case "fillet": f.curvature[i] += g * 0.7; break;
      }
    }
    // Mild bulk thermal drift so the field is non-trivial even with no seeds.
    f.thermal[i] += 0.05 * (1 + Math.cos(t + p[0] + p[1]));
  }
  return f;
}

/** L_inf normalize a field to [0,1]; returns peak too. */
export function normalize(arr: Float32Array): { norm: Float32Array; peak: number } {
  let peak = 0;
  for (let i = 0; i < arr.length; i++) if (arr[i] > peak) peak = arr[i];
  const norm = new Float32Array(arr.length);
  if (peak > 0) for (let i = 0; i < arr.length; i++) norm[i] = arr[i] / peak;
  return { norm, peak };
}
