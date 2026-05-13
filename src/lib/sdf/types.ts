/** Shared SDF types. */
export type Vec3 = [number, number, number];
export interface AABB { min: Vec3; max: Vec3 }

/** A single signed-distance "primitive" — analytic source used to bake the field. */
export type SDFPrim =
  | { kind: "sphere"; center: Vec3; radius: number }
  | { kind: "box"; center: Vec3; half: Vec3 }
  | { kind: "cylinder"; center: Vec3; axis: Vec3; radius: number; height: number }
  | { kind: "plane"; point: Vec3; normal: Vec3 }
  | { kind: "torus"; center: Vec3; major: number; minor: number };

/** Optional adaptive refinement hints (mirrors meshing seed kinds). */
export type AdaptiveHintKind =
  | "stress" | "thermal" | "contact" | "thin_wall" | "high_curvature";

export interface AdaptiveHint {
  kind: AdaptiveHintKind;
  center: Vec3;
  radius: number;   // world-space radius of influence
  weight: number;   // 0..1 — boosts local resolution
}
