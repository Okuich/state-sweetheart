import type { Vec3, AABB } from "../meshing/octree";

export type { Vec3, AABB };

/** A node in the topology graph (one per octree leaf). */
export interface TopoNode {
  /** Leaf index in the source octree mesh. */
  leaf: number;
  /** Centroid in world coords. */
  center: Vec3;
  /** Equivalent radius (half cell diagonal). */
  radius: number;
  /** Density score from refinement. */
  density: number;
  /** Refinement tag (hole, fillet, sharp, hotspot, overhang, thin_wall, contact, bulk). */
  tag: string;
  /** Detected feature class (see FeatureClass). */
  feature: FeatureClass;
  /** Wall-thickness estimate in world units (0 = bulk). */
  wallThickness: number;
  /** Discrete curvature proxy (face-deficit: 6 - faceNeighbors). */
  curvature: number;
  /** True if this node sits on the part boundary. */
  boundary: boolean;
  /** True if this node is downward-facing (overhang candidate). */
  downward: boolean;
}

/** A bidirectional edge between two adjacent leaves. */
export interface TopoEdge {
  a: number;       // node index
  b: number;       // node index
  shared: number;  // shared face area (world units²)
  axis: 0 | 1 | 2; // axis of the shared face normal
}

export type FeatureClass =
  | "bulk"
  | "boundary"
  | "thin_wall"
  | "overhang"
  | "cavity"
  | "stress_concentrator"
  | "thermal_bottleneck"
  | "symmetry_seed";

export const FEATURE_LABELS: Record<FeatureClass, string> = {
  bulk: "bulk",
  boundary: "boundary",
  thin_wall: "thin wall",
  overhang: "overhang",
  cavity: "cavity",
  stress_concentrator: "stress conc.",
  thermal_bottleneck: "thermal bottleneck",
  symmetry_seed: "symmetry seed",
};
