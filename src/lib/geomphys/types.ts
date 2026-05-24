/**
 * Geometry-Aware Physics Intelligence — types.
 *
 * Operates on a lightweight triangle-mesh representation that lives
 * adjacent to the simulator. The engine unifies computational geometry
 * (curvature, distortion, topology genus) with physics state (stress,
 * strain energy, displacement) to:
 *
 *   - measure deformation between a rest mesh and a deformed mesh
 *   - predict per-vertex stress from local curvature + topology
 *   - route a solve to the cheapest viable solver based on shape signal
 *   - optimize shape for stored elastic energy while preserving topology
 *
 * All geometry math is closed-form, runs on CPU, and exposes results
 * the UI can chart frame-to-frame.
 */

export interface TriMesh {
  /** Float64Array of length 3 × V — flat (x, y, z) per vertex. */
  positions: Float64Array;
  /** Uint32Array of length 3 × T — triangle vertex indices. */
  indices: Uint32Array;
  /** Optional human-readable label. */
  label?: string;
}

export type SolverKind = "explicit" | "implicit_cg" | "implicit_direct" | "rom_reduced";

export interface DeformationMetrics {
  /** L2 displacement magnitude per vertex (length V). */
  perVertexDisp: Float64Array;
  /** Max displacement magnitude across all vertices. */
  maxDisp: number;
  /** Mean displacement magnitude. */
  meanDisp: number;
  /** Mean edge stretch ratio (|e_def| / |e_rest|). */
  meanStretch: number;
  /** Max edge stretch ratio. */
  maxStretch: number;
  /** Mean per-triangle area-ratio (def / rest). */
  meanAreaRatio: number;
  /** Mean dihedral-angle change in radians. */
  meanBendingRad: number;
  /** Surface distortion score (0 = isometric, ↑ = worse). */
  distortion: number;
}

export interface CurvatureField {
  /** Discrete mean curvature per vertex (Laplace-Beltrami magnitude). */
  meanCurvature: Float64Array;
  /** Angle-defect-based Gaussian curvature per vertex. */
  gaussianCurvature: Float64Array;
  /** Per-vertex Voronoi area. */
  vertexArea: Float64Array;
}

export interface StressPrediction {
  /** Predicted stress per vertex (relative units). */
  vertexStress: Float64Array;
  /** Index of the vertex with highest stress (hotspot). */
  hotspotVertex: number;
  /** Hotspot stress value. */
  hotspotValue: number;
  /** Stress concentration factor: max / mean. */
  concentrationFactor: number;
  /** Predicted strain energy (sum over triangles). */
  totalStrainEnergy: number;
}

export interface TopologyProfile {
  V: number; E: number; F: number;
  /** Connected component count (≥1). */
  components: number;
  /** Euler characteristic V - E + F. */
  euler: number;
  /** Genus (for closed orientable surfaces): (2c - χ) / 2. */
  genus: number;
  /** Average vertex valence. */
  avgValence: number;
  /** Max vertex valence (indicates star defects). */
  maxValence: number;
  /** Min / max / mean triangle quality (radius ratio, 0..1). */
  triQualityMin: number;
  triQualityMean: number;
}

export interface SolverDecision {
  solver: SolverKind;
  reasoning: string[];
  /** Estimated relative cost vs implicit_direct baseline (0..1). */
  costFactor: number;
  /** Estimated accuracy factor (0..1). */
  accuracyFactor: number;
  /** Composite confidence in this routing (0..1). */
  confidence: number;
}

export interface ShapeOptimizationResult {
  before: { energy: number; distortion: number };
  after:  { energy: number; distortion: number };
  iterations: number;
  /** Energy reduction in [0,1] (1 = total). */
  energyReduction: number;
  /** True when topology unchanged through optimization. */
  topologyPreserved: boolean;
  /** New mesh positions (same indices). */
  mesh: TriMesh;
}

export interface GeometryReport {
  topology: TopologyProfile;
  curvature: CurvatureField;
  deformation: DeformationMetrics | null;
  stress: StressPrediction | null;
  routing: SolverDecision;
}
