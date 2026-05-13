/**
 * Manufacturability scoring.
 *
 * Aggregates the FeatureReport into actionable scores for fabrication, plus
 * physics priors used by Physics OS for adaptive meshing and deformation
 * heuristics.
 *
 * All scores are in [0,1]. Higher = better/easier (except risk fields).
 */

import type { TopologyGraph } from "./graph";
import type { FeatureReport } from "./features";
import type { AABB } from "./types";

export interface ManufacturabilityScores {
  /** Aggregate fabrication feasibility in [0,1]. */
  feasibility: number;
  /** Fraction of boundary that needs print support. */
  supportFraction: number;
  /** Tool accessibility for subtractive ops in [0,1]. */
  machiningAccess: number;
  /** Risk of thermal warp in [0,1]. */
  thermalDistortionRisk: number;
  /** Assembly complexity in [0,1] (higher = harder). */
  assemblyComplexity: number;
  /** Per-driver breakdown for the UI. */
  drivers: {
    overhangPenalty: number;
    cavityPenalty: number;
    thinWallPenalty: number;
    stressPenalty: number;
    thermalPenalty: number;
  };
}

export function scoreManufacturability(
  graph: TopologyGraph,
  feats: FeatureReport,
  bbox: AABB,
): ManufacturabilityScores {
  const N = Math.max(1, graph.nodes.length);
  const boundaryCount = Math.max(1, feats.counts.boundary + feats.counts.thin_wall + feats.counts.overhang + feats.counts.cavity + feats.counts.stress_concentrator);

  const overhangFrac = feats.counts.overhang / boundaryCount;
  const cavityFrac = feats.counts.cavity / boundaryCount;
  const thinFrac = feats.counts.thin_wall / boundaryCount;
  const stressFrac = feats.counts.stress_concentrator / boundaryCount;
  const thermalFrac = feats.counts.thermal_bottleneck / N;

  const overhangPenalty = clamp01(overhangFrac * 1.5);
  const cavityPenalty = clamp01(cavityFrac * 2.0);
  const thinWallPenalty = clamp01(thinFrac * 1.2);
  const stressPenalty = clamp01(stressFrac * 0.8);
  const thermalPenalty = clamp01(thermalFrac * 1.4);

  const feasibility = clamp01(
    1 - 0.30 * overhangPenalty
      - 0.20 * cavityPenalty
      - 0.20 * thinWallPenalty
      - 0.15 * stressPenalty
      - 0.15 * thermalPenalty,
  );

  // Support fraction: every overhang plus half of severe thin walls.
  const supportFraction = clamp01(overhangFrac + 0.4 * thinFrac);

  // Machining access: 1 minus a function of cavity fraction and aspect ratio.
  const ext = [bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2]];
  const aspect = Math.max(...ext) / Math.max(1e-6, Math.min(...ext));
  const aspectPenalty = clamp01((aspect - 2) / 8);
  const machiningAccess = clamp01(1 - cavityPenalty * 0.7 - aspectPenalty * 0.3);

  // Thermal distortion: dominated by thin walls + thermal bottlenecks + low symmetry.
  const symBoost = (1 - feats.symmetryScore) * 0.4;
  const thermalDistortionRisk = clamp01(thermalPenalty * 0.5 + thinWallPenalty * 0.4 + symBoost);

  // Assembly complexity: scaled by part count proxy (boundary surface area).
  const surfaceFrac = boundaryCount / N;
  const assemblyComplexity = clamp01(surfaceFrac * 1.3 + cavityPenalty * 0.4);

  return {
    feasibility,
    supportFraction,
    machiningAccess,
    thermalDistortionRisk,
    assemblyComplexity,
    drivers: { overhangPenalty, cavityPenalty, thinWallPenalty, stressPenalty, thermalPenalty },
  };
}

function clamp01(x: number) { return Math.max(0, Math.min(1, x)); }
