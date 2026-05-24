/**
 * Material Metric Intelligence — shared types.
 *
 * Materials are represented as fixed-order feature vectors covering
 * tensile, thermal, elasticity, conductivity, fatigue, cost, and
 * environmental-resilience axes. The engine reasons in this metric
 * space to recommend, substitute, rank, and score materials against
 * a design's constraints.
 */

export type MaterialFamily =
  | "metal"
  | "polymer"
  | "ceramic"
  | "composite"
  | "elastomer"
  | "other";

export type Fabrication =
  | "machining" | "casting" | "injection_molding" | "additive_dmls"
  | "additive_fdm" | "sheet_forming" | "extrusion" | "layup" | "sintering";

export type Environment =
  | "ambient" | "marine" | "cryogenic" | "high_temp" | "uv_exposed"
  | "chemical" | "abrasive";

export interface MaterialRecord {
  id: string;
  name: string;
  family: MaterialFamily;
  /** g/cm^3 */
  density: number;
  /** GPa */
  youngsModulus: number;
  /** MPa */
  yieldStrength: number;
  /** MPa */
  ultimateStrength: number;
  /** MPa — endurance limit (~10^7 cycles). */
  fatigueLimit: number;
  /** W/(m·K) */
  thermalConductivity: number;
  /** 1e-6 / K */
  thermalExpansion: number;
  /** °C — max safe service temp. */
  maxServiceTempC: number;
  /** USD / kg */
  costPerKg: number;
  /** kg CO2e / kg — embodied carbon. */
  embodiedCO2: number;
  /** 0..1 — corrosion resistance (1 = inert). */
  corrosionResistance: number;
  /** 0..1 — UV / weathering resistance. */
  weatherResistance: number;
  /** Compatible fabrication methods. */
  fabrication: Fabrication[];
}

/** Design constraints — any field may be omitted. */
export interface DesignConstraints {
  /** Minimum yield strength required (MPa). */
  minYield?: number;
  /** Minimum stiffness (GPa). */
  minStiffness?: number;
  /** Maximum density (g/cm^3). */
  maxDensity?: number;
  /** Service temperature requirement (°C). */
  minServiceTempC?: number;
  /** Maximum allowable cost ($/kg). */
  maxCostPerKg?: number;
  /** Required environment(s). */
  environment?: Environment[];
  /** Required fabrication method(s). */
  fabrication?: Fabrication[];
  /** Cyclic loading (number of design cycles); used for fatigue check. */
  designCycles?: number;
  /** Design stress (MPa); used to estimate safety + lifecycle. */
  designStress?: number;
}

/** Multi-objective weights — engine normalizes them internally. */
export interface ObjectiveWeights {
  performance: number; // stress safety + stiffness/density
  cost: number;        // lower $/kg better
  weight: number;      // lower density better
  sustainability: number; // lower CO2 better
  fatigue: number;     // higher fatigue:design-stress ratio better
}

export const DEFAULT_WEIGHTS: ObjectiveWeights = {
  performance: 0.35,
  cost: 0.20,
  weight: 0.15,
  sustainability: 0.10,
  fatigue: 0.20,
};

export interface MaterialScore {
  material: MaterialRecord;
  /** Composite 0..1 score (higher = better). */
  score: number;
  /** Pass/fail summary against hard constraints. */
  feasible: boolean;
  /** Per-axis sub-scores in [0,1]. */
  breakdown: {
    performance: number;
    cost: number;
    weight: number;
    sustainability: number;
    fatigue: number;
  };
  /** Stress safety factor (yield / designStress). */
  stressSafety: number;
  /** Thermal safety (maxServiceTemp − requiredTemp), normalized. */
  thermalSafety: number;
  /** Estimated lifecycle (cycles) under designStress via Basquin power law. */
  lifecycleCycles: number;
  /** Cosine similarity to query material if substituting. */
  similarity?: number;
  /** Constraint failure reasons (empty if feasible). */
  violations: string[];
  /** Fabrication compatibility 0..1. */
  fabCompatibility: number;
}
