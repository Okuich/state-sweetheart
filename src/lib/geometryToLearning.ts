/**
 * geometryToLearning
 *
 * Mapper + tiny pub/sub bridge that turns a `FeatureIntelligence` (the
 * geometry panel's output) into the 12-d normalized feature vector the
 * learning engine consumes (`FEATURE_NAMES`). This lets predictions update
 * from real CAD designs instead of slider-only candidates.
 *
 * The mapping is heuristic — geometry panel and learning engine were
 * developed against different feature schemas, so we project semantically
 * matching signals into [0..1] using bbox dims, curvature mix, and risk
 * heuristics. Each output dimension has a documented rationale in
 * `MAPPING_NOTES` so the UI can show "why this value".
 */
import type { GeomDescriptor } from "@/lib/stepParser";
import type { FeatureIntelligence } from "@/lib/geometryFeatures";
import { FEATURE_NAMES, D } from "@/lib/learningEngine";

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

export interface MappedFeatures {
  vector: number[];                       // length D (12)
  rationale: Record<string, string>;      // per-feature explanation
  source: { partName?: string; schema?: string; entities: number };
}

export const MAPPING_NOTES: Record<(typeof FEATURE_NAMES)[number], string> = {
  thickness:       "min(bbox dim) / max(bbox dim) — slab vs. blocky",
  aspectRatio:     "max(bbox dim) / min(bbox dim), /10",
  minRadius:       "1 − stress-concentration risk (sharp-corner proxy)",
  holeDensity:     "fab feature vector[0] (holes / 10)",
  filletCount:     "toroidal-surface fraction (tori ≈ fillets)",
  volume:          "bbox volume / 1000",
  surfaceArea:     "bbox surface area / 1300",
  simEnergy:       "thermal-risk heuristic (proxy for strain-energy density)",
  simMaxStress:    "stress-concentration heuristic",
  fabSpindleLoad:  "fabrication-difficulty heuristic",
  fabVibration:    "spline-surface fraction (freeform → vibration proxy)",
  qaSurfaceDefect: "1 − planar fraction (less planar → more defects)",
};

export function mapGeometryToLearningVector(
  intel: FeatureIntelligence,
  desc: GeomDescriptor,
): MappedFeatures {
  const dims = desc.bbox
    ? [
        desc.bbox.max[0] - desc.bbox.min[0],
        desc.bbox.max[1] - desc.bbox.min[1],
        desc.bbox.max[2] - desc.bbox.min[2],
      ]
    : [1, 1, 1];
  const dmin = Math.max(1e-6, Math.min(...dims));
  const dmax = Math.max(1e-6, Math.max(...dims));
  const vol = dims[0] * dims[1] * dims[2];
  const area = 2 * (dims[0] * dims[1] + dims[1] * dims[2] + dims[0] * dims[2]);

  const v = new Array(D).fill(0);
  v[0] = clamp01(dmin / dmax);                                     // thickness
  v[1] = clamp01(dmax / dmin / 10);                                // aspectRatio
  v[2] = clamp01(1 - intel.risk.stressConcentration);              // minRadius
  v[3] = clamp01(intel.fabFeatureVector[0] ?? 0);                  // holeDensity
  v[4] = clamp01(intel.curvature.toroidal);                        // filletCount
  v[5] = clamp01(vol / 1000);                                      // volume
  v[6] = clamp01(area / 1300);                                     // surfaceArea
  v[7] = clamp01(intel.risk.thermalRisk);                          // simEnergy
  v[8] = clamp01(intel.risk.stressConcentration);                  // simMaxStress
  v[9] = clamp01(intel.risk.fabricationDifficulty);                // fabSpindleLoad
  v[10] = clamp01(intel.curvature.spline);                         // fabVibration
  v[11] = clamp01(1 - intel.curvature.planar);                     // qaSurfaceDefect

  return {
    vector: v,
    rationale: { ...MAPPING_NOTES },
    source: { entities: 0, schema: undefined },
  };
}

// ── Pub/sub bridge ───────────────────────────────────────────────────────
//
// GeometryFeaturePanel publishes mapped features whenever its analysis
// completes; LearningEnginePanel subscribes and (optionally) overrides its
// slider candidate with the mapped vector. Decoupled so neither file has to
// import the other.

export interface PublishedDesign extends MappedFeatures {
  ts: number;
  variantName: string;
}

type Listener = (d: PublishedDesign | null) => void;

let current: PublishedDesign | null = null;
const listeners = new Set<Listener>();

export const designBridge = {
  publish(d: PublishedDesign): void {
    current = d;
    for (const l of listeners) try { l(d); } catch { /* ignore */ }
  },
  clear(): void {
    current = null;
    for (const l of listeners) try { l(null); } catch { /* ignore */ }
  },
  current(): PublishedDesign | null { return current; },
  subscribe(l: Listener): () => void {
    listeners.add(l);
    return () => listeners.delete(l);
  },
};
