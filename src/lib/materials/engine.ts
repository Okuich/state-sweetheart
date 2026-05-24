/**
 * Material Metric Intelligence — recommendation engine.
 *
 *   recommend(corpus, constraints, weights) →
 *     1. embed corpus + cache scaler
 *     2. filter by hard constraints (yield, density, temp, fab, env)
 *     3. compute per-axis sub-scores (performance/cost/weight/sus/fatigue)
 *     4. fuse with weighted average → composite score
 *     5. estimate fatigue lifecycle (Basquin) + stress / thermal safety
 *
 *   similar(corpus, query, k) →
 *     cosine-similarity nearest neighbors in feature space
 *
 *   substitute(corpus, query, constraints) →
 *     similar() then filter / rank by constraints
 */

import { cosine, embed, fitScaler, l2, type FeatureScaler } from "./embeddings";
import type {
  DesignConstraints, MaterialRecord, MaterialScore, ObjectiveWeights,
} from "./types";
import { DEFAULT_WEIGHTS } from "./types";

export interface CorpusIndex {
  records: MaterialRecord[];
  embeddings: Float64Array[];
  scaler: FeatureScaler;
}

export function buildIndex(materials: MaterialRecord[]): CorpusIndex {
  const scaler = fitScaler(materials);
  const embeddings = materials.map((m) => embed(m, scaler));
  return { records: materials, embeddings, scaler };
}

// ---------------- scoring ----------------

function clamp01(x: number): number { return x < 0 ? 0 : x > 1 ? 1 : x; }

/**
 * Basquin S–N power law lifecycle estimate.
 *   N = (sigma_fatigue / sigma_design) ^ k   (k = 8 typical for metals)
 * Returns +Infinity when designStress is missing or below endurance limit.
 */
function basquinCycles(fatigueLimit: number, designStress: number, k = 8): number {
  if (designStress <= 0) return Infinity;
  if (designStress <= fatigueLimit) return Infinity;
  return Math.pow(fatigueLimit / designStress, k) * 1e7;
}

function evaluate(
  m: MaterialRecord,
  c: DesignConstraints,
): { violations: string[]; stressSafety: number; thermalSafety: number;
     lifecycleCycles: number; fabCompatibility: number; envOK: boolean } {
  const violations: string[] = [];

  if (c.minYield != null && m.yieldStrength < c.minYield) {
    violations.push(`yield ${m.yieldStrength} < ${c.minYield} MPa`);
  }
  if (c.minStiffness != null && m.youngsModulus < c.minStiffness) {
    violations.push(`stiffness ${m.youngsModulus} < ${c.minStiffness} GPa`);
  }
  if (c.maxDensity != null && m.density > c.maxDensity) {
    violations.push(`density ${m.density} > ${c.maxDensity} g/cm³`);
  }
  if (c.minServiceTempC != null && m.maxServiceTempC < c.minServiceTempC) {
    violations.push(`Tmax ${m.maxServiceTempC}°C < ${c.minServiceTempC}°C`);
  }
  if (c.maxCostPerKg != null && m.costPerKg > c.maxCostPerKg) {
    violations.push(`cost $${m.costPerKg} > $${c.maxCostPerKg}/kg`);
  }
  if (c.fabrication?.length) {
    const ok = c.fabrication.some((f) => m.fabrication.includes(f));
    if (!ok) violations.push(`no fab match (${c.fabrication.join(",")})`);
  }

  // Environment scoring (soft + hard).
  let envOK = true;
  if (c.environment?.length) {
    for (const e of c.environment) {
      if (e === "marine" && m.corrosionResistance < 0.7) {
        violations.push(`marine env requires corrosion≥0.7 (got ${m.corrosionResistance})`);
        envOK = false;
      }
      if (e === "uv_exposed" && m.weatherResistance < 0.7) {
        violations.push(`uv exposure requires weather≥0.7 (got ${m.weatherResistance})`);
        envOK = false;
      }
      if (e === "high_temp" && m.maxServiceTempC < 300) {
        violations.push(`high_temp env requires Tmax≥300°C (got ${m.maxServiceTempC})`);
        envOK = false;
      }
      if (e === "cryogenic" && m.thermalExpansion > 25) {
        violations.push(`cryogenic env requires low α (got ${m.thermalExpansion})`);
        envOK = false;
      }
    }
  }

  const stressSafety = c.designStress && c.designStress > 0
    ? m.yieldStrength / c.designStress : Infinity;
  const thermalSafety = c.minServiceTempC != null
    ? clamp01((m.maxServiceTempC - c.minServiceTempC) / Math.max(50, c.minServiceTempC * 0.5))
    : 1;
  const lifecycleCycles = basquinCycles(m.fatigueLimit, c.designStress ?? 0);

  const fabCompatibility = c.fabrication?.length
    ? c.fabrication.filter((f) => m.fabrication.includes(f)).length / c.fabrication.length
    : 1;

  return { violations, stressSafety, thermalSafety, lifecycleCycles, fabCompatibility, envOK };
}

function scoreMaterial(
  m: MaterialRecord, c: DesignConstraints, w: ObjectiveWeights,
  corpus: { maxCost: number; maxCO2: number; maxDensity: number; maxStiffPerRho: number },
  evald: ReturnType<typeof evaluate>,
): MaterialScore {
  // Performance: stress-safety (capped at 4) blended with stiffness-density.
  const ss = Number.isFinite(evald.stressSafety)
    ? clamp01(Math.min(4, evald.stressSafety) / 4)
    : 1; // no design stress provided → assume safe
  const sd = clamp01((m.youngsModulus / Math.max(0.01, m.density)) / corpus.maxStiffPerRho);
  const performance = 0.65 * ss + 0.35 * sd;

  // Cost: inverse-scaled.
  const cost = clamp01(1 - m.costPerKg / Math.max(1, corpus.maxCost));

  // Weight: inverse density.
  const weight = clamp01(1 - m.density / Math.max(0.5, corpus.maxDensity));

  // Sustainability: inverse embodied CO2.
  const sustainability = clamp01(1 - m.embodiedCO2 / Math.max(0.5, corpus.maxCO2));

  // Fatigue: ratio of fatigue limit to design stress (or to yield/3 as proxy).
  const refStress = c.designStress && c.designStress > 0 ? c.designStress : m.yieldStrength / 3;
  const fatigue = clamp01(m.fatigueLimit / Math.max(1, refStress * 2));

  // Normalize weights and fuse.
  const total = w.performance + w.cost + w.weight + w.sustainability + w.fatigue || 1;
  const composite = (
    w.performance   * performance   +
    w.cost          * cost          +
    w.weight        * weight        +
    w.sustainability * sustainability +
    w.fatigue       * fatigue
  ) / total;

  return {
    material: m,
    score: composite,
    feasible: evald.violations.length === 0,
    breakdown: { performance, cost, weight, sustainability, fatigue },
    stressSafety: evald.stressSafety,
    thermalSafety: evald.thermalSafety,
    lifecycleCycles: evald.lifecycleCycles,
    violations: evald.violations,
    fabCompatibility: evald.fabCompatibility,
  };
}

// ---------------- public API ----------------

/**
 * Rank a corpus against design constraints + objective weights.
 * Returns all materials, sorted: feasible first, then by composite score.
 */
export function recommend(
  index: CorpusIndex,
  constraints: DesignConstraints = {},
  weights: ObjectiveWeights = DEFAULT_WEIGHTS,
): MaterialScore[] {
  const corpus = {
    maxCost: Math.max(...index.records.map((r) => r.costPerKg)),
    maxCO2: Math.max(...index.records.map((r) => r.embodiedCO2)),
    maxDensity: Math.max(...index.records.map((r) => r.density)),
    maxStiffPerRho: Math.max(...index.records.map((r) => r.youngsModulus / Math.max(0.01, r.density))),
  };
  const out: MaterialScore[] = [];
  for (const m of index.records) {
    const ev = evaluate(m, constraints);
    out.push(scoreMaterial(m, constraints, weights, corpus, ev));
  }
  out.sort((a, b) => Number(b.feasible) - Number(a.feasible) || b.score - a.score);
  return out;
}

/** Nearest neighbors of a query material by cosine similarity. */
export function similar(
  index: CorpusIndex, queryId: string, k = 5,
): Array<{ material: MaterialRecord; similarity: number; dist: number }> {
  const idx = index.records.findIndex((r) => r.id === queryId);
  if (idx < 0) return [];
  const q = index.embeddings[idx];
  const out: Array<{ material: MaterialRecord; similarity: number; dist: number }> = [];
  for (let i = 0; i < index.records.length; i++) {
    if (i === idx) continue;
    out.push({
      material: index.records[i],
      similarity: cosine(q, index.embeddings[i]),
      dist: l2(q, index.embeddings[i]),
    });
  }
  out.sort((a, b) => b.similarity - a.similarity);
  return out.slice(0, k);
}

/**
 * Substitute recommendations for a given query material under
 * design constraints — combines metric similarity with feasibility
 * scoring so the alternates are both close *and* valid.
 */
export function substitute(
  index: CorpusIndex, queryId: string,
  constraints: DesignConstraints = {},
  weights: ObjectiveWeights = DEFAULT_WEIGHTS,
  k = 5,
): MaterialScore[] {
  const sims = similar(index, queryId, index.records.length - 1);
  const ranked = recommend(index, constraints, weights);
  const byId = new Map(ranked.map((s) => [s.material.id, s]));
  const out: MaterialScore[] = [];
  for (const { material, similarity } of sims) {
    const s = byId.get(material.id);
    if (!s) continue;
    out.push({ ...s, similarity });
  }
  out.sort((a, b) => {
    // Feasible substitutes first; then 70% similarity / 30% composite score.
    if (a.feasible !== b.feasible) return Number(b.feasible) - Number(a.feasible);
    const ka = 0.7 * (a.similarity ?? 0) + 0.3 * a.score;
    const kb = 0.7 * (b.similarity ?? 0) + 0.3 * b.score;
    return kb - ka;
  });
  return out.slice(0, k);
}
