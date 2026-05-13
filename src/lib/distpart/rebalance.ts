/**
 * Dynamic repartitioning + migration planning.
 *
 * Given a current assignment + per-tet workload weights (e.g. step time from
 * Physics OS telemetry), propose:
 *   - migrate list: (tet → newPart) moves that reduce makespan
 *   - estimated comm cost of migration vs compute saved
 *   - new size + imbalance projection
 *
 * Strategy: diffusion-based — each over-loaded partition donates boundary
 * tets to its lowest-loaded neighbor in the comm graph until imbalance is
 * within tolerance or no further beneficial move exists.
 */

import type { AdjacencyTensors } from "../meshing/adjacency";

export interface RebalanceInput {
  tetPart: Uint16Array;
  weights: Float32Array;
  partitionCount: number;
  adj: AdjacencyTensors;
  /** Imbalance target (max/mean). */
  tolerance?: number;
  /** Cap on tet migrations (rate-limit). */
  maxMigrations?: number;
}

export interface MigrationMove {
  tet: number;
  from: number;
  to: number;
  weight: number;
}

export interface RebalancePlan {
  moves: MigrationMove[];
  beforeImbalance: number;
  afterImbalance: number;
  beforeMakespan: number;
  afterMakespan: number;
  /** Estimated bytes to migrate (state copy). */
  migrationBytes: number;
  /** Per-partition load before/after. */
  beforeLoad: Float32Array;
  afterLoad: Float32Array;
}

export function planRebalance(input: RebalanceInput, payloadBytes = 256): RebalancePlan {
  const { tetPart, weights, partitionCount: P, adj } = input;
  const tol = input.tolerance ?? 1.1;
  const cap = input.maxMigrations ?? 5000;
  const T = tetPart.length;
  const load = new Float32Array(P);
  for (let t = 0; t < T; t++) load[tetPart[t]] += weights[t] || 1;
  const beforeLoad = new Float32Array(load);

  const total = load.reduce((a, b) => a + b, 0);
  const target = total / P;
  const beforeMakespan = Math.max(...load);
  const beforeImb = beforeMakespan / Math.max(target, 1e-9);

  // For each over-loaded partition, scan boundary tets and donate to the
  // most-receiving neighbor partition (one with deepest deficit AND highest
  // adjacency to the donor).
  const moves: MigrationMove[] = [];
  const part = new Uint16Array(tetPart); // mutable copy

  // Pre-bucket boundary tets per partition.
  const boundaryByPart: number[][] = Array.from({ length: P }, () => []);
  for (let t = 0; t < T; t++) {
    const pt = part[t];
    const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
    for (let k = s; k < e; k++) {
      if (part[adj.ttCol[k]] !== pt) { boundaryByPart[pt].push(t); break; }
    }
  }

  let iter = 0;
  while (iter++ < 64 && moves.length < cap) {
    let donor = 0;
    for (let p = 1; p < P; p++) if (load[p] > load[donor]) donor = p;
    if (load[donor] <= target * tol) break;

    const tList = boundaryByPart[donor];
    if (!tList.length) break;
    let progressed = false;

    for (let i = 0; i < tList.length && moves.length < cap; i++) {
      const t = tList[i];
      if (part[t] !== donor) continue;
      // Pick destination neighbor partition with most boundary contact AND deficit.
      const counts = new Map<number, number>();
      const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
      for (let k = s; k < e; k++) {
        const np = part[adj.ttCol[k]];
        if (np !== donor) counts.set(np, (counts.get(np) ?? 0) + 1);
      }
      let bestDst = -1, bestScore = -Infinity;
      for (const [p, c] of counts) {
        if (load[p] >= target * tol) continue;
        const score = c - (load[p] - target * 0.85) * 0.5;
        if (score > bestScore) { bestScore = score; bestDst = p; }
      }
      if (bestDst < 0) continue;
      const w = weights[t] || 1;
      if (load[donor] - w < target * 0.85) continue;
      moves.push({ tet: t, from: donor, to: bestDst, weight: w });
      load[donor] -= w;
      load[bestDst] += w;
      part[t] = bestDst;
      boundaryByPart[bestDst].push(t);
      progressed = true;
      if (load[donor] <= target * tol) break;
    }
    if (!progressed) break;
  }

  const afterMakespan = Math.max(...load);
  const afterImb = afterMakespan / Math.max(target, 1e-9);
  const migrationBytes = moves.length * payloadBytes;

  return {
    moves,
    beforeImbalance: beforeImb,
    afterImbalance: afterImb,
    beforeMakespan,
    afterMakespan,
    migrationBytes,
    beforeLoad,
    afterLoad: load,
  };
}

/** Apply a migration plan to a tet partition (in-place clone). */
export function applyMigration(tetPart: Uint16Array, plan: RebalancePlan): Uint16Array {
  const out = new Uint16Array(tetPart);
  for (const m of plan.moves) out[m.tet] = m.to;
  return out;
}
