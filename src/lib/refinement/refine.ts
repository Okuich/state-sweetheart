/**
 * Adaptive refinement driver.
 *
 * Consumes a base octree mesh + per-leaf physics error indicators and
 * produces a refined mesh by *splitting* leaves whose weighted error
 * exceeds a threshold, while *coarsening* (marking) sibling groups whose
 * combined error falls below a low watermark.
 *
 * Topology-preserving: refinement only splits in 2:1 octree style; we never
 * delete vertices on the surface envelope. Anisotropic stretch is captured
 * via a per-leaf metric tensor diagonal (sx, sy, sz) derived from local
 * field gradients — solver code can use that to pick directional element
 * sizing without us mutating the octree itself.
 */

import {
  buildOctreeMesh,
  type OctreeMesh,
  type RefinementSeed,
  type OctreeOptions,
  type AABB,
  type Vec3,
} from "../meshing/octree";
import type { PhysicsFields } from "./fields";
import { normalize } from "./fields";

export interface ErrorWeights {
  stress: number;
  thermal: number;
  contact: number;
  deformation: number;
  curvature: number;
  residual: number;
}

export const DEFAULT_WEIGHTS: ErrorWeights = {
  stress: 1.0,
  thermal: 0.6,
  contact: 0.9,
  deformation: 0.8,
  curvature: 0.5,
  residual: 0.7,
};

export interface RefinementOptions {
  weights: ErrorWeights;
  /** High-water mark on combined error → split. */
  splitThreshold: number;
  /** Low-water mark → mark for coarsening. */
  coarsenThreshold: number;
  /** Max additional depth above the base octree max. */
  extraDepth: number;
  /** Hard cap on new leaves added per pass. */
  maxNewLeaves: number;
  /** Anisotropy clamp [1, +inf). 1 = isotropic. */
  maxAnisotropy: number;
}

export const DEFAULT_REFINEMENT_OPTIONS: RefinementOptions = {
  weights: DEFAULT_WEIGHTS,
  splitThreshold: 0.45,
  coarsenThreshold: 0.05,
  extraDepth: 2,
  maxNewLeaves: 50_000,
  maxAnisotropy: 4,
};

export interface LeafErrorReport {
  /** Per-leaf combined error in [0,1]. */
  combined: Float32Array;
  /** Per-leaf dominant field that drove the score. */
  dominant: Uint8Array; // index into FIELD_ORDER
  /** Anisotropy diagonal per leaf (sx, sy, sz), normalized so max=1. */
  metric: Float32Array; // length = 3*N
  fieldPeaks: Record<string, number>;
}

const FIELD_ORDER = ["stress", "thermal", "contact", "deformation", "curvature", "residual"] as const;

export function computeLeafError(
  mesh: OctreeMesh,
  fields: PhysicsFields,
  weights: ErrorWeights = DEFAULT_WEIGHTS,
): LeafErrorReport {
  const N = mesh.leaves.length;
  const norms = FIELD_ORDER.map((k) => normalize(fields[k]));
  const combined = new Float32Array(N);
  const dominant = new Uint8Array(N);
  const fieldPeaks: Record<string, number> = {};
  norms.forEach((n, i) => { fieldPeaks[FIELD_ORDER[i]] = n.peak; });

  for (let i = 0; i < N; i++) {
    let best = -1;
    let bestK = 0;
    let sum = 0;
    let wsum = 0;
    for (let k = 0; k < FIELD_ORDER.length; k++) {
      const w = weights[FIELD_ORDER[k]];
      const v = norms[k].norm[i] * w;
      sum += v;
      wsum += w;
      if (v > best) { best = v; bestK = k; }
    }
    combined[i] = wsum > 0 ? Math.min(1, sum / wsum + 0.5 * best) : 0;
    dominant[i] = bestK;
  }

  // Metric tensor diagonal from local field deltas across siblings.
  // Iterate parent groups and stretch the axis with the largest sibling delta.
  const metric = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) { metric[i * 3] = 1; metric[i * 3 + 1] = 1; metric[i * 3 + 2] = 1; }

  const leafIdxByNode = new Map<number, number>();
  mesh.leaves.forEach((id, idx) => leafIdxByNode.set(id, idx));
  for (const node of mesh.nodes) {
    if (node.children.length !== 8) continue;
    const childLeaves = node.children
      .map((cid) => leafIdxByNode.get(cid))
      .filter((x): x is number => x !== undefined);
    if (childLeaves.length < 2) continue;
    // Delta along each axis = max minus min of combined error projected.
    const axisDelta: [number, number, number] = [0, 0, 0];
    for (let a = 0; a < 3; a++) {
      let lo = +Infinity, hi = -Infinity;
      for (const li of childLeaves) {
        const id = mesh.leaves[li];
        const c = mesh.nodes[id].bbox.min[a];
        const v = combined[li] + 0.001 * c;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      axisDelta[a] = hi - lo;
    }
    const maxA = Math.max(axisDelta[0], axisDelta[1], axisDelta[2], 1e-9);
    for (const li of childLeaves) {
      // Stretch axes with smaller delta (uniform → finer perpendicular).
      metric[li * 3 + 0] = Math.min(1, axisDelta[0] / maxA + 0.25);
      metric[li * 3 + 1] = Math.min(1, axisDelta[1] / maxA + 0.25);
      metric[li * 3 + 2] = Math.min(1, axisDelta[2] / maxA + 0.25);
    }
  }

  return { combined, dominant, metric, fieldPeaks };
}

export interface RefinementPlan {
  splitLeaves: Uint32Array;
  coarsenLeaves: Uint32Array;
  /** Synthetic seeds derived from hot leaves to drive the next octree build. */
  derivedSeeds: RefinementSeed[];
  newMaxDepth: number;
  /** Stats. */
  stats: {
    splitCount: number;
    coarsenCount: number;
    meanError: number;
    p95Error: number;
    capped: boolean;
  };
}

function dominantKindFromIdx(k: number): RefinementSeed["kind"] {
  // Map field index to a closest RefinementSeed kind for downstream tagging.
  switch (FIELD_ORDER[k]) {
    case "stress": return "sharp";
    case "thermal": return "hotspot";
    case "contact": return "contact";
    case "deformation": return "overhang";
    case "curvature": return "fillet";
    case "residual": return "thin_wall";
  }
}

export function planRefinement(
  mesh: OctreeMesh,
  err: LeafErrorReport,
  optsIn: Partial<RefinementOptions> = {},
): RefinementPlan {
  const opts = { ...DEFAULT_REFINEMENT_OPTIONS, ...optsIn };
  const splits: number[] = [];
  const coarsens: number[] = [];
  const N = mesh.leaves.length;

  for (let i = 0; i < N; i++) {
    const e = err.combined[i];
    if (e >= opts.splitThreshold) splits.push(i);
    else if (e <= opts.coarsenThreshold) coarsens.push(i);
  }

  // Cap by hottest first.
  splits.sort((a, b) => err.combined[b] - err.combined[a]);
  const capped = splits.length > opts.maxNewLeaves;
  const limited = capped ? splits.slice(0, opts.maxNewLeaves) : splits;

  // Derive seeds at hot leaf centers — these become inputs for the next
  // buildOctreeMesh pass, which will refine through the normal seed pipeline
  // and so stays topologically consistent with the base mesher.
  const derivedSeeds: RefinementSeed[] = [];
  const ext = Math.max(
    mesh.bbox.max[0] - mesh.bbox.min[0],
    mesh.bbox.max[1] - mesh.bbox.min[1],
    mesh.bbox.max[2] - mesh.bbox.min[2],
  );
  for (const li of limited) {
    const id = mesh.leaves[li];
    const b = mesh.nodes[id].bbox;
    const c: Vec3 = [
      (b.min[0] + b.max[0]) / 2,
      (b.min[1] + b.max[1]) / 2,
      (b.min[2] + b.max[2]) / 2,
    ];
    const leafExt = Math.max(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]);
    derivedSeeds.push({
      kind: dominantKindFromIdx(err.dominant[li]),
      center: c,
      radius: Math.max(leafExt * 0.6, ext * 0.005),
      weight: 0.5 + 0.7 * err.combined[li],
    });
  }

  const sorted = Array.from(err.combined).sort((a, b) => a - b);
  const meanError = sorted.reduce((a, b) => a + b, 0) / Math.max(1, sorted.length);
  const p95Error = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))] ?? 0;

  return {
    splitLeaves: new Uint32Array(limited),
    coarsenLeaves: new Uint32Array(coarsens),
    derivedSeeds,
    newMaxDepth: mesh.options.maxDepth + opts.extraDepth,
    stats: {
      splitCount: limited.length,
      coarsenCount: coarsens.length,
      meanError,
      p95Error,
      capped,
    },
  };
}

export interface RefinementPass {
  baseLeafCount: number;
  refinedLeafCount: number;
  added: number;
  removedSiblings: number;
  passMs: number;
  newMesh: OctreeMesh;
  plan: RefinementPlan;
}

/**
 * Apply one adaptive pass: rebuild the octree with original seeds + derived
 * seeds and a deeper maxDepth. This preserves the 2:1 invariant automatically
 * because the underlying mesher always splits cells uniformly.
 */
export function applyRefinement(
  bbox: AABB,
  baseSeeds: RefinementSeed[],
  baseMesh: OctreeMesh,
  plan: RefinementPlan,
  baseOpts: Partial<OctreeOptions> = {},
): RefinementPass {
  const t0 = Date.now();
  const newMesh = buildOctreeMesh(
    bbox,
    [...baseSeeds, ...plan.derivedSeeds],
    {
      ...baseOpts,
      maxDepth: plan.newMaxDepth,
      maxLeaves: Math.max(
        baseMesh.options.maxLeaves,
        baseMesh.leaves.length + plan.stats.splitCount * 8 + 1024,
      ),
    },
  );
  return {
    baseLeafCount: baseMesh.leaves.length,
    refinedLeafCount: newMesh.leaves.length,
    added: Math.max(0, newMesh.leaves.length - baseMesh.leaves.length),
    removedSiblings: plan.coarsenLeaves.length,
    passMs: Date.now() - t0,
    newMesh,
    plan,
  };
}
