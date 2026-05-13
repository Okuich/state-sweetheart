/**
 * Adaptive Geometry Refinement Engine — top-level orchestration.
 *
 *   octree mesh + physics fields  →  per-leaf error indicator
 *                                  →  split / coarsen plan
 *                                  →  refined mesh (2:1 preserved)
 *                                  →  partition-aware halo update
 *                                  →  repartition hint
 *                                  →  prior store ingest
 *
 * Designed to run iteratively: each pass tightens the mesh where Physics OS
 * reports instability or high gradients, and relaxes it everywhere else.
 */

import type { AABB, OctreeMesh, OctreeOptions, RefinementSeed } from "../meshing/octree";
import type { PartitionPlan } from "../meshing/partition";
import { synthesizeFields, type PhysicsFields } from "./fields";
import {
  applyRefinement,
  computeLeafError,
  DEFAULT_REFINEMENT_OPTIONS,
  planRefinement,
  type LeafErrorReport,
  type RefinementOptions,
  type RefinementPass,
} from "./refine";
import {
  estimateHaloDelta,
  planRepartition,
  shouldRepartition,
  type RepartitionHint,
} from "./partition";
import { sharedPriorStore } from "./priors";

export * from "./fields";
export * from "./refine";
export * from "./partition";
export * from "./priors";

export interface AdaptivePassInput {
  bbox: AABB;
  baseSeeds: RefinementSeed[];
  baseMesh: OctreeMesh;
  basePartition: PartitionPlan;
  /** Provide explicit physics fields (real solver) or omit to synthesize. */
  fields?: PhysicsFields;
  step?: number;
  options?: Partial<RefinementOptions>;
  octreeOpts?: Partial<OctreeOptions>;
  ingestPriors?: boolean;
}

export interface AdaptivePassResult {
  fields: PhysicsFields;
  error: LeafErrorReport;
  pass: RefinementPass;
  haloDelta: { addedHalo: number; perPartition: number[] };
  repartition: RepartitionHint;
  shouldRepartition: boolean;
  priorsAdded: number;
  totalMs: number;
}

export function runAdaptivePass(input: AdaptivePassInput): AdaptivePassResult {
  const t0 = Date.now();
  const fields = input.fields ?? synthesizeFields(input.baseMesh, input.baseSeeds, input.step ?? 0);
  const error = computeLeafError(input.baseMesh, fields, {
    ...DEFAULT_REFINEMENT_OPTIONS.weights,
    ...(input.options?.weights ?? {}),
  });
  const plan = planRefinement(input.baseMesh, error, input.options);
  const pass = applyRefinement(input.bbox, input.baseSeeds, input.baseMesh, plan, input.octreeOpts);
  const haloDelta = estimateHaloDelta(input.baseMesh, input.basePartition, pass.newMesh, plan.splitLeaves);
  const repartition = planRepartition(input.basePartition, pass.newMesh, plan.splitLeaves, input.baseMesh);
  const priorsAdded = input.ingestPriors === false ? 0 : sharedPriorStore().ingest(input.baseMesh, error, plan);
  return {
    fields,
    error,
    pass,
    haloDelta,
    repartition,
    shouldRepartition: shouldRepartition(repartition),
    priorsAdded,
    totalMs: Date.now() - t0,
  };
}
