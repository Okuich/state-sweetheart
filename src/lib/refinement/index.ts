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
  physicsFeedbackBus,
  projectFeedbackToFields,
  type PhysicsSnapshot,
} from "./physicsFeedback";
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
export * from "./physicsFeedback";
export * from "./distributed";

/** How the per-leaf physics fields are obtained for this pass. */
export type FieldSource = "physics" | "synthetic" | "explicit";

export interface AdaptivePassInput {
  bbox: AABB;
  baseSeeds: RefinementSeed[];
  baseMesh: OctreeMesh;
  basePartition: PartitionPlan;
  /** Provide explicit physics fields (real solver) or omit to derive. */
  fields?: PhysicsFields;
  /**
   * Where to source per-leaf fields from when `fields` is not provided:
   *   - "auto" (default): use a fresh `physicsFeedbackBus` snapshot if
   *     available within `feedbackMaxAgeMs`, else fall back to synthetic.
   *   - "physics": require a fresh snapshot; fall back to synthetic if
   *     none is available (and surface that in `fieldSource`).
   *   - "synthetic": always synthesize.
   */
  feedbackSource?: "auto" | "physics" | "synthetic";
  /** Max age for a feedback snapshot to count as fresh, in ms. */
  feedbackMaxAgeMs?: number;
  /** Override snapshot lookup (tests). */
  snapshot?: PhysicsSnapshot | null;
  step?: number;
  options?: Partial<RefinementOptions>;
  octreeOpts?: Partial<OctreeOptions>;
  ingestPriors?: boolean;
}

export interface AdaptivePassResult {
  fields: PhysicsFields;
  fieldSource: FieldSource;
  /** Snapshot used (when fieldSource === "physics"). */
  snapshot?: PhysicsSnapshot;
  /** Age in ms of the snapshot at pass time. */
  snapshotAgeMs?: number;
  error: LeafErrorReport;
  pass: RefinementPass;
  haloDelta: { addedHalo: number; perPartition: number[] };
  repartition: RepartitionHint;
  shouldRepartition: boolean;
  priorsAdded: number;
  totalMs: number;
}

const DEFAULT_FEEDBACK_MAX_AGE_MS = 1500;

export function runAdaptivePass(input: AdaptivePassInput): AdaptivePassResult {
  const t0 = Date.now();
  const mode = input.feedbackSource ?? "auto";
  const maxAge = input.feedbackMaxAgeMs ?? DEFAULT_FEEDBACK_MAX_AGE_MS;
  const snap = input.snapshot !== undefined ? input.snapshot : physicsFeedbackBus.latest();
  const snapshotAgeMs = snap ? t0 - snap.t : Infinity;

  let fields: PhysicsFields;
  let fieldSource: FieldSource;
  let usedSnapshot: PhysicsSnapshot | undefined;

  if (input.fields) {
    fields = input.fields;
    fieldSource = "explicit";
  } else if (mode !== "synthetic" && snap && snapshotAgeMs <= maxAge) {
    fields = projectFeedbackToFields(input.baseMesh, snap);
    fieldSource = "physics";
    usedSnapshot = snap;
  } else {
    fields = synthesizeFields(input.baseMesh, input.baseSeeds, input.step ?? 0);
    fieldSource = "synthetic";
  }

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
    fieldSource,
    snapshot: usedSnapshot,
    snapshotAgeMs: usedSnapshot ? snapshotAgeMs : undefined,
    error,
    pass,
    haloDelta,
    repartition,
    shouldRepartition: shouldRepartition(repartition),
    priorsAdded,
    totalMs: Date.now() - t0,
  };
}
