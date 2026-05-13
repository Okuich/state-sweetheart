/**
 * Distributed Geometry Partitioning Engine.
 *
 *   mesh + adjacency  →  algorithm (regionGrow|spectral|morton|kway)
 *                     →  halo plan + comm scheduling (NCCL-compatible rounds)
 *                     →  dynamic rebalance from telemetry weights
 *                     →  checkpoint + recovery for partition failures
 *
 * Drives multi-GPU/multi-node Physics OS execution beneath Fabrication OS.
 */

import type { OctreeMesh } from "../meshing/octree";
import type { AdjacencyTensors } from "../meshing/adjacency";
import {
  runAlgorithm,
  type PartitionAlgorithm,
  type PartitionAssignment,
} from "./algorithms";
import {
  buildHaloPlan,
  commLatencyUs,
  DEFAULT_COMM_MODEL,
  type CommModelOptions,
  type HaloPlan,
} from "./comm";
import { planRebalance, applyMigration, type RebalancePlan } from "./rebalance";
import { buildCheckpoint, type DistributedCheckpoint } from "./checkpoint";
import { partitionAwareTraversal, type TraversalResult } from "./traversal";
import { simulateHaloSync, type HaloSyncResult } from "./halosync";
import { partitionBroadphase, type BroadphaseResult } from "./broadphase";

export * from "./algorithms";
export * from "./comm";
export * from "./rebalance";
export * from "./checkpoint";
export * from "./traversal";
export * from "./halosync";
export * from "./broadphase";

export interface DistPartInput {
  mesh: OctreeMesh;
  adj: AdjacencyTensors;
  partitionCount: number;
  algorithm?: PartitionAlgorithm;
  comm?: CommModelOptions;
  /** Per-tet workload weights for rebalance (defaults to uniform). */
  weights?: Float32Array;
  /** When true, run rebalance after initial partition. */
  rebalance?: boolean;
  /** Optional checkpoint step. */
  checkpointStep?: number;
  /** Run partition-aware traversal simulation. */
  traversal?: boolean;
  /** Run cross-partition halo sync simulation (iterations). */
  haloSyncIterations?: number;
  /** Run partition-aware broadphase. */
  broadphase?: boolean;
}

export interface DistPartResult {
  algorithm: PartitionAlgorithm;
  partitionCount: number;
  assignment: PartitionAssignment;
  halo: HaloPlan;
  latency: ReturnType<typeof commLatencyUs>;
  rebalance?: RebalancePlan;
  rebalanced?: PartitionAssignment;
  rebalancedHalo?: HaloPlan;
  checkpoint?: DistributedCheckpoint;
  traversal?: TraversalResult;
  haloSync?: HaloSyncResult;
  broadphase?: BroadphaseResult;
  totalMs: number;
}

export function planDistributed(input: DistPartInput): DistPartResult {
  const t0 = Date.now();
  const algo = input.algorithm ?? "kway";
  const comm = input.comm ?? DEFAULT_COMM_MODEL;
  const assignment = runAlgorithm(algo, input.mesh, input.adj, input.partitionCount);
  const halo = buildHaloPlan(assignment.tetPart, input.adj, input.partitionCount, comm);
  const latency = commLatencyUs(halo, comm);

  let rebalance: RebalancePlan | undefined;
  let rebalanced: PartitionAssignment | undefined;
  let rebalancedHalo: HaloPlan | undefined;
  if (input.rebalance) {
    const T = assignment.tetPart.length;
    const w = input.weights ?? new Float32Array(T).fill(1);
    rebalance = planRebalance({
      tetPart: assignment.tetPart,
      weights: w,
      partitionCount: input.partitionCount,
      adj: input.adj,
    });
    if (rebalance.moves.length) {
      const newPart = applyMigration(assignment.tetPart, rebalance);
      const stats = {
        ...assignment.stats,
        algorithm: "kway" as PartitionAlgorithm,
      };
      const sizes = new Uint32Array(input.partitionCount);
      for (let t = 0; t < newPart.length; t++) sizes[newPart[t]]++;
      let max = 0;
      for (let p = 0; p < sizes.length; p++) if (sizes[p] > max) max = sizes[p];
      const mean = T / input.partitionCount;
      rebalanced = {
        tetPart: newPart,
        stats: { ...stats, sizes, imbalance: mean ? max / mean : 1 },
      };
      rebalancedHalo = buildHaloPlan(newPart, input.adj, input.partitionCount, comm);
    }
  }

  const checkpoint = input.checkpointStep != null
    ? buildCheckpoint(
        (rebalanced ?? assignment).tetPart,
        input.checkpointStep,
        input.partitionCount,
      )
    : undefined;

  return {
    algorithm: algo,
    partitionCount: input.partitionCount,
    assignment,
    halo,
    latency,
    rebalance,
    rebalanced,
    rebalancedHalo,
    checkpoint,
    totalMs: Date.now() - t0,
  };
}
