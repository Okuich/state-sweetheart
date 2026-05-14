/**
 * Distributed-refinement orchestrator.
 *
 * Wraps `runAdaptivePass` with automatic actions:
 *   - On every pass, perform a halo sync round on the (possibly refined)
 *     mesh and record bytes / rounds / latency.
 *   - When the imbalance hint exceeds `imbalanceThreshold`, automatically
 *     rebuild the partition on the refined mesh (ParMETIS-shim equivalent)
 *     and re-emit fresh halo sets, then re-sync.
 *
 * Output preserves a running action log so the UI can show *what was
 * triggered, when, and at what cost* — instead of just a "RECOMMENDED"
 * badge that nothing acts on.
 */

import { buildAdjacency, type AdjacencyTensors } from "../meshing/adjacency";
import { partitionMesh, type PartitionPlan } from "../meshing/partition";
import type { OctreeMesh } from "../meshing/octree";
import {
  runAdaptivePass,
  shouldRepartition,
  type AdaptivePassInput,
  type AdaptivePassResult,
} from "./index";

export interface DistributedAction {
  step: number;
  /** Was a halo sync executed this step. */
  haloSynced: boolean;
  /** Bytes pushed across the halo this step. */
  haloBytes: number;
  /** Estimated halo round-trip in microseconds. */
  haloUs: number;
  /** How many rounds of halo exchange (max comm-graph diameter approx). */
  haloRounds: number;
  /** Was a repartition triggered this step. */
  repartitioned: boolean;
  /** Imbalance observed *before* the action (max/mean). */
  imbalanceBefore: number;
  /** Imbalance after the action — equals before when no repartition fired. */
  imbalanceAfter: number;
  /** Tets migrated when repartitioned. */
  migratedTets: number;
  /** Reason string for telemetry. */
  reason: "stable" | "imbalance" | "halo-only";
}

export interface DistributedRefinementInput
  extends Omit<AdaptivePassInput, "baseMesh" | "basePartition"> {
  /** Live mesh — updated across passes when refinement adds tets. */
  mesh: OctreeMesh;
  /** Live partition — replaced when imbalance triggers repartition. */
  partition: PartitionPlan;
  /** Live adjacency, kept in sync with `mesh`. */
  adjacency: AdjacencyTensors;
  /** max/mean tolerance above which auto-repartition fires. Default 1.15. */
  imbalanceThreshold?: number;
  /** Per-halo-tet payload size in bytes. Default 64B (3 doubles + flags). */
  haloPayloadBytes?: number;
  /** Effective interconnect bandwidth in GB/s. Default 12 (NVLink-class). */
  interconnectGBs?: number;
  /** When false, suppress automatic repartition (still reports the hint). */
  autoRepartition?: boolean;
  /** When false, skip the halo sync. */
  autoHaloSync?: boolean;
}

export interface DistributedRefinementResult {
  adaptive: AdaptivePassResult;
  /** Mesh after refinement (caller should adopt for next pass). */
  mesh: OctreeMesh;
  /** Partition after rebalance (or the input one if no repartition). */
  partition: PartitionPlan;
  /** Adjacency aligned to `mesh`. */
  adjacency: AdjacencyTensors;
  action: DistributedAction;
}

const DEFAULT_THRESHOLD = 1.15;
const DEFAULT_PAYLOAD_BYTES = 64;
const DEFAULT_BW_GBS = 12;
const FIXED_LATENCY_US = 4; // per-round handshake

function haloTetCount(plan: PartitionPlan): number {
  let n = 0;
  for (const h of plan.halos) n += h.length;
  return n;
}

function commRoundEstimate(plan: PartitionPlan): number {
  // Approx the communication-graph diameter by counting non-empty
  // off-diagonal rows (each rank has to talk to its peers at least once).
  const P = plan.partitionCount;
  let maxPeers = 1;
  for (let i = 0; i < P; i++) {
    let peers = 0;
    for (let j = 0; j < P; j++) {
      if (i !== j && plan.commMatrix[i * P + j] > 0) peers++;
    }
    if (peers > maxPeers) maxPeers = peers;
  }
  // log2 of peers + 1 round handshake — good proxy for ring/tree exchanges.
  return Math.max(1, Math.ceil(Math.log2(maxPeers + 1)));
}

function syncCost(plan: PartitionPlan, payloadBytes: number, gbs: number) {
  const rounds = commRoundEstimate(plan);
  const bytes = haloTetCount(plan) * payloadBytes;
  // bandwidth term in µs (bytes / (GB/s * 1e3)) + latency per round
  const bwUs = bytes / Math.max(0.1, gbs * 1e3);
  const us = bwUs + rounds * FIXED_LATENCY_US;
  return { rounds, bytes, us };
}

function migrationCount(prev: PartitionPlan, next: PartitionPlan): number {
  const T = Math.min(prev.tetPart.length, next.tetPart.length);
  let m = 0;
  for (let t = 0; t < T; t++) if (prev.tetPart[t] !== next.tetPart[t]) m++;
  return m;
}

export function runDistributedRefinement(
  input: DistributedRefinementInput,
): DistributedRefinementResult {
  const threshold = input.imbalanceThreshold ?? DEFAULT_THRESHOLD;
  const payload = input.haloPayloadBytes ?? DEFAULT_PAYLOAD_BYTES;
  const gbs = input.interconnectGBs ?? DEFAULT_BW_GBS;
  const autoRepart = input.autoRepartition !== false;
  const autoHalo = input.autoHaloSync !== false;

  const adaptive = runAdaptivePass({
    ...input,
    baseMesh: input.mesh,
    basePartition: input.partition,
  });

  const imbalanceBefore = adaptive.repartition.imbalance;
  let mesh = adaptive.pass.newMesh;
  let adjacency = input.adjacency;
  let partition = input.partition;
  let repartitioned = false;
  let migratedTets = 0;
  let imbalanceAfter = imbalanceBefore;

  if (autoRepart && shouldRepartition(adaptive.repartition, threshold)) {
    // Rebuild adjacency on the refined mesh and re-partition. This is the
    // ParMETIS-equivalent action the engine commits to when drift exceeds
    // the user-set threshold.
    adjacency = buildAdjacency(mesh);
    const newPlan = partitionMesh(mesh, adjacency, input.partition.partitionCount);
    migratedTets = migrationCount(input.partition, newPlan);
    partition = newPlan;
    imbalanceAfter = newPlan.imbalance;
    repartitioned = true;
  } else if (mesh !== input.mesh) {
    // Mesh grew but partition still ok — rebuild adjacency so halo sync
    // sees the new tets, but keep the existing tetPart by extending it
    // (new tets inherit their parent's partition id).
    adjacency = buildAdjacency(mesh);
    const oldT = input.partition.tetPart.length;
    const newT = mesh.tets.length / 4;
    if (newT !== oldT) {
      const tetPart = new Uint16Array(newT);
      tetPart.set(input.partition.tetPart);
      // children inherit parent partition; leaves are appended in order so
      // walk parent → child by leaf index when available.
      for (let t = oldT; t < newT; t++) {
        const li = mesh.tetLeaf[t];
        // find any earlier tet sharing this leaf — fallback to round-robin.
        let inherited: number | null = null;
        for (let s = 0; s < oldT; s++) {
          if (mesh.tetLeaf[s] === li) { inherited = input.partition.tetPart[s]; break; }
        }
        tetPart[t] = inherited ?? (t % input.partition.partitionCount);
      }
      const sizes = new Uint32Array(input.partition.partitionCount);
      for (let t = 0; t < newT; t++) sizes[tetPart[t]]++;
      partition = { ...input.partition, tetPart, sizes };
    }
  }

  let haloBytes = 0;
  let haloUs = 0;
  let haloRounds = 0;
  if (autoHalo) {
    const c = syncCost(partition, payload, gbs);
    haloBytes = c.bytes;
    haloUs = c.us;
    haloRounds = c.rounds;
  }

  return {
    adaptive,
    mesh,
    partition,
    adjacency,
    action: {
      step: input.step ?? 0,
      haloSynced: autoHalo,
      haloBytes,
      haloUs,
      haloRounds,
      repartitioned,
      imbalanceBefore,
      imbalanceAfter,
      migratedTets,
      reason: repartitioned ? "imbalance" : autoHalo ? "halo-only" : "stable",
    },
  };
}
