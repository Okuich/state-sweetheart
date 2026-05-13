/**
 * Partition-aware multi-GPU traversal.
 *
 * Simulates a synchronous BFS over the tet-tet adjacency graph where each
 * rank owns a partition. Traversal proceeds in `super-steps`:
 *
 *   1. Each rank expands its local frontier in parallel (intra-rank).
 *   2. Cross-partition neighbors are deferred to a halo queue.
 *   3. Halo seeds are exchanged via the existing HaloPlan (one comm round).
 *   4. Receiving ranks enqueue them as next-step frontier seeds.
 *
 * This mirrors how a multi-GPU physics solver advances a wave (stress, heat,
 * collision) across partition boundaries: local sweep → halo sync → repeat.
 *
 * The simulator returns per-step parallel cost (max-rank work) and
 * sequential cost (sum across ranks) so we can derive theoretical speedup.
 */

import type { AdjacencyTensors } from "../meshing/adjacency";
import type { HaloPlan } from "./comm";
import type { CommModelOptions } from "./comm";
import { DEFAULT_COMM_MODEL } from "./comm";

export interface TraversalStep {
  step: number;
  /** Tets visited per rank during the local sweep this step. */
  localVisits: Uint32Array;
  /** Cross-partition messages dispatched this step. */
  haloMessages: number;
  /** Halo tets exchanged this step (sum across all sender→receiver pairs). */
  haloTets: number;
  /** Bytes shipped this step using payload model. */
  haloBytes: number;
  /** Max(localVisits) — the rank that bottlenecked this step. */
  parallelWork: number;
  /** Σ(localVisits) — equivalent serial work. */
  serialWork: number;
}

export interface TraversalResult {
  partitionCount: number;
  steps: TraversalStep[];
  totalVisited: number;
  totalParallelWork: number;
  totalSerialWork: number;
  /** Σ serial / Σ parallel — Brent-style upper bound on speedup. */
  speedup: number;
  /** Total bytes synchronized across all halo exchanges. */
  totalHaloBytes: number;
  /** Estimated wall-clock μs combining compute + halo latency per step. */
  estimatedUs: number;
  /** Compute-time budget per local visit (μs/tet) used in the estimate. */
  perTetUs: number;
}

export interface TraversalOptions {
  /** One seed tet per rank. Defaults to the first locally-owned tet. */
  seeds?: (number | undefined)[];
  /** Hard cap on steps — guards pathological cases. */
  maxSteps?: number;
  /** Compute time per visited tet (microseconds), for the wall-clock model. */
  perTetUs?: number;
  comm?: CommModelOptions;
}

export function partitionAwareTraversal(
  tetPart: Uint16Array,
  adj: AdjacencyTensors,
  halo: HaloPlan,
  opts: TraversalOptions = {},
): TraversalResult {
  const T = tetPart.length;
  const P = halo.partitionCount;
  const comm = opts.comm ?? DEFAULT_COMM_MODEL;
  const perTetUs = opts.perTetUs ?? 0.05;
  const maxSteps = opts.maxSteps ?? T + 4;

  // Pick seeds: first owned tet per rank.
  const seeds = (opts.seeds ?? []).slice();
  if (seeds.length < P) {
    const filled = new Uint8Array(P);
    for (let i = 0; i < seeds.length; i++) if (seeds[i] != null) filled[i] = 1;
    for (let t = 0; t < T && seeds.length < P; t++) {
      const p = tetPart[t];
      if (!filled[p]) {
        seeds[p] = t;
        filled[p] = 1;
      }
    }
    for (let p = 0; p < P; p++) if (seeds[p] == null) seeds[p] = -1;
  }

  const visited = new Uint8Array(T);
  // Per-rank current frontier (local tets) and pending inbox (from halo).
  const frontiers: number[][] = Array.from({ length: P }, () => []);
  const inbox: number[][] = Array.from({ length: P }, () => []);
  for (let p = 0; p < P; p++) {
    const s = seeds[p];
    if (s != null && s >= 0 && s < T && tetPart[s] === p && !visited[s]) {
      frontiers[p].push(s);
      visited[s] = 1;
    }
  }

  const steps: TraversalStep[] = [];
  let totalVisited = 0;
  let totalParallel = 0;
  let totalSerial = 0;
  let totalHaloBytes = 0;
  let estimatedUs = 0;

  const sendByPair = new Map<string, number[]>(); // src>dst → [tet ids]

  for (let step = 0; step < maxSteps; step++) {
    // Drain inbox into frontier.
    for (let p = 0; p < P; p++) {
      if (inbox[p].length) {
        for (const t of inbox[p]) {
          if (!visited[t] && tetPart[t] === p) {
            visited[t] = 1;
            frontiers[p].push(t);
          }
        }
        inbox[p].length = 0;
      }
    }

    const localVisits = new Uint32Array(P);
    sendByPair.clear();
    let stepHaloMsgs = 0;
    let stepHaloTets = 0;
    let anyWork = false;

    // Snapshot frontiers (so neighbours added this step expand next step).
    const snapshot = frontiers.map((f) => f.slice());
    for (let p = 0; p < P; p++) frontiers[p].length = 0;

    for (let p = 0; p < P; p++) {
      const front = snapshot[p];
      if (!front.length) continue;
      anyWork = true;
      localVisits[p] = front.length;
      for (const t of front) {
        const s = adj.ttRowPtr[t];
        const e = adj.ttRowPtr[t + 1];
        for (let k = s; k < e; k++) {
          const n = adj.ttCol[k];
          if (visited[n]) continue;
          const pn = tetPart[n];
          if (pn === p) {
            // Local expansion — schedule for next step.
            visited[n] = 1;
            frontiers[p].push(n);
          } else {
            // Cross-partition: ship neighbour as halo seed to its owner.
            const key = `${p}>${pn}`;
            let list = sendByPair.get(key);
            if (!list) { list = []; sendByPair.set(key, list); }
            list.push(n);
          }
        }
      }
    }

    // Apply halo plan rounds for this step.
    for (const round of halo.rounds) {
      for (const { src, dst } of round) {
        const list = sendByPair.get(`${src}>${dst}`);
        if (!list || !list.length) continue;
        // Dedupe per message.
        const uniq = Array.from(new Set(list));
        stepHaloMsgs++;
        stepHaloTets += uniq.length;
        for (const t of uniq) inbox[dst].push(t);
        sendByPair.delete(`${src}>${dst}`);
      }
    }
    // Any leftover pairs (not covered by scheduled rounds) flush directly.
    for (const [key, list] of sendByPair) {
      const [, dStr] = key.split(">");
      const dst = +dStr;
      const uniq = Array.from(new Set(list));
      stepHaloMsgs++;
      stepHaloTets += uniq.length;
      for (const t of uniq) inbox[dst].push(t);
    }

    if (!anyWork && !steps.length) break;

    let parallel = 0;
    let serial = 0;
    for (let p = 0; p < P; p++) {
      serial += localVisits[p];
      if (localVisits[p] > parallel) parallel = localVisits[p];
    }
    const haloBytes = stepHaloTets * comm.payloadBytes;
    totalVisited += serial;
    totalParallel += parallel;
    totalSerial += serial;
    totalHaloBytes += haloBytes;
    // Wall-clock: max-rank compute + (one halo round latency per nonzero pair group).
    const computeUs = parallel * perTetUs;
    const haloUs = stepHaloMsgs ? comm.latencyUs + (haloBytes / (comm.bandwidthGBs * 1e9)) * 1e6 : 0;
    estimatedUs += computeUs + haloUs;

    steps.push({
      step,
      localVisits,
      haloMessages: stepHaloMsgs,
      haloTets: stepHaloTets,
      haloBytes,
      parallelWork: parallel,
      serialWork: serial,
    });

    // Termination: nothing produced and no inbox.
    let pending = 0;
    for (let p = 0; p < P; p++) pending += frontiers[p].length + inbox[p].length;
    if (!pending) break;
  }

  return {
    partitionCount: P,
    steps,
    totalVisited,
    totalParallelWork: totalParallel,
    totalSerialWork: totalSerial,
    speedup: totalParallel ? totalSerial / totalParallel : 1,
    totalHaloBytes,
    estimatedUs,
    perTetUs,
  };
}
