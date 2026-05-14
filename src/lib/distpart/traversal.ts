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
import type { BatchSchedule } from "./scheduler";

export interface TraversalStep {
  step: number;
  /** Tets visited per rank during the local sweep this step. */
  localVisits: Uint32Array;
  /** Cross-partition messages dispatched this step (after coalescing). */
  haloMessages: number;
  /** Halo tets exchanged this step (sum across all sender→receiver pairs). */
  haloTets: number;
  /** Bytes shipped this step using payload model (after compression). */
  haloBytes: number;
  /** Was this a batched flush step. */
  flushed: boolean;
  /** Max(localVisits) — the rank that bottlenecked this step. */
  parallelWork: number;
  /** Σ(localVisits) — equivalent serial work. */
  serialWork: number;
}

export interface BatchedTraversalStats {
  /** Total physical NCCL launches across all flush steps. */
  messages: number;
  /** Total bytes physically transferred (post compression / coalescing). */
  bytes: number;
  /** Total wall μs spent on halo sync (sum of flush rounds). */
  us: number;
  /** Number of flush events. */
  flushes: number;
  /** Worst observed gap between produce-step and deliver-step (in steps). */
  maxStaleness: number;
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
  /** Populated when traversal ran with a BatchSchedule. */
  batched?: BatchedTraversalStats;
}

export interface TraversalOptions {
  /** One seed tet per rank. Defaults to the first locally-owned tet. */
  seeds?: (number | undefined)[];
  /** Hard cap on steps — guards pathological cases. */
  maxSteps?: number;
  /** Compute time per visited tet (microseconds), for the wall-clock model. */
  perTetUs?: number;
  comm?: CommModelOptions;
  /**
   * When provided, the traversal honours the batched-schedule protocol:
   * halo seeds are accumulated for `batchSize` consecutive steps and shipped
   * in one coalesced flush per src (up to `coalesceCap` dsts), with the
   * scheduler's `deltaCompressionRatio` applied to repeated tets.
   * This makes simulator wall μs / bytes match the BatchSchedule projection.
   */
  schedule?: BatchSchedule;
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

  // Batched-protocol state. When `opts.schedule` is provided, halo seeds are
  // accumulated across `batchSize` consecutive steps and shipped in one
  // coalesced flush per src — matching the BatchSchedule projection so the
  // simulator's wall μs / bytes are consistent with what the scheduler reports.
  const sched = opts.schedule;
  const batchSize = sched ? Math.max(1, sched.batchSize) : 1;
  const coalesceCap = sched ? Math.max(1, Math.floor((sched.iterations / Math.max(1, sched.batches)) * 0)) || 4 : 4;
  // delta-compression ratio derived from the schedule's batched/baseline byte
  // ratio per fused batch (1 - savings.bytes is the residual fraction; with
  // batchSize iters fused, per-iter compression ≈ 1 - residual^(1/batchSize)).
  const compressRatio = sched && sched.baseline.bytes
    ? Math.max(0, Math.min(0.95, 1 - sched.batched.bytes / sched.baseline.bytes))
    : 0;

  // Pending tets per (src,dst) and per-pair earliest produce step (for staleness).
  const pending = new Map<string, Set<number>>();
  const pairFirstStep = new Map<string, number>();

  let batchMessages = 0;
  let batchBytes = 0;
  let batchUs = 0;
  let flushes = 0;
  let maxStaleness = 0;

  const sendByPair = new Map<string, number[]>(); // src>dst → [tet ids]
  const bps = comm.bandwidthGBs * 1e9;

  function flushBatch(currentStep: number): { msgs: number; tets: number; bytes: number; us: number } {
    if (!pending.size) return { msgs: 0, tets: 0, bytes: 0, us: 0 };
    flushes++;
    // Group pending pairs by src, sort dsts deterministically.
    const bySrc = new Map<number, { dst: number; tets: number[] }[]>();
    for (const [key, set] of pending) {
      const [s, d] = key.split(">").map(Number);
      let arr = bySrc.get(s);
      if (!arr) { arr = []; bySrc.set(s, arr); }
      arr.push({ dst: d, tets: Array.from(set) });
    }
    for (const arr of bySrc.values()) arr.sort((a, b) => a.dst - b.dst);

    let flushMsgs = 0;
    let flushTets = 0;
    let flushBytes = 0;
    // Edge-color the coalesced chunks the same way the scheduler does.
    const cursor = new Map<number, number>();
    for (const s of bySrc.keys()) cursor.set(s, 0);
    let safety = 0;
    let maxRoundUs = 0;
    let totalRoundUs = 0;
    while (safety++ < P * P + 8) {
      const usedSrc = new Uint8Array(P);
      const usedDst = new Uint8Array(P);
      const transfers: { src: number; dsts: number[]; bytes: number }[] = [];
      let progressed = false;
      for (const src of Array.from(bySrc.keys()).sort((a, b) => a - b)) {
        if (usedSrc[src]) continue;
        const arr = bySrc.get(src)!;
        let cur = cursor.get(src)!;
        if (cur >= arr.length) continue;
        const slice = arr.slice(cur, cur + coalesceCap);
        if (slice.some((s) => usedDst[s.dst])) continue;
        // Coalesced multi-dst transfer: bytes = Σ uniq tets * payload, with
        // delta compression on the repeats accumulated across the batch.
        let rawTets = 0;
        for (const s of slice) rawTets += s.tets.length;
        const fused = Math.round(rawTets * (1 - compressRatio * (batchSize - 1) / batchSize));
        const bytes = fused * comm.payloadBytes;
        usedSrc[src] = 1;
        for (const s of slice) usedDst[s.dst] = 1;
        transfers.push({ src, dsts: slice.map((s) => s.dst), bytes });
        flushMsgs++;
        flushTets += fused;
        flushBytes += bytes;
        // Deliver into receiver inboxes — staleness is now bounded by batchSize.
        for (const s of slice) {
          for (const t of s.tets) inbox[s.dst].push(t);
          const k = `${src}>${s.dst}`;
          const first = pairFirstStep.get(k) ?? currentStep;
          const stale = currentStep - first;
          if (stale > maxStaleness) maxStaleness = stale;
          pending.delete(k);
          pairFirstStep.delete(k);
        }
        cursor.set(src, cur + slice.length);
        progressed = true;
      }
      if (!progressed) break;
      let parallelBytes = 0;
      for (const t of transfers) if (t.bytes > parallelBytes) parallelBytes = t.bytes;
      const roundUs = comm.latencyUs + (parallelBytes / bps) * 1e6;
      totalRoundUs += roundUs;
      if (roundUs > maxRoundUs) maxRoundUs = roundUs;
    }
    return { msgs: flushMsgs, tets: flushTets, bytes: flushBytes, us: totalRoundUs };
  }

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
    let stepHaloBytes = 0;
    let stepHaloUs = 0;
    let flushed = false;
    let anyWork = false;

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
            visited[n] = 1;
            frontiers[p].push(n);
          } else {
            const key = `${p}>${pn}`;
            let list = sendByPair.get(key);
            if (!list) { list = []; sendByPair.set(key, list); }
            list.push(n);
          }
        }
      }
    }

    if (sched) {
      // Accumulate this step's sends into the pending batch.
      for (const [key, list] of sendByPair) {
        let set = pending.get(key);
        if (!set) {
          set = new Set<number>();
          pending.set(key, set);
          pairFirstStep.set(key, step);
        }
        for (const t of list) set.add(t);
      }
      // Flush every `batchSize` steps, OR when no further work would unblock
      // remaining receivers (no local progress AND pending is non-empty).
      const isBoundary = (step + 1) % batchSize === 0;
      const stalled = !anyWork && pending.size > 0;
      if (isBoundary || stalled) {
        const f = flushBatch(step);
        flushed = f.msgs > 0;
        stepHaloMsgs = f.msgs;
        stepHaloTets = f.tets;
        stepHaloBytes = f.bytes;
        stepHaloUs = f.us;
        batchMessages += f.msgs;
        batchBytes += f.bytes;
        batchUs += f.us;
      }
    } else {
      // Baseline: deliver via plan rounds + flush stragglers immediately.
      for (const round of halo.rounds) {
        for (const { src, dst } of round) {
          const list = sendByPair.get(`${src}>${dst}`);
          if (!list || !list.length) continue;
          const uniq = Array.from(new Set(list));
          stepHaloMsgs++;
          stepHaloTets += uniq.length;
          for (const t of uniq) inbox[dst].push(t);
          sendByPair.delete(`${src}>${dst}`);
        }
      }
      for (const [key, list] of sendByPair) {
        const [, dStr] = key.split(">");
        const dst = +dStr;
        const uniq = Array.from(new Set(list));
        stepHaloMsgs++;
        stepHaloTets += uniq.length;
        for (const t of uniq) inbox[dst].push(t);
      }
      stepHaloBytes = stepHaloTets * comm.payloadBytes;
      stepHaloUs = stepHaloMsgs ? comm.latencyUs + (stepHaloBytes / bps) * 1e6 : 0;
    }

    if (!anyWork && !steps.length && !flushed) break;

    let parallel = 0;
    let serial = 0;
    for (let p = 0; p < P; p++) {
      serial += localVisits[p];
      if (localVisits[p] > parallel) parallel = localVisits[p];
    }
    totalVisited += serial;
    totalParallel += parallel;
    totalSerial += serial;
    totalHaloBytes += stepHaloBytes;
    const computeUs = parallel * perTetUs;
    estimatedUs += computeUs + stepHaloUs;

    steps.push({
      step,
      localVisits,
      haloMessages: stepHaloMsgs,
      haloTets: stepHaloTets,
      haloBytes: stepHaloBytes,
      flushed,
      parallelWork: parallel,
      serialWork: serial,
    });

    let pendingFrontier = 0;
    for (let p = 0; p < P; p++) pendingFrontier += frontiers[p].length + inbox[p].length;
    if (!pendingFrontier && !pending.size) break;
  }

  // Drain anything still pending at termination.
  if (sched && pending.size) {
    const f = flushBatch(steps.length);
    if (f.msgs) {
      batchMessages += f.msgs;
      batchBytes += f.bytes;
      batchUs += f.us;
      totalHaloBytes += f.bytes;
      estimatedUs += f.us;
    }
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
    batched: sched
      ? { messages: batchMessages, bytes: batchBytes, us: batchUs, flushes, maxStaleness }
      : undefined,
  };
}
