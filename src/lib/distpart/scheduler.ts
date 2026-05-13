/**
 * Communication-minimization scheduler.
 *
 * Takes a baseline HaloPlan + comm model and produces a *batched* schedule
 * that reduces inter-GPU traffic via three orthogonal techniques:
 *
 *   1. Temporal batching   — fuse `batchSize` consecutive sync iterations
 *                            into one larger transfer per (src,dst) pair so
 *                            per-message latency is amortized.
 *   2. Coalescing          — for each src rank, gather all destinations into
 *                            a single multi-dst NCCL send (one launch, one
 *                            handshake) up to `coalesceCap` peers per group.
 *   3. Delta compression   — drop tets that already shipped in the previous
 *                            iteration's batch when a stable subscription
 *                            covers them; only ship the residual delta.
 *
 * Returns the new schedule plus a savings report comparing baseline vs
 * batched bytes/messages/latency. The schedule is deterministic and reuses
 * the existing edge-coloring contract: at most one transfer per rank per
 * round.
 */

import type { HaloPlan, CommModelOptions } from "./comm";
import { DEFAULT_COMM_MODEL } from "./comm";

export interface BatchScheduleOptions {
  /** Iterations to fuse into a single physical transfer (≥1). */
  batchSize?: number;
  /** Total iterations the scheduler must cover. */
  iterations?: number;
  /** Max destinations coalesced under a single src per round. */
  coalesceCap?: number;
  /** Fraction of payload eliminated by delta compression (0..1). */
  deltaCompressionRatio?: number;
  comm?: CommModelOptions;
}

export interface BatchedTransfer {
  src: number;
  /** Destination ranks served by this physical transfer. */
  dsts: number[];
  /** Tets shipped (sum after coalescing + compression). */
  tets: number;
  /** Bytes shipped (tets × payloadBytes). */
  bytes: number;
  /** Iterations this transfer satisfies. */
  coversIterations: number;
}

export interface BatchedRound {
  /** Round index in the fused schedule. */
  index: number;
  /** Physical transfers issued in parallel this round. */
  transfers: BatchedTransfer[];
  /** Bytes shipped this round (max across active srcs). */
  parallelBytes: number;
  /** Wall-clock μs for this round (latency + max bytes / bandwidth). */
  roundUs: number;
}

export interface BatchSchedule {
  partitionCount: number;
  batchSize: number;
  iterations: number;
  rounds: BatchedRound[];
  /** Number of fused batches (= ⌈iterations / batchSize⌉). */
  batches: number;
  /** Baseline (un-batched) totals over the same iteration window. */
  baseline: {
    messages: number;
    bytes: number;
    totalUs: number;
  };
  /** Batched totals after fuse + coalesce + delta compression. */
  batched: {
    messages: number;
    bytes: number;
    totalUs: number;
  };
  /** 1 - batched/baseline ratios; report cards. */
  savings: {
    messages: number;
    bytes: number;
    latency: number;
  };
}

export function buildBatchSchedule(
  plan: HaloPlan,
  opts: BatchScheduleOptions = {},
): BatchSchedule {
  const comm = opts.comm ?? DEFAULT_COMM_MODEL;
  const batchSize = Math.max(1, Math.floor(opts.batchSize ?? 4));
  const iterations = Math.max(1, Math.floor(opts.iterations ?? 16));
  const coalesceCap = Math.max(1, Math.floor(opts.coalesceCap ?? 4));
  const compress = Math.max(0, Math.min(0.95, opts.deltaCompressionRatio ?? 0.35));
  const P = plan.partitionCount;
  const bps = comm.bandwidthGBs * 1e9;

  // Per (src,dst) baseline payload.
  type Edge = { src: number; dst: number; baseTets: number; baseBytes: number };
  const edges: Edge[] = [];
  for (const [key, list] of plan.sendLists) {
    if (!list.length) continue;
    const [s, d] = key.split(">").map(Number);
    const tets = list.length;
    edges.push({ src: s, dst: d, baseTets: tets, baseBytes: tets * comm.payloadBytes });
  }

  // Baseline totals across the iteration window.
  let baselineMsgs = 0;
  let baselineBytes = 0;
  let baselineUs = 0;
  for (const round of plan.rounds) {
    let bytesThisRound = 0;
    let active = 0;
    for (const { src, dst } of round) {
      const list = plan.sendLists.get(`${src}>${dst}`);
      if (!list || !list.length) continue;
      active++;
      bytesThisRound += list.length * comm.payloadBytes;
    }
    if (!active) continue;
    baselineMsgs += active * iterations;
    baselineBytes += bytesThisRound * iterations;
    baselineUs += iterations * (comm.latencyUs + (bytesThisRound / bps) * 1e6);
  }

  // Build batched rounds. Approach:
  //   - Number of fused batches = ⌈iterations / batchSize⌉.
  //   - For each batch, group edges by src; chunk dsts into coalesceCap.
  //   - Schedule chunks across rounds using the same edge-coloring rule
  //     (each rank participates in at most one transfer per round).
  const batches = Math.ceil(iterations / batchSize);
  const rounds: BatchedRound[] = [];
  let batchedMsgs = 0;
  let batchedBytes = 0;
  let batchedUs = 0;

  // Per-src grouped chunks (computed once; repeated per batch).
  const chunksPerSrc = new Map<number, { src: number; dsts: number[]; tets: number; bytes: number }[]>();
  const grouped = new Map<number, Edge[]>();
  for (const e of edges) {
    let arr = grouped.get(e.src);
    if (!arr) { arr = []; grouped.set(e.src, arr); }
    arr.push(e);
  }
  for (const [src, arr] of grouped) {
    arr.sort((a, b) => a.dst - b.dst);
    const chunks: { src: number; dsts: number[]; tets: number; bytes: number }[] = [];
    for (let i = 0; i < arr.length; i += coalesceCap) {
      const slice = arr.slice(i, i + coalesceCap);
      let tets = 0;
      let bytes = 0;
      for (const e of slice) { tets += e.baseTets; bytes += e.baseBytes; }
      chunks.push({ src, dsts: slice.map((e) => e.dst), tets, bytes });
    }
    chunksPerSrc.set(src, chunks);
  }

  for (let b = 0; b < batches; b++) {
    const itersInBatch = Math.min(batchSize, iterations - b * batchSize);
    // Compression: first iter pays full cost, subsequent pay (1-compress).
    const fusionFactor = 1 + (itersInBatch - 1) * (1 - compress);

    // Round-robin schedule chunks under the at-most-one-per-rank constraint.
    const pending = new Map<number, number>(); // src -> chunk cursor
    for (const src of chunksPerSrc.keys()) pending.set(src, 0);

    let safety = 0;
    while (safety++ < P * P + 8) {
      const usedSrc = new Uint8Array(P);
      const usedDst = new Uint8Array(P);
      const transfers: BatchedTransfer[] = [];
      let progressed = false;
      // Iterate srcs in deterministic order.
      const srcs = Array.from(chunksPerSrc.keys()).sort((a, b) => a - b);
      for (const src of srcs) {
        if (usedSrc[src]) continue;
        const cursor = pending.get(src)!;
        const chunks = chunksPerSrc.get(src)!;
        if (cursor >= chunks.length) continue;
        const chunk = chunks[cursor];
        // Skip if any destination busy this round.
        if (chunk.dsts.some((d) => usedDst[d])) continue;
        usedSrc[src] = 1;
        for (const d of chunk.dsts) usedDst[d] = 1;
        const tets = Math.round(chunk.tets * fusionFactor);
        const bytes = Math.round(chunk.bytes * fusionFactor);
        transfers.push({
          src,
          dsts: chunk.dsts.slice(),
          tets,
          bytes,
          coversIterations: itersInBatch,
        });
        pending.set(src, cursor + 1);
        progressed = true;
      }
      if (!progressed) break;
      let parallelBytes = 0;
      for (const t of transfers) if (t.bytes > parallelBytes) parallelBytes = t.bytes;
      const roundUs = comm.latencyUs + (parallelBytes / bps) * 1e6;
      rounds.push({ index: rounds.length, transfers, parallelBytes, roundUs });
      batchedMsgs += transfers.length;
      let bytesSum = 0;
      for (const t of transfers) bytesSum += t.bytes;
      batchedBytes += bytesSum;
      batchedUs += roundUs;
      // All srcs done?
      let done = true;
      for (const src of srcs) if (pending.get(src)! < chunksPerSrc.get(src)!.length) { done = false; break; }
      if (done) break;
    }
  }

  return {
    partitionCount: P,
    batchSize,
    iterations,
    rounds,
    batches,
    baseline: { messages: baselineMsgs, bytes: baselineBytes, totalUs: baselineUs },
    batched: { messages: batchedMsgs, bytes: batchedBytes, totalUs: batchedUs },
    savings: {
      messages: baselineMsgs ? 1 - batchedMsgs / baselineMsgs : 0,
      bytes: baselineBytes ? 1 - batchedBytes / baselineBytes : 0,
      latency: baselineUs ? 1 - batchedUs / baselineUs : 0,
    },
  };
}
