/**
 * Cross-partition halo synchronization simulator.
 *
 * Runs N synchronization iterations against an existing HaloPlan,
 * producing per-iteration packet/byte/latency telemetry plus a staleness
 * model that captures how many ghost cells lag the owner by one step.
 *
 * Mirrors what a real multi-GPU solver does between physics sub-steps:
 *   pack owner field → NCCL all-to-all in scheduled rounds → unpack into
 *   ghost slots → solver reads owner+ghost as one fused buffer.
 */

import type { HaloPlan, CommModelOptions } from "./comm";
import { DEFAULT_COMM_MODEL } from "./comm";

export interface HaloSyncIteration {
  iter: number;
  /** Number of distinct (src,dst) packets sent. */
  packets: number;
  /** Tets shipped (sum across packets). */
  tetsExchanged: number;
  /** Bytes shipped (tets × payloadBytes). */
  bytes: number;
  /** Number of comm rounds (latency events). */
  rounds: number;
  /** Wall-clock μs estimate using bandwidth + latency model. */
  iterUs: number;
}

export interface HaloSyncResult {
  iterations: HaloSyncIteration[];
  totalBytes: number;
  totalUs: number;
  avgUs: number;
  /** Effective synchronization bandwidth (GB/s) achieved. */
  effectiveGBs: number;
  /** Worst-case ghost staleness in iterations across the schedule. */
  maxStaleness: number;
}

export interface HaloSyncOptions {
  iterations?: number;
  /** Inject a fault-injection probability (0–1) that drops a packet (forces resend). */
  packetLossRate?: number;
  comm?: CommModelOptions;
}

export function simulateHaloSync(
  plan: HaloPlan,
  opts: HaloSyncOptions = {},
): HaloSyncResult {
  const iterations = opts.iterations ?? 8;
  const comm = opts.comm ?? DEFAULT_COMM_MODEL;
  const loss = Math.max(0, Math.min(1, opts.packetLossRate ?? 0));
  const bps = comm.bandwidthGBs * 1e9;

  // deterministic LCG so traces are reproducible
  let seed = 0x12345678;
  const rand = () => {
    seed = (seed * 1664525 + 1013904223) >>> 0;
    return seed / 0xffffffff;
  };

  const iters: HaloSyncIteration[] = [];
  let totalBytes = 0;
  let totalUs = 0;
  let maxStaleness = 0;

  for (let i = 0; i < iterations; i++) {
    let packets = 0;
    let tets = 0;
    let bytes = 0;
    let iterUs = 0;
    let stale = 0;
    for (const round of plan.rounds) {
      let roundBytes = 0;
      let roundPackets = 0;
      for (const { src, dst } of round) {
        const list = plan.sendLists.get(`${src}>${dst}`);
        if (!list || !list.length) continue;
        const dropped = loss > 0 && rand() < loss;
        if (dropped) {
          // Resend doubles the packet (and adds another latency hop).
          stale++;
          packets++;
          roundPackets++;
          tets += list.length;
          roundBytes += list.length * comm.payloadBytes;
        }
        packets++;
        roundPackets++;
        tets += list.length;
        roundBytes += list.length * comm.payloadBytes;
      }
      if (roundPackets) {
        bytes += roundBytes;
        iterUs += comm.latencyUs + (roundBytes / bps) * 1e6;
      }
    }
    if (stale > maxStaleness) maxStaleness = stale;
    totalBytes += bytes;
    totalUs += iterUs;
    iters.push({ iter: i, packets, tetsExchanged: tets, bytes, rounds: plan.rounds.length, iterUs });
  }

  const effectiveGBs = totalUs > 0 ? totalBytes / 1e3 / totalUs : 0; // bytes / μs = MB/s; /1000 → GB/s
  return {
    iterations: iters,
    totalBytes,
    totalUs,
    avgUs: iterations ? totalUs / iterations : 0,
    effectiveGBs,
    maxStaleness,
  };
}
