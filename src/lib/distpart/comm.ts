/**
 * Communication graph + halo planning.
 *
 * Given a tet partition, compute:
 *   - per-partition halo (neighbor tets owned by other ranks)
 *   - send/recv lists per (src, dst) pair (NCCL-compatible)
 *   - communication volume + latency model
 *   - schedule of non-overlapping rounds (edge-coloring on the comm graph)
 *
 * The comm-volume model is the standard: bytes_round ≈ Σ |send| × payload.
 */

import type { AdjacencyTensors } from "../meshing/adjacency";

export interface HaloPlan {
  partitionCount: number;
  /** per-partition tet ids that are remote-owned but read by this rank */
  haloRecv: Uint32Array[];
  /** per (src,dst) tet ids src must send to dst (ghost copies) */
  sendLists: Map<string, Uint32Array>;
  /** total cross-partition tet faces */
  edgeCut: number;
  /** flat P×P comm-volume (tets sent) matrix */
  commMatrix: Uint32Array;
  /** estimated bytes per synchronization round given payload */
  syncBytes: number;
  /** scheduled comm rounds (edge-coloring) */
  rounds: { src: number; dst: number }[][];
}

export interface CommModelOptions {
  /** Bytes per tet payload (e.g. 24 = vec3 f64 stress + scalar). */
  payloadBytes: number;
  /** Bandwidth for round budget (GB/s). */
  bandwidthGBs: number;
  /** Per-message latency (microseconds). */
  latencyUs: number;
}

export const DEFAULT_COMM_MODEL: CommModelOptions = {
  payloadBytes: 32,
  bandwidthGBs: 200, // NVLink-class
  latencyUs: 5,
};

export function buildHaloPlan(
  tetPart: Uint16Array,
  adj: AdjacencyTensors,
  P: number,
  model: CommModelOptions = DEFAULT_COMM_MODEL,
): HaloPlan {
  const T = tetPart.length;
  const recvSets: Set<number>[] = Array.from({ length: P }, () => new Set<number>());
  const sendKey = (src: number, dst: number) => `${src}>${dst}`;
  const sendSets = new Map<string, Set<number>>();
  const commMatrix = new Uint32Array(P * P);
  let edgeCut = 0;

  for (let t = 0; t < T; t++) {
    const pt = tetPart[t];
    const s = adj.ttRowPtr[t], e = adj.ttRowPtr[t + 1];
    for (let k = s; k < e; k++) {
      const n = adj.ttCol[k];
      const pn = tetPart[n];
      if (pn === pt) continue;
      // n is owned by pn, read by pt → recv into pt.
      recvSets[pt].add(n);
      // pt must send t to pn.
      const key = sendKey(pt, pn);
      let s2 = sendSets.get(key);
      if (!s2) { s2 = new Set<number>(); sendSets.set(key, s2); }
      s2.add(t);
      commMatrix[pt * P + pn]++;
      edgeCut++;
    }
  }

  const haloRecv = recvSets.map((s) => new Uint32Array(Array.from(s).sort((a, b) => a - b)));
  const sendLists = new Map<string, Uint32Array>();
  for (const [k, s] of sendSets) sendLists.set(k, new Uint32Array(Array.from(s).sort((a, b) => a - b)));

  // Sync bytes: union of sends per pair × payload.
  let totalTetSends = 0;
  for (const v of sendLists.values()) totalTetSends += v.length;
  const syncBytes = totalTetSends * model.payloadBytes;

  // Edge-color the communication graph so each rank does at most one
  // send/recv per round (1-factorization on the symmetric pattern).
  const rounds = scheduleRounds(P, sendLists);

  return {
    partitionCount: P,
    haloRecv,
    sendLists,
    edgeCut: edgeCut / 2,
    commMatrix,
    syncBytes,
    rounds,
  };
}

function scheduleRounds(
  P: number,
  sendLists: Map<string, Uint32Array>,
): { src: number; dst: number }[][] {
  const pending = new Set<string>();
  for (const k of sendLists.keys()) if (sendLists.get(k)!.length) pending.add(k);
  const rounds: { src: number; dst: number }[][] = [];
  const guard = P * P + 1;
  while (pending.size && rounds.length < guard) {
    const used = new Uint8Array(P);
    const round: { src: number; dst: number }[] = [];
    // Greedy: pick edges in lexicographic order, skipping busy ranks.
    const sorted = Array.from(pending).sort();
    for (const k of sorted) {
      const [s, d] = k.split(">").map(Number);
      if (used[s] || used[d]) continue;
      used[s] = 1; used[d] = 1;
      round.push({ src: s, dst: d });
      pending.delete(k);
    }
    if (!round.length) break;
    rounds.push(round);
  }
  return rounds;
}

export function commLatencyUs(
  plan: HaloPlan,
  model: CommModelOptions = DEFAULT_COMM_MODEL,
): { roundsCount: number; roundLatencyUs: number; totalUs: number; bytesPerRound: number[] } {
  const bytesPerRound = plan.rounds.map((round) => {
    let bytes = 0;
    for (const { src, dst } of round) {
      const list = plan.sendLists.get(`${src}>${dst}`);
      if (list) bytes += list.length * model.payloadBytes;
    }
    return bytes;
  });
  const bps = model.bandwidthGBs * 1e9;
  let total = 0;
  for (const b of bytesPerRound) total += model.latencyUs + (b / bps) * 1e6;
  return {
    roundsCount: plan.rounds.length,
    roundLatencyUs: plan.rounds.length ? total / plan.rounds.length : 0,
    totalUs: total,
    bytesPerRound,
  };
}
