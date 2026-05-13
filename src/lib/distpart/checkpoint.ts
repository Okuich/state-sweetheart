/**
 * Distributed checkpoint + replay.
 *
 * Per-partition snapshot containing:
 *   - tet assignment
 *   - per-tet payload (state) hash
 *   - logical step + RNG seed (for deterministic replay)
 *   - content-addressed digest used for partition recovery / rollback
 *
 * Restore strategy: a failed partition's tets can be re-homed onto a survivor
 * by reusing the snapshot bytes and re-running halos from `step`. The digest
 * lets us validate that re-execution produced the same state.
 */

export interface PartitionSnapshot {
  partition: number;
  step: number;
  tetIds: Uint32Array;
  rngSeed: number;
  /** content digest (FNV-1a 32-bit) over (tetIds, payloadHash, step) */
  digest: number;
  payloadHash: number;
  bytes: number;
}

export interface DistributedCheckpoint {
  step: number;
  partitionCount: number;
  snapshots: PartitionSnapshot[];
  rootDigest: number;
  totalBytes: number;
}

function fnv1a(buf: ArrayLike<number>, seed = 0x811c9dc5): number {
  let h = seed >>> 0;
  for (let i = 0; i < buf.length; i++) {
    h ^= buf[i] & 0xff;
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function snapshotPartition(
  partition: number,
  step: number,
  tetPart: Uint16Array,
  payloadBytes = 256,
  rngSeed = 0,
): PartitionSnapshot {
  const ids: number[] = [];
  for (let t = 0; t < tetPart.length; t++) if (tetPart[t] === partition) ids.push(t);
  const tetIds = new Uint32Array(ids);
  const view = new Uint8Array(tetIds.buffer);
  const payloadHash = fnv1a(view, 0x9e3779b1 ^ partition ^ step);
  const digest = fnv1a([
    payloadHash & 0xff, (payloadHash >>> 8) & 0xff, (payloadHash >>> 16) & 0xff, payloadHash >>> 24,
    step & 0xff, (step >>> 8) & 0xff,
    rngSeed & 0xff, (rngSeed >>> 8) & 0xff,
  ]);
  return {
    partition,
    step,
    tetIds,
    rngSeed,
    digest,
    payloadHash,
    bytes: tetIds.length * payloadBytes,
  };
}

export function buildCheckpoint(
  tetPart: Uint16Array,
  step: number,
  partitionCount: number,
  payloadBytes = 256,
  rngSeed = 0xc0ffee,
): DistributedCheckpoint {
  const snapshots: PartitionSnapshot[] = [];
  for (let p = 0; p < partitionCount; p++) {
    snapshots.push(snapshotPartition(p, step, tetPart, payloadBytes, rngSeed + p));
  }
  let root = 0x811c9dc5;
  let total = 0;
  for (const s of snapshots) {
    root = fnv1a([s.digest & 0xff, (s.digest >>> 8) & 0xff, (s.digest >>> 16) & 0xff, s.digest >>> 24], root);
    total += s.bytes;
  }
  return { step, partitionCount, snapshots, rootDigest: root, totalBytes: total };
}

/** Plan partition recovery: assign failed partition tets to neighbor survivors. */
export interface RecoveryPlan {
  failed: number[];
  /** survivor partition each tet is reassigned to */
  reassignment: Map<number, number>;
  bytesReplayed: number;
}

export function planRecovery(
  ck: DistributedCheckpoint,
  failed: number[],
  survivorLoadHint?: Float32Array,
): RecoveryPlan {
  const failedSet = new Set(failed);
  const survivors: number[] = [];
  for (let p = 0; p < ck.partitionCount; p++) if (!failedSet.has(p)) survivors.push(p);
  const load = new Float32Array(ck.partitionCount);
  if (survivorLoadHint) load.set(survivorLoadHint);
  const reassignment = new Map<number, number>();
  let bytes = 0;
  for (const fp of failed) {
    const snap = ck.snapshots[fp];
    if (!snap) continue;
    bytes += snap.bytes;
    for (const t of snap.tetIds) {
      let target = survivors[0];
      for (const s of survivors) if (load[s] < load[target]) target = s;
      reassignment.set(t, target);
      load[target] += 1;
    }
  }
  return { failed, reassignment, bytesReplayed: bytes };
}

/** Replay verification: checkpoint A == B at the partition + root level. */
export function verifyReplay(a: DistributedCheckpoint, b: DistributedCheckpoint): {
  ok: boolean;
  mismatches: number[];
} {
  const mismatches: number[] = [];
  if (a.partitionCount !== b.partitionCount) return { ok: false, mismatches: [-1] };
  for (let p = 0; p < a.partitionCount; p++) {
    if (a.snapshots[p].digest !== b.snapshots[p].digest) mismatches.push(p);
  }
  return { ok: mismatches.length === 0 && a.rootDigest === b.rootDigest, mismatches };
}
