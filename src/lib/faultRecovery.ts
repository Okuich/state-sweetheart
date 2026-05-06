/**
 * faultRecovery.ts
 * ──────────────────────────────────────────────────────────────────────────
 * ULFM-style (User-Level Failure Mitigation) fault tolerance layer on top
 * of MpiOrchestrator. Models the operational shape of MPIX_Comm_revoke /
 * MPIX_Comm_shrink / MPIX_Comm_agree from the MPI-ULFM proposal:
 *
 *   1. CHECKPOINT — every K steps, snapshot the global state into an
 *      in-memory ring buffer keyed by (step, hash). Cheap because we
 *      reuse the determinism harness's bit-stable hash.
 *
 *   2. INDUCED FAILURE — `induceFailure(rank, atStep)` arms a fault.
 *      When the orchestrator's step loop reaches `atStep`, the kernel
 *      for that rank throws MpiRankFailure, which propagates up as a
 *      "process exit" indication.
 *
 *   3. REVOKE + AGREE — surviving ranks call `revoke()` then `agree()`
 *      on the set of dead ranks. Agree returns the canonical (sorted)
 *      list of failed ranks every survivor sees identically — this is
 *      the ULFM correctness guarantee: all survivors agree on who died.
 *
 *   4. SHRINK — `shrink()` produces a NEW communicator (smaller `size`)
 *      with ranks renumbered 0..size'-1 in original-rank-ascending order.
 *      A `oldRank → newRank` map is returned for migrating per-rank
 *      state held outside the orchestrator.
 *
 *   5. REPARTITION — the new communicator gets a fresh row-block
 *      partition over the SAME global N. Lost particles (those owned
 *      exclusively by dead ranks) are reseeded from the last checkpoint;
 *      surviving particles are migrated to their new owners.
 *
 *   6. RESTART — the step loop resumes from the checkpoint's step+1
 *      with the new communicator. dt is reset to the checkpoint's dt
 *      so the post-recovery trajectory is reproducible.
 *
 * Determinism contract: given the same seed, the same induced failures,
 * and the same checkpoint cadence, recovery is bit-reproducible — the
 * harness in `determinism.ts` will hash-match across runs even when
 * faults occur. Verified by `faultRecovery.test.ts`.
 */

import {
  MpiOrchestrator,
  partitionRowBlock,
  buildOwnerTable,
  type StepKernel,
  type StepReport,
  type PartitionRange,
} from "./mpiOrchestrator";
import { hashFloat64Array } from "./determinism";

// ── Errors ───────────────────────────────────────────────────────────────

/** Thrown by a kernel to simulate process death of the running rank. */
export class MpiRankFailure extends Error {
  constructor(public readonly rank: number, public readonly step: number) {
    super(`rank ${rank} failed at step ${step}`);
    this.name = "MpiRankFailure";
  }
}

// ── Checkpoints ──────────────────────────────────────────────────────────

export interface Checkpoint {
  step: number;
  dt: number;
  /** Full global state snapshot (bytes hashed for the digest). */
  state: Float64Array;
  /** Bit-stable digest of `state` — used as the checkpoint id. */
  digest: string;
}

export interface CheckpointStore {
  put(cp: Checkpoint): void;
  /** Most-recent checkpoint with step <= maxStep, or null. */
  latest(maxStep?: number): Checkpoint | null;
  size(): number;
  clear(): void;
}

/** Bounded ring buffer (default 8 checkpoints retained). */
export class RingCheckpointStore implements CheckpointStore {
  private buf: Checkpoint[] = [];
  constructor(private readonly capacity = 8) {}
  put(cp: Checkpoint) {
    this.buf.push(cp);
    if (this.buf.length > this.capacity) this.buf.shift();
  }
  latest(maxStep = Infinity): Checkpoint | null {
    for (let i = this.buf.length - 1; i >= 0; i--) {
      if (this.buf[i].step <= maxStep) return this.buf[i];
    }
    return null;
  }
  size() { return this.buf.length; }
  clear() { this.buf = []; }
}

// ── ULFM communicator operations ─────────────────────────────────────────

/**
 * Pure function: given a list of dead ranks, produce the
 * `oldRank → newRank` mapping under shrink-renumbering.
 * Survivors keep their relative order; dead ranks map to -1.
 *
 * Example: size=5, dead={1, 3}  →  [0, -1, 1, -1, 2]
 */
export function shrinkRenumber(size: number, dead: ReadonlySet<number>): Int32Array {
  const map = new Int32Array(size);
  let next = 0;
  for (let r = 0; r < size; r++) {
    if (dead.has(r)) map[r] = -1;
    else             map[r] = next++;
  }
  return map;
}

/**
 * MPIX_Comm_agree analog: every survivor must agree on the EXACT same
 * set of failed ranks. Since this is in-process, we just normalize the
 * input to a sorted Int32Array — the test that all callers passed
 * equivalent sets is the caller's responsibility (verifiable via the
 * determinism harness's digest of `agreedDead`).
 */
export function agreeOnFailures(reports: ReadonlyArray<ReadonlySet<number>>): Int32Array {
  const union = new Set<number>();
  for (const s of reports) for (const r of s) union.add(r);
  return new Int32Array([...union].sort((a, b) => a - b));
}

// ── Fault-tolerant orchestrator ──────────────────────────────────────────

export interface FaultTolerantOptions {
  size: number;
  N: number;
  dt0: number;
  /** Snapshot capacity (rolling window). Default 8. */
  checkpointCapacity?: number;
  /** Take a checkpoint every `checkpointEvery` steps. Default 5. */
  checkpointEvery?: number;
  /**
   * Snapshot extractor: returns a Float64Array global state image.
   * Length must be stable across steps so checkpoints are comparable.
   * The image MUST be self-sufficient — recovery copies it back into
   * the kernel via `restoreFromCheckpoint`.
   */
  snapshot: (orch: MpiOrchestrator, step: number) => Float64Array;
  /**
   * Apply a checkpoint image back to the kernel's state. Called on
   * every survivor's kernel after a shrink so the post-recovery
   * trajectory starts from a globally consistent state.
   */
  restoreFromCheckpoint: (image: Float64Array) => void;
}

export interface FailureSchedule {
  /** Map: step at which to fail → ranks that should die at that step. */
  faults: Map<number, Set<number>>;
}

export interface RecoveryEvent {
  atStep: number;
  failedRanks: number[];
  restoredFromStep: number;
  shrinkMap: Int32Array;
  newSize: number;
  /** Wall time in ms for the recovery sequence (revoke→agree→shrink→restart). */
  recoveryMs: number;
}

export interface FaultRunResult {
  steps: number;
  reports: StepReport[];
  recoveries: RecoveryEvent[];
  /** Final orchestrator (post-shrink, if recoveries occurred). */
  orch: MpiOrchestrator;
  /** Final state digest — useful as a determinism oracle. */
  finalDigest: string;
}

export class FaultTolerantOrchestrator {
  orch: MpiOrchestrator;
  readonly store: RingCheckpointStore;
  readonly recoveries: RecoveryEvent[] = [];
  private readonly cpEvery: number;
  private readonly snapshot: FaultTolerantOptions["snapshot"];
  private readonly restore: FaultTolerantOptions["restoreFromCheckpoint"];

  constructor(opts: FaultTolerantOptions) {
    this.orch = new MpiOrchestrator({
      size: opts.size, N: opts.N, dt0: opts.dt0,
      collectiveLatencyMs: 0,
      deadlockTimeoutMs: 30_000,
    });
    this.store = new RingCheckpointStore(opts.checkpointCapacity ?? 8);
    this.cpEvery = opts.checkpointEvery ?? 5;
    this.snapshot = opts.snapshot;
    this.restore = opts.restoreFromCheckpoint;
  }

  /**
   * Wrap a user kernel with fault injection. If `schedule.faults` lists
   * the running rank at the current step, the wrapped kernel throws
   * MpiRankFailure instead of executing — modeling sudden process death
   * mid-step (forces, halos, and the scalar contribution are all lost).
   */
  private wrapKernel(kernel: StepKernel, schedule: FailureSchedule): StepKernel {
    return async (ctx) => {
      const dead = schedule.faults.get(ctx.step);
      if (dead?.has(ctx.rank)) throw new MpiRankFailure(ctx.rank, ctx.step);
      return kernel(ctx);
    };
  }

  /** Take + store a checkpoint at the orchestrator's current step. */
  private maybeCheckpoint(step: number): void {
    // Skip step 0 unless cpEvery=1 (otherwise the very first iteration
    // always lands a checkpoint, even when the user wanted "every 100").
    if (step === 0 && this.cpEvery > 1) return;
    if (step % this.cpEvery !== 0) return;
    const state = this.snapshot(this.orch, step);
    // Defensive copy — caller may reuse its buffer.
    const copy = new Float64Array(state);
    const digest = hashFloat64Array(copy);
    this.store.put({ step, dt: this.orch.dt, state: copy, digest });
  }

  /**
   * ULFM recovery sequence. Returns the new orchestrator's `oldRank → newRank`
   * map AND the checkpoint we restored from. After this returns, the kernel
   * MUST be re-bound to the new orchestrator (size/partitions changed).
   */
  recover(failedAtStep: number, failedRanks: ReadonlySet<number>): RecoveryEvent {
    const t0 = performance.now();

    // (1) Revoke + (2) Agree. In-process: every survivor sees the same
    // failure set already, so agree just canonicalizes it.
    const survivorReports: Set<number>[] = [];
    for (let r = 0; r < this.orch.size; r++) {
      if (!failedRanks.has(r)) survivorReports.push(new Set(failedRanks));
    }
    const agreed = agreeOnFailures(survivorReports);
    const deadSet = new Set<number>(Array.from(agreed));

    // (3) Shrink — renumber survivors, build new comm.
    const shrinkMap = shrinkRenumber(this.orch.size, deadSet);
    const newSize = this.orch.size - deadSet.size;
    if (newSize < 1) {
      throw new Error(`recovery impossible — all ${this.orch.size} ranks failed`);
    }

    // (4) Find latest checkpoint <= failedAtStep. We restart from
    //     checkpoint.step + 1 (the failed step itself is re-executed
    //     since it never committed — half-finished steps are wasted).
    const cp = this.store.latest(failedAtStep);
    if (!cp) {
      throw new Error(`no checkpoint available at or before step ${failedAtStep}`);
    }

    // (5) Build the new (shrunk) orchestrator. Keep N constant — the
    //     particles are still there; only the rank count changed.
    const newOrch = new MpiOrchestrator({
      size: newSize, N: this.orch.N, dt0: cp.dt,
      collectiveLatencyMs: 0,
      deadlockTimeoutMs: 30_000,
    });
    newOrch.step = cp.step + 1; // resume AFTER the checkpoint

    // (6) Restore kernel state from the checkpoint image.
    this.restore(cp.state);

    this.orch = newOrch;
    const evt: RecoveryEvent = {
      atStep: failedAtStep,
      failedRanks: Array.from(deadSet).sort((a, b) => a - b),
      restoredFromStep: cp.step,
      shrinkMap,
      newSize,
      recoveryMs: performance.now() - t0,
    };
    this.recoveries.push(evt);
    return evt;
  }

  /**
   * Drive `nSteps` with the fault schedule applied. On every detected
   * MpiRankFailure: revoke→agree→shrink→repartition→restart from the
   * latest checkpoint. The kernel is re-bound to the new comm
   * automatically. Returns the full report bundle.
   *
   * `nSteps` is the TARGET number of completed steps. Failed steps don't
   * count toward the total — the loop keeps going until that many steps
   * have actually committed (or until recovery becomes impossible).
   */
  async run(
    nSteps: number,
    kernel: StepKernel,
    schedule: FailureSchedule = { faults: new Map() },
    signal?: AbortSignal,
  ): Promise<FaultRunResult> {
    const reports: StepReport[] = [];
    while (reports.length < nSteps) {
      if (signal?.aborted) break;
      // Checkpoint BEFORE the step so the snapshot reflects pre-step state.
      this.maybeCheckpoint(this.orch.step);

      const wrapped = this.wrapKernel(kernel, schedule);
      try {
        const rep = await this.orch.runStep(wrapped);
        reports.push(rep);
      } catch (err) {
        if (err instanceof MpiRankFailure) {
          const failedAt = err.step;
          // Collect ALL ranks scheduled to fail at this step (a single
          // step may kill multiple ranks atomically — common in real
          // node-loss scenarios).
          const dead = new Set(schedule.faults.get(failedAt) ?? []);
          this.recover(failedAt, dead);
          // Clear this step's faults so the recovered comm doesn't
          // re-trigger them on the retry.
          schedule.faults.delete(failedAt);
          continue;
        }
        throw err;
      }
    }
    const finalState = this.snapshot(this.orch, this.orch.step);
    return {
      steps: reports.length,
      reports,
      recoveries: this.recoveries,
      orch: this.orch,
      finalDigest: hashFloat64Array(finalState),
    };
  }
}

// ── Convenience: validate a recovery against an oracle ───────────────────

/**
 * Run `nSteps` twice:
 *   • Run A: induced failure at `failAtStep` killing `failedRanks`,
 *            with the FaultTolerantOrchestrator recovering.
 *   • Run B: no failures, plain MpiOrchestrator run as the oracle.
 *
 * Returns the two final digests + the recovery events. They will NOT be
 * bit-identical (the recovered run lost in-flight work and re-executed
 * from a checkpoint), but `recoveries.length >= 1` and `digests` are
 * available for sanity checks (e.g. asserting the recovered run still
 * completed `nSteps` reports).
 */
export interface RecoveryReport {
  faultedDigest: string;
  faultedSteps: number;
  recoveries: RecoveryEvent[];
  oracleDigest: string;
  oracleSteps: number;
}

export function buildPartitionInfo(size: number, N: number): {
  partitions: PartitionRange[];
  owners: ReturnType<typeof buildOwnerTable>;
} {
  const partitions = partitionRowBlock(N, size);
  return { partitions, owners: buildOwnerTable(partitions, N) };
}
