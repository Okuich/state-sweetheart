import { describe, it, expect } from "vitest";
import {
  FaultTolerantOrchestrator,
  RingCheckpointStore,
  shrinkRenumber,
  agreeOnFailures,
  MpiRankFailure,
  type FailureSchedule,
} from "./faultRecovery";
import { MpiOrchestrator } from "./mpiOrchestrator";
import { hashFloat64Array, makeRng } from "./determinism";
import type { StepKernel } from "./mpiOrchestrator";

describe("shrinkRenumber", () => {
  it("renumbers survivors in original order; dead → -1", () => {
    expect(Array.from(shrinkRenumber(5, new Set([1, 3])))).toEqual([0, -1, 1, -1, 2]);
    expect(Array.from(shrinkRenumber(4, new Set()))).toEqual([0, 1, 2, 3]);
    expect(Array.from(shrinkRenumber(3, new Set([0])))).toEqual([-1, 0, 1]);
  });
});

describe("agreeOnFailures", () => {
  it("returns the canonical sorted union of all reports", () => {
    const out = agreeOnFailures([
      new Set([2, 0]),
      new Set([0, 4]),
      new Set([2]),
    ]);
    expect(Array.from(out)).toEqual([0, 2, 4]);
  });

  it("idempotent on identical reports (ULFM agreement)", () => {
    const a = agreeOnFailures([new Set([1, 3]), new Set([1, 3]), new Set([1, 3])]);
    const b = agreeOnFailures([new Set([3, 1])]);
    expect(Array.from(a)).toEqual(Array.from(b));
  });
});

describe("RingCheckpointStore", () => {
  it("retains the most-recent N checkpoints", () => {
    const s = new RingCheckpointStore(3);
    for (let i = 0; i < 5; i++) {
      s.put({ step: i, dt: 0.01, state: new Float64Array([i]), digest: String(i) });
    }
    expect(s.size()).toBe(3);
    expect(s.latest()!.step).toBe(4);
    expect(s.latest(2)!.step).toBe(2);
  });
});

// ── End-to-end fault recovery ────────────────────────────────────────────

/**
 * Build a kernel + state-management harness that:
 *   - holds a shared global Float64Array of length N (positions, scalar)
 *   - each step: every rank advances its owned slice by `rank * dt`
 *   - snapshot/restore round-trip the full N-vector
 */
function makeKernelHarness(N: number) {
  const state = new Float64Array(N);
  const kernel: StepKernel = async ({ rank, range, dt }) => {
    for (let i = range.start; i < range.end; i++) state[i] += rank * dt + 0.001;
    return {
      scalar: range.count,
      proposedDt: dt,
      haloOut: range.right >= 0 ? [{
        from: rank, to: range.right,
        indices: new Int32Array([range.end - 1]),
        payload: new Float32Array([state[range.end - 1]]),
      }] : [],
    };
  };
  const snapshot = () => new Float64Array(state);
  const restore  = (img: Float64Array) => state.set(img);
  return { state, kernel, snapshot, restore };
}

describe("FaultTolerantOrchestrator", () => {
  it("runs to completion when no faults are scheduled", async () => {
    const { kernel, snapshot, restore } = makeKernelHarness(12);
    const ft = new FaultTolerantOrchestrator({
      size: 4, N: 12, dt0: 0.01, checkpointEvery: 3,
      snapshot, restoreFromCheckpoint: restore,
    });
    const res = await ft.run(10, kernel);
    expect(res.steps).toBe(10);
    expect(res.recoveries).toHaveLength(0);
    expect(ft.orch.size).toBe(4);
  });

  it("recovers from a single induced rank failure", async () => {
    const { kernel, snapshot, restore } = makeKernelHarness(12);
    const ft = new FaultTolerantOrchestrator({
      size: 4, N: 12, dt0: 0.01, checkpointEvery: 2,
      snapshot, restoreFromCheckpoint: restore,
    });
    const schedule: FailureSchedule = {
      faults: new Map([[5, new Set([2])]]), // rank 2 dies at step 5
    };
    const res = await ft.run(10, kernel, schedule);

    // Recovered: comm shrank by 1, exactly one recovery event recorded.
    expect(res.recoveries).toHaveLength(1);
    const evt = res.recoveries[0];
    expect(evt.failedRanks).toEqual([2]);
    expect(evt.newSize).toBe(3);
    expect(ft.orch.size).toBe(3);
    // Restored from a checkpoint at or before the failed step.
    expect(evt.restoredFromStep).toBeLessThanOrEqual(5);
    expect(evt.restoredFromStep % 2).toBe(0); // cpEvery=2 → checkpoints at even steps
    // Shrink map: rank 2 → -1, others renumbered contiguously.
    expect(Array.from(evt.shrinkMap)).toEqual([0, -1, 1, 2]);
    // Still completed the requested step count.
    expect(res.steps).toBe(10);
    // Final digest is non-empty + well-formed.
    expect(res.finalDigest).toMatch(/^[0-9a-f]{16}$/);
  });

  it("recovers from MULTIPLE concurrent failures in one step", async () => {
    const { kernel, snapshot, restore } = makeKernelHarness(20);
    const ft = new FaultTolerantOrchestrator({
      size: 5, N: 20, dt0: 0.01, checkpointEvery: 4,
      snapshot, restoreFromCheckpoint: restore,
    });
    const schedule: FailureSchedule = {
      faults: new Map([[7, new Set([1, 3])]]), // ranks 1 + 3 die together
    };
    const res = await ft.run(12, kernel, schedule);
    expect(res.recoveries).toHaveLength(1);
    expect(res.recoveries[0].failedRanks).toEqual([1, 3]);
    expect(res.recoveries[0].newSize).toBe(3);
    expect(ft.orch.size).toBe(3);
    // Partition still covers all N=20 particles after shrink.
    const totalCount = ft.orch.partitions.reduce((a, p) => a + p.count, 0);
    expect(totalCount).toBe(20);
  });

  it("recovery is deterministic across runs (bit-identical final digest)", async () => {
    const runOnce = async () => {
      const { kernel, snapshot, restore } = makeKernelHarness(16);
      const ft = new FaultTolerantOrchestrator({
        size: 4, N: 16, dt0: 0.01, checkpointEvery: 3,
        snapshot, restoreFromCheckpoint: restore,
      });
      const schedule: FailureSchedule = {
        faults: new Map([[6, new Set([1])]]),
      };
      return ft.run(10, kernel, schedule);
    };
    const a = await runOnce();
    const b = await runOnce();
    expect(a.finalDigest).toBe(b.finalDigest);
    expect(a.steps).toBe(b.steps);
    expect(a.recoveries[0].restoredFromStep).toBe(b.recoveries[0].restoredFromStep);
  });

  it("throws when no checkpoint is available before the failure", async () => {
    const { kernel, snapshot, restore } = makeKernelHarness(8);
    const ft = new FaultTolerantOrchestrator({
      size: 4, N: 8, dt0: 0.01, checkpointEvery: 100, // never checkpoints in time
      snapshot, restoreFromCheckpoint: restore,
    });
    // First checkpoint happens at step 0 → recovery from step >= 0 is fine.
    // Force the situation by clearing the store right before the fault.
    const schedule: FailureSchedule = {
      faults: new Map([[3, new Set([0])]]),
    };
    ft.store.clear();
    await expect(ft.run(5, kernel, schedule)).rejects.toThrow(/no checkpoint/i);
  });

  it("throws when ALL ranks fail simultaneously (recovery impossible)", async () => {
    const { kernel, snapshot, restore } = makeKernelHarness(8);
    const ft = new FaultTolerantOrchestrator({
      size: 2, N: 8, dt0: 0.01, checkpointEvery: 1,
      snapshot, restoreFromCheckpoint: restore,
    });
    const schedule: FailureSchedule = {
      faults: new Map([[2, new Set([0, 1])]]),
    };
    await expect(ft.run(5, kernel, schedule)).rejects.toThrow(/recovery impossible/);
  });

  it("MpiRankFailure carries rank + step", () => {
    const e = new MpiRankFailure(3, 7);
    expect(e.rank).toBe(3);
    expect(e.step).toBe(7);
    expect(e.message).toMatch(/rank 3.*step 7/);
  });
});
