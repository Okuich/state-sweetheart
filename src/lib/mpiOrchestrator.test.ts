import { describe, it, expect } from "vitest";
import {
  MpiOrchestrator,
  partitionRowBlock,
  buildOwnerTable,
  reduce,
  type StepKernel,
  type HaloPacket,
} from "./mpiOrchestrator";

describe("partitionRowBlock", () => {
  it("distributes remainder to lowest ranks and covers [0, N)", () => {
    const parts = partitionRowBlock(10, 3); // 4, 3, 3
    expect(parts.map((p) => p.count)).toEqual([4, 3, 3]);
    expect(parts[0].start).toBe(0);
    expect(parts[2].end).toBe(10);
    // Coverage + non-overlap
    let cur = 0;
    for (const p of parts) { expect(p.start).toBe(cur); cur = p.end; }
    expect(cur).toBe(10);
  });

  it("sets neighbor ranks correctly (no wrap)", () => {
    const parts = partitionRowBlock(8, 4);
    expect(parts[0].left).toBe(-1);
    expect(parts[0].right).toBe(1);
    expect(parts[3].left).toBe(2);
    expect(parts[3].right).toBe(-1);
  });

  it("handles size > N (some ranks get count=0)", () => {
    const parts = partitionRowBlock(2, 5);
    expect(parts.map((p) => p.count)).toEqual([1, 1, 0, 0, 0]);
  });
});

describe("buildOwnerTable + reduce", () => {
  it("owners[i] == rank that owns i", () => {
    const parts = partitionRowBlock(7, 3); // 3,2,2
    const owners = buildOwnerTable(parts, 7);
    expect(Array.from(owners)).toEqual([0, 0, 0, 1, 1, 2, 2]);
  });

  it("reduce sum/min/max", () => {
    expect(reduce([1, 2, 3, 4], "sum")).toBe(10);
    expect(reduce([5, -1, 3], "min")).toBe(-1);
    expect(reduce([5, -1, 3], "max")).toBe(5);
  });
});

describe("MpiOrchestrator.runStep", () => {
  const makeKernel = (proposedDt: (rank: number) => number, scalar: (rank: number) => number): StepKernel =>
    async ({ rank, range }) => ({
      scalar: scalar(rank),
      proposedDt: proposedDt(rank),
      haloOut: [
        // Each rank sends 2 floats to its right neighbor (if any)
        ...(range.right >= 0 ? [{
          from: rank, to: range.right,
          indices: new Int32Array([range.end - 1]),
          payload: new Float32Array([rank * 1.0, rank * 2.0]),
        } as HaloPacket] : []),
      ],
    });

  it("agrees on min(dt) across ranks for the NEXT step", async () => {
    const orch = new MpiOrchestrator({ size: 4, N: 16, dt0: 0.01 });
    // Rank 2 proposes the smallest dt.
    const k = makeKernel((r) => (r === 2 ? 0.001 : 0.01), () => 1);
    await orch.runStep(k);
    expect(orch.dt).toBeCloseTo(0.001, 12);
  });

  it("sums per-rank scalars in StepReport.sum", async () => {
    const orch = new MpiOrchestrator({ size: 5, N: 20, dt0: 0.01 });
    const k = makeKernel(() => 0.01, (r) => r + 1); // 1+2+3+4+5 = 15
    const rep = await orch.runStep(k);
    expect(rep.sum).toBe(15);
    expect(Array.from(rep.scalars)).toEqual([1, 2, 3, 4, 5]);
  });

  it("delivers halos to the destination rank's inbox on the NEXT step", async () => {
    const orch = new MpiOrchestrator({ size: 3, N: 9, dt0: 0.01 });
    const seenInboxSizes: number[][] = [];
    const k: StepKernel = async ({ rank, range, step }) => {
      // On step 1, record what we received from step 0.
      if (step === 1) {
        seenInboxSizes[rank] = orch["inbox"][rank].map((p) => p.indices.length);
      }
      return {
        scalar: 0,
        proposedDt: 0.01,
        haloOut: range.right >= 0 ? [{
          from: rank, to: range.right,
          indices: new Int32Array([range.end - 1]),
          payload: new Float32Array([1, 2]),
        }] : [],
      };
    };
    await orch.runStep(k);
    await orch.runStep(k);
    // Ranks 1 and 2 should have received 1 packet from their left neighbor.
    expect(seenInboxSizes[0]).toEqual([]);
    expect(seenInboxSizes[1]).toEqual([1]);
    expect(seenInboxSizes[2]).toEqual([1]);
  });

  it("clamps dt to [dtMin, dtMax]", async () => {
    const orch = new MpiOrchestrator({
      size: 2, N: 4, dt0: 0.01, dtMin: 0.005, dtMax: 0.02,
    });
    await orch.runStep(makeKernel(() => 1e-9, () => 0)); // would be < dtMin
    expect(orch.dt).toBe(0.005);
    await orch.runStep(makeKernel(() => 1e3, () => 0));  // would be > dtMax
    expect(orch.dt).toBe(0.02);
  });

  it("repartition() rebuilds owners and clears inbox", () => {
    const orch = new MpiOrchestrator({ size: 3, N: 9, dt0: 0.01 });
    orch.repartition(12);
    expect(orch.N).toBe(12);
    expect(orch.partitions.reduce((a, p) => a + p.count, 0)).toBe(12);
    expect(orch.owners.length).toBe(12);
  });
});
