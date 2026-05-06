import { describe, it, expect } from "vitest";
import {
  RingTelemetry,
  compare,
  endPhase,
  recordSample,
  startPhase,
  type BroadphaseCounters,
  type PhaseTiming,
} from "./broadphaseTelemetry";

function fakeTiming(phase: PhaseTiming["phase"], wallMs: number, gpuMs = NaN): PhaseTiming {
  return { phase, wallMs, gpuMs };
}

const ZERO_COUNTERS: BroadphaseCounters = { N: 0, candidates: 0, emitted: 0, unique: 0, dropped: 0 };

describe("startPhase / endPhase", () => {
  it("produces a positive wall ms", async () => {
    const h = startPhase("build_struct");
    await new Promise(r => setTimeout(r, 5));
    const t = endPhase(h);
    expect(t.phase).toBe("build_struct");
    expect(t.wallMs).toBeGreaterThan(0);
    expect(Number.isNaN(t.gpuMs)).toBe(true);
  });

  it("uses the GPU readback when supplied", () => {
    const h = startPhase("pair_emit");
    const t = endPhase(h, { gpu: { elapsedNs: 2_500_000 } });
    expect(t.gpuMs).toBeCloseTo(2.5, 6);
  });
});

describe("recordSample derived rates", () => {
  it("computes pairs/ms, candidates/ms, selectivity, dedup, saturation", () => {
    const phases = [
      fakeTiming("build_struct", 1.0),
      fakeTiming("build_query",  2.0),
      fakeTiming("pair_emit",    4.0),
      fakeTiming("readback",     1.0),
    ];
    const counters: BroadphaseCounters = {
      N: 1000, candidates: 10_000, emitted: 1_000, unique: 950, dropped: 50,
    };
    const s = recordSample("spatial-hash", 7, phases, counters);
    expect(s.rates.totalWallMs).toBeCloseTo(8, 6);
    expect(s.rates.pairsEmittedPerMs).toBeCloseTo(125, 6);
    expect(s.rates.candidatesPerMs).toBeCloseTo(1250, 6);
    expect(s.rates.emissionRatio).toBeCloseTo(0.1, 6);
    expect(s.rates.dedupRatio).toBeCloseTo(0.95, 6);
    expect(s.rates.saturationRatio).toBeCloseTo(50 / 1050, 6);
    expect(s.gpuTimingAvailable).toBe(false);
  });

  it("gpuTimingAvailable=true when any phase has a GPU reading", () => {
    const s = recordSample("lbvh", 0,
      [fakeTiming("build_struct", 1, 0.5), fakeTiming("pair_emit", 1)],
      { ...ZERO_COUNTERS, candidates: 100, emitted: 10, unique: 10 });
    expect(s.gpuTimingAvailable).toBe(true);
    expect(s.rates.totalGpuMs).toBeCloseTo(0.5, 6);
  });

  it("zero-division uses safe fallbacks (1.0 dedup, 0 elsewhere)", () => {
    const s = recordSample("lbvh", 0, [fakeTiming("pair_emit", 0)], ZERO_COUNTERS);
    expect(s.rates.dedupRatio).toBe(1);
    expect(s.rates.emissionRatio).toBe(0);
    expect(s.rates.saturationRatio).toBe(0);
    expect(s.rates.pairsEmittedPerMs).toBe(0);
  });
});

describe("RingTelemetry", () => {
  it("retains only the last windowSize samples per strategy", () => {
    const ring = new RingTelemetry({ windowSize: 3 });
    for (let i = 0; i < 5; i++) {
      ring.push(recordSample("spatial-hash", i,
        [fakeTiming("pair_emit", 1)],
        { ...ZERO_COUNTERS, candidates: 10, emitted: i + 1, unique: i + 1 }));
    }
    const arr = ring.toArray("spatial-hash");
    expect(arr.length).toBe(3);
    expect(arr.map(s => s.tick)).toEqual([2, 3, 4]);
    expect(ring.latest("spatial-hash")?.tick).toBe(4);
  });

  it("aggregate averages rates and tolerates a missing GPU stream", () => {
    const ring = new RingTelemetry();
    ring.push(recordSample("lbvh", 0,
      [fakeTiming("pair_emit", 2, 1)], { ...ZERO_COUNTERS, candidates: 100, emitted: 100, unique: 100 }));
    ring.push(recordSample("lbvh", 1,
      [fakeTiming("pair_emit", 4)],    { ...ZERO_COUNTERS, candidates: 100, emitted: 100, unique: 100 }));
    const agg = ring.aggregate("lbvh")!;
    expect(agg.totalWallMs).toBeCloseTo(3, 6);
    expect(agg.totalGpuMs).toBeCloseTo(1, 6); // averaged only over GPU sample(s)
    expect(agg.dedupRatio).toBeCloseTo(1, 6);
  });

  it("reset clears both strategy buckets", () => {
    const ring = new RingTelemetry();
    ring.push(recordSample("spatial-hash", 0, [fakeTiming("pair_emit", 1)], ZERO_COUNTERS));
    ring.push(recordSample("lbvh", 0,         [fakeTiming("pair_emit", 1)], ZERO_COUNTERS));
    ring.reset();
    expect(ring.latest("spatial-hash")).toBeUndefined();
    expect(ring.latest("lbvh")).toBeUndefined();
  });
});

describe("compare()", () => {
  const phases = [fakeTiming("pair_emit", 4)];
  const hashS = recordSample("spatial-hash", 0, phases,
    { N: 100, candidates: 1000, emitted: 200, unique: 200, dropped: 0 });
  const lbvhS = recordSample("lbvh", 0, [fakeTiming("pair_emit", 8)],
    { N: 100, candidates: 400, emitted: 200, unique: 200, dropped: 0 });

  it("flags lower-is-better wall-ms for the faster strategy", () => {
    const rows = compare(hashS.rates, lbvhS.rates);
    const wall = rows.find(r => r.metric === "wall ms")!;
    expect(wall.winner).toBe("spatial-hash"); // 4 ms < 8 ms
  });

  it("flags higher-is-better selectivity for the more precise strategy", () => {
    const rows = compare(hashS.rates, lbvhS.rates);
    const sel = rows.find(r => r.metric === "selectivity")!;
    expect(sel.winner).toBe("lbvh"); // 0.5 > 0.2
  });

  it("declares a tie when within 1 percent", () => {
    const a = recordSample("spatial-hash", 0, [fakeTiming("pair_emit", 1.000)], ZERO_COUNTERS).rates;
    const b = recordSample("lbvh",        0, [fakeTiming("pair_emit", 1.005)], ZERO_COUNTERS).rates;
    const wall = compare(a, b).find(r => r.metric === "wall ms")!;
    expect(wall.winner).toBe("tie");
  });
});
