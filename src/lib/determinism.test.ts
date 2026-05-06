import { describe, it, expect } from "vitest";
import {
  fnv1a64Hex,
  hashFloat64Array,
  deterministicReduce,
  makeRng,
  checkDeterminism,
} from "./determinism";
import type { StepKernel } from "./mpiOrchestrator";

describe("fnv1a64Hex", () => {
  it("produces a stable 16-hex digest for known inputs", () => {
    expect(fnv1a64Hex(new Uint8Array([]))).toBe("cbf29ce484222325");
    // FNV-1a 64 of "a" → e40c292c... (well-documented test vector)
    expect(fnv1a64Hex(new Uint8Array([0x61]))).toBe("af63dc4c8601ec8c");
  });

  it("changes when any single bit flips", () => {
    const a = new Uint8Array([1, 2, 3, 4]);
    const b = new Uint8Array([1, 2, 3, 5]);
    expect(fnv1a64Hex(a)).not.toBe(fnv1a64Hex(b));
  });

  it("hashes Float64Array bytes", () => {
    const a = new Float64Array([1.0, 2.0, 3.0]);
    const b = new Float64Array([1.0, 2.0, 3.0]);
    const c = new Float64Array([1.0, 2.0, 3.0000000000000004]); // ULP off
    expect(hashFloat64Array(a)).toBe(hashFloat64Array(b));
    expect(hashFloat64Array(a)).not.toBe(hashFloat64Array(c));
  });
});

describe("deterministicReduce", () => {
  it("sum is order-independent under Neumaier compensation", () => {
    // Classic catastrophic-cancellation case
    const xs = [1e16, 1, -1e16, 1];
    expect(deterministicReduce(xs, "sum")).toBe(2);
    // Reverse order produces the same bit-identical result
    const ys = [1, -1e16, 1, 1e16];
    expect(deterministicReduce(ys, "sum")).toBe(2);
  });

  it("min/max are exact left-folds", () => {
    expect(deterministicReduce([3, 1, 4, 1, 5], "min")).toBe(1);
    expect(deterministicReduce([3, 1, 4, 1, 5], "max")).toBe(5);
  });
});

describe("makeRng", () => {
  it("is reproducible for a given seed", () => {
    const a = makeRng(42);
    const b = makeRng(42);
    for (let i = 0; i < 100; i++) expect(a()).toBe(b());
  });
  it("differs for different seeds", () => {
    expect(makeRng(1)()).not.toBe(makeRng(2)());
  });
});

describe("checkDeterminism", () => {
  // A kernel whose state per rank is just (rank + step·rng()) — the snapshot
  // captures the full contribution vector so any RNG drift surfaces.
  const stateByRank: number[][] = [];
  const detKernelFactory = (rng: () => number): StepKernel => {
    return async ({ rank, range, step }) => {
      stateByRank[rank] ??= [];
      const r = rng();
      stateByRank[rank].push(rank + step * r);
      return {
        scalar: r,
        proposedDt: 0.01,
        haloOut: range.right >= 0 ? [{
          from: rank, to: range.right,
          indices: new Int32Array([range.end - 1]),
          payload: new Float32Array([r]),
        }] : [],
      };
    };
  };

  it("flags a deterministic kernel as deterministic", async () => {
    // Snapshot = packed scalars from the report (Neumaier-stable per step).
    const res = await checkDeterminism({
      nSteps: 8, size: 4, N: 16, dt0: 0.01,
      kernelFactory: detKernelFactory,
      snapshot: (_o, _s, rep) => {
        const out = new Float64Array(rep.scalars.length + 1);
        out.set(rep.scalars);
        out[rep.scalars.length] = rep.sum;
        return out;
      },
    });
    expect(res.deterministic).toBe(true);
    expect(res.firstDivergentStep).toBe(-1);
    expect(res.steps).toBe(8);
    // Every digest is non-empty and 16 hex chars wide
    expect(res.digestsA.every((d) => /^[0-9a-f]{16}$/.test(d))).toBe(true);
  });

  it("DETECTS non-determinism when kernel uses Math.random()", async () => {
    const nondet: (rng: () => number) => StepKernel = () => async ({ rank, range }) => ({
      scalar: Math.random(), // intentionally non-deterministic
      proposedDt: 0.01,
      haloOut: range.right >= 0 ? [{
        from: rank, to: range.right,
        indices: new Int32Array([range.end - 1]),
        payload: new Float32Array([Math.random()]),
      }] : [],
    });
    const res = await checkDeterminism({
      nSteps: 4, size: 3, N: 9, dt0: 0.01,
      kernelFactory: nondet,
      snapshot: (_o, _s, rep) => {
        const out = new Float64Array(rep.scalars);
        return out;
      },
    });
    expect(res.deterministic).toBe(false);
    expect(res.firstDivergentStep).toBeGreaterThanOrEqual(0);
    expect(res.divergence).not.toBeNull();
    expect(res.divergence!.a).not.toBe(res.divergence!.b);
  });

  it("digests are bit-stable: identical kernels → identical digest sequences", async () => {
    const run = () => checkDeterminism({
      nSteps: 5, size: 2, N: 6, dt0: 0.005,
      kernelFactory: detKernelFactory,
      snapshot: (_o, _s, rep) => new Float64Array(rep.scalars),
    });
    const r1 = await run();
    const r2 = await run();
    expect(r1.digestsA).toEqual(r2.digestsA);
    expect(r1.digestsA).toEqual(r2.digestsB);
  });
});
