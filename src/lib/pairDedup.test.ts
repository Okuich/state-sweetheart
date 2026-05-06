import { describe, it, expect } from "vitest";
import { dedupPairs, dedupPacked, PairDedupStream } from "./pairDedup";

describe("dedupPairs — basic correctness", () => {
  it("removes exact duplicates", () => {
    const r = dedupPairs([
      { i: 0, j: 1 }, { i: 2, j: 3 }, { i: 0, j: 1 },
    ], { N: 4, maxPairs: 16 });
    expect(r.count).toBe(2);
    expect(r.overflow.duplicates).toBe(1);
    expect(Array.from(r.pairs)).toEqual([0, 1, 2, 3]);
  });

  it("normalizes (i,j) vs (j,i) — unordered uniqueness", () => {
    const r = dedupPairs([
      { i: 5, j: 2 }, { i: 2, j: 5 }, { i: 5, j: 2 },
    ], { N: 8, maxPairs: 16 });
    expect(r.count).toBe(1);
    expect(Array.from(r.pairs)).toEqual([2, 5]);
    expect(r.overflow.duplicates).toBe(2);
  });

  it("drops self-pairs and out-of-range indices", () => {
    const r = dedupPairs([
      { i: 1, j: 1 },         // self
      { i: -1, j: 0 },        // negative
      { i: 0, j: 99 },        // OOR (N=4)
      { i: 0, j: 1 },         // valid
      { i: 99, j: 99 },       // both OOR
    ], { N: 4, maxPairs: 16 });
    expect(r.count).toBe(1);
    expect(r.overflow.invalid).toBe(4);
    expect(Array.from(r.pairs)).toEqual([0, 1]);
  });

  it("output is canonically sorted regardless of input order", () => {
    const a = dedupPairs([
      { i: 7, j: 3 }, { i: 0, j: 1 }, { i: 5, j: 2 },
    ], { N: 8, maxPairs: 16 });
    const b = dedupPairs([
      { i: 5, j: 2 }, { i: 7, j: 3 }, { i: 0, j: 1 },
    ], { N: 8, maxPairs: 16 });
    expect(Array.from(a.pairs)).toEqual(Array.from(b.pairs));
    expect(Array.from(a.pairs)).toEqual([0, 1, 2, 5, 3, 7]);
  });
});

describe("dedupPairs — saturation", () => {
  it("keeps the first maxPairs in canonical order on overflow", () => {
    const inputs = [
      { i: 9, j: 8 }, { i: 0, j: 1 }, { i: 4, j: 2 }, { i: 7, j: 3 },
    ];
    const r = dedupPairs(inputs, { N: 10, maxPairs: 2 });
    expect(r.count).toBe(2);
    expect(r.saturated).toBe(true);
    expect(r.overflow.dropped).toBe(2);
    // Canonical sort: (0,1), (2,4), (3,7), (8,9) → keep first two.
    expect(Array.from(r.pairs)).toEqual([0, 1, 2, 4]);
  });

  it("truncation is stable — same input → same kept set across runs", () => {
    const inputs = Array.from({ length: 50 }, (_, k) => ({
      i: (k * 7) % 30, j: (k * 11 + 1) % 30,
    }));
    const r1 = dedupPairs(inputs, { N: 30, maxPairs: 10 });
    const r2 = dedupPairs([...inputs].reverse(), { N: 30, maxPairs: 10 });
    expect(Array.from(r1.pairs)).toEqual(Array.from(r2.pairs));
  });
});

describe("dedupPacked", () => {
  it("matches dedupPairs on equivalent input", () => {
    const arr = [
      { i: 3, j: 1 }, { i: 1, j: 3 }, { i: 5, j: 2 }, { i: 0, j: 4 },
    ];
    const packed = new Uint32Array(arr.flatMap((p) => [p.i, p.j]));
    const a = dedupPairs(arr, { N: 8, maxPairs: 16 });
    const b = dedupPacked(packed, { N: 8, maxPairs: 16 });
    expect(Array.from(a.pairs)).toEqual(Array.from(b.pairs));
    expect(a.count).toBe(b.count);
  });

  it("rejects odd-length input", () => {
    expect(() => dedupPacked(new Uint32Array([1, 2, 3]), { N: 4, maxPairs: 4 }))
      .toThrow(/even/);
  });
});

describe("Overflow safety with large indices", () => {
  it("disambiguates pairs whose packed key would overflow Number", () => {
    // Indices > 2^16 mean i*2^32 > 2^48, which is fine, but products
    // around 2^53 break Number-keyed sets. Use BigInt-keyed Set.
    const big = 1 << 30;          // 2^30 — well above 2^16
    const N   = big + 4;
    const r = dedupPairs([
      { i: big,     j: big + 1 },
      { i: big + 1, j: big },     // duplicate (normalized)
      { i: big + 2, j: big + 3 },
    ], { N, maxPairs: 16 });
    expect(r.count).toBe(2);
    expect(r.overflow.duplicates).toBe(1);
    expect(Array.from(r.pairs)).toEqual([big, big + 1, big + 2, big + 3]);
  });
});

describe("PairDedupStream", () => {
  it("accumulates across add() calls and dedups across batches", () => {
    const s = new PairDedupStream({ N: 10, maxPairs: 100 });
    s.addArray([{ i: 0, j: 1 }, { i: 2, j: 3 }]);
    s.addArray([{ i: 1, j: 0 }, { i: 4, j: 5 }]); // first is duplicate
    s.addPacked(new Uint32Array([2, 3, 7, 6]));   // first is duplicate
    const r = s.flush();
    expect(r.count).toBe(4);
    expect(r.overflow.duplicates).toBe(2);
    expect(Array.from(r.pairs)).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
  });

  it("bounds memory under saturation (does not grow past maxPairs)", () => {
    const s = new PairDedupStream({ N: 1000, maxPairs: 5 });
    for (let k = 0; k < 100; k++) s.addPair(k, k + 1);
    expect(s.uniqueCount).toBe(5);
    const r = s.flush();
    expect(r.saturated).toBe(true);
    expect(r.overflow.dropped).toBeGreaterThan(0);
    // The first 5 pairs accepted (insertion order before saturation).
    // After sort: (0,1)..(4,5).
    expect(Array.from(r.pairs)).toEqual([0,1, 1,2, 2,3, 3,4, 4,5]);
  });

  it("reset() clears all state", () => {
    const s = new PairDedupStream({ N: 8, maxPairs: 8 });
    s.addPair(0, 1);
    s.addPair(0, 1); // duplicate
    s.reset();
    s.addPair(2, 3);
    const r = s.flush();
    expect(r.count).toBe(1);
    expect(r.overflow.duplicates).toBe(0);
    expect(Array.from(r.pairs)).toEqual([2, 3]);
  });

  it("invariant: inputCount = count + dropped + duplicates + invalid", () => {
    const s = new PairDedupStream({ N: 5, maxPairs: 3 });
    s.addArray([
      { i: 0, j: 1 }, { i: 2, j: 1 }, { i: 3, j: 4 },        // 3 unique → fills
      { i: 0, j: 1 },                                        // duplicate
      { i: 1, j: 1 }, { i: 99, j: 0 },                       // invalid (self, OOR)
      { i: 0, j: 2 },                                        // dropped (saturated)
    ]);
    const r = s.flush();
    expect(s.inputCount).toBe(7);
    expect(r.count + r.overflow.dropped + r.overflow.duplicates + r.overflow.invalid)
      .toBe(7);
  });
});
