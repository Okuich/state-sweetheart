// Verify pairDedup is integrated into the gpuNarrowPhase pipeline:
//   1. Duplicate pairs in the input produce ONE contact (not many).
//   2. Reversed (j,i) and self-pairs (i,i) are normalized / dropped.
//   3. Out-of-range indices are dropped (no crash).
//   4. SolveResult.dedup reports per-category drop counts and saturation.
//   5. dedup: "skip" bypasses the pipeline (caller-guaranteed canonical).

import { describe, it, expect } from "vitest";
import {
  BodyKind,
  solvePairsCpu,
  type BodySet,
  type CandidatePair,
  type SolveOptions,
} from "./gpuNarrowPhase";

function makeBodies(positions: number[][], radius = 1): BodySet {
  const N = positions.length;
  const pos = new Float32Array(N * 2);
  const vel = new Float32Array(N * 2);
  const invMass = new Float32Array(N);
  const kind = new Uint32Array(N);
  const extra = new Float32Array(N);
  const restitution = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    pos[i * 2]     = positions[i][0];
    pos[i * 2 + 1] = positions[i][1];
    invMass[i] = 1;
    kind[i] = BodyKind.Particle;
    extra[i] = radius;
    restitution[i] = 0;
  }
  return { N, pos, vel, invMass, kind, extra, restitution };
}

describe("gpuNarrowPhase ⨯ pairDedup integration", () => {
  it("collapses duplicate input pairs to a single canonical contact", () => {
    // 4 bodies, all within contact range; emit each pair 3× plus reversed.
    const bodies = makeBodies([[0, 0], [1.0, 0], [0, 1.0], [1.0, 1.0]], 1);
    const dups: CandidatePair[] = [];
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        if (i === j) continue;
        dups.push({ i, j }); // both (i,j) and (j,i) appear in the loop
        dups.push({ i, j }); // and again
      }
    }
    const opts: SolveOptions = { bodies, pairs: dups };
    const r = solvePairsCpu(opts, 64);
    // 4 bodies → 6 unique pairs max; all overlap so all become contacts.
    expect(r.contacts.length).toBe(6);
    // Every contact must have i < j (canonical normalization).
    for (const c of r.contacts) expect(c.i).toBeLessThan(c.j);
    // Dedup diagnostics
    expect(r.dedup).toBeDefined();
    expect(r.dedup!.inputCount).toBe(dups.length);
    expect(r.dedup!.uniqueCount).toBe(6);
    expect(r.dedup!.duplicates).toBe(dups.length - 6);
    expect(r.dedup!.invalid).toBe(0);
    expect(r.dedup!.dropped).toBe(0);
    expect(r.dedup!.saturated).toBe(false);
  });

  it("drops self-pairs and out-of-range indices as invalid", () => {
    const bodies = makeBodies([[0, 0], [1.0, 0]], 1);
    const pairs: CandidatePair[] = [
      { i: 0, j: 0 },          // self
      { i: 1, j: 1 },          // self
      { i: 0, j: 5 },          // out of range
      { i: 99, j: 0 },         // out of range
      { i: 0, j: 1 },          // valid
      { i: 1, j: 0 },          // duplicate of above (reversed)
    ];
    const r = solvePairsCpu({ bodies, pairs }, 32);
    expect(r.contacts.length).toBe(1);
    expect(r.contacts[0].i).toBe(0);
    expect(r.contacts[0].j).toBe(1);
    expect(r.dedup!.invalid).toBe(4);
    expect(r.dedup!.duplicates).toBe(1);
    expect(r.dedup!.uniqueCount).toBe(1);
  });

  it("reports saturation when maxPairs caps the unique set", () => {
    const bodies = makeBodies([[0, 0], [1, 0], [0, 1], [1, 1]], 1);
    const pairs: CandidatePair[] = [];
    for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) pairs.push({ i, j });
    // Cap at 2 unique pairs → 4 should be dropped.
    const r = solvePairsCpu({ bodies, pairs, maxPairs: 2 }, 32);
    expect(r.dedup!.uniqueCount).toBe(2);
    expect(r.dedup!.dropped).toBe(4);
    expect(r.dedup!.saturated).toBe(true);
    expect(r.contacts.length).toBeLessThanOrEqual(2);
  });

  it('dedup: "skip" bypasses the pipeline', () => {
    const bodies = makeBodies([[0, 0], [1, 0]], 1);
    const r = solvePairsCpu({ bodies, pairs: [{ i: 0, j: 1 }], dedup: "skip" }, 32);
    expect(r.contacts.length).toBe(1);
    expect(r.dedup).toBeUndefined();
  });

  it("is order-independent: shuffled duplicates produce the same contacts", () => {
    const bodies = makeBodies([[0, 0], [1, 0], [0, 1]], 1);
    const a: CandidatePair[] = [
      { i: 0, j: 1 }, { i: 1, j: 0 }, { i: 1, j: 2 }, { i: 0, j: 2 },
    ];
    const b: CandidatePair[] = [
      { i: 2, j: 0 }, { i: 0, j: 2 }, { i: 1, j: 2 }, { i: 1, j: 0 },
    ];
    const ra = solvePairsCpu({ bodies, pairs: a }, 32);
    const rb = solvePairsCpu({ bodies, pairs: b }, 32);
    expect(ra.contacts.length).toBe(rb.contacts.length);
    for (let k = 0; k < ra.contacts.length; k++) {
      expect(ra.contacts[k].i).toBe(rb.contacts[k].i);
      expect(ra.contacts[k].j).toBe(rb.contacts[k].j);
    }
  });
});
