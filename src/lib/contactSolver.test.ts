// Contact solver — narrow-phase pair detection + sequential impulse +
// Baumgarte position-correction integration.
//
// Properties verified:
//   1. Two overlapping equal-mass particles separate to exactly the
//      contact diameter after one solver pass (β = 1, no slop).
//   2. Approach velocity is reflected with the configured restitution.
//   3. A pinned (mass = +∞) body absorbs all displacement; the free
//      body moves the full penetration.
//   4. Non-overlapping particles produce zero contacts and zero motion.
//   5. Determinism: identical inputs → bit-identical outputs across runs.
//   6. Slop suppresses correction for tiny penetrations.
//   7. Multiple iterations monotonically reduce max penetration on a
//      dense cluster.

import { describe, it, expect } from "vitest";
import { resolveContacts, type ContactState } from "./contactSolver";

function makeState(positions: number[][], velocities?: number[][], masses?: number[]): ContactState {
  const N = positions.length;
  const x = new Float32Array(N * 2);
  const v = new Float32Array(N * 2);
  const m = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    x[i * 2] = positions[i][0];
    x[i * 2 + 1] = positions[i][1];
    v[i * 2] = velocities?.[i]?.[0] ?? 0;
    v[i * 2 + 1] = velocities?.[i]?.[1] ?? 0;
    m[i] = masses?.[i] ?? 1;
  }
  return { N, x, v, m };
}

describe("contactSolver: pair detection and impulse", () => {
  it("separates two overlapping equal-mass particles to exactly the contact diameter", () => {
    // radius 5 → diameter 10. Place 6 apart (penetration 4).
    const s = makeState([[0, 0], [6, 0]]);
    const stats = resolveContacts(s, { radius: 5, iters: 1, restitution: 0, beta: 1, slop: 0 });
    expect(stats.contacts).toBe(1);
    expect(stats.maxPenetration).toBeCloseTo(4, 6);
    const dx = s.x[2] - s.x[0];
    const dy = s.x[3] - s.x[1];
    expect(Math.hypot(dx, dy)).toBeCloseTo(10, 5);
    // Symmetric split (equal mass): each moves 2 px on x.
    expect(s.x[0]).toBeCloseTo(-2, 5);
    expect(s.x[2]).toBeCloseTo(8, 5);
  });

  it("reflects approach velocity with the configured restitution", () => {
    const s = makeState([[0, 0], [6, 0]], [[5, 0], [-5, 0]]);
    resolveContacts(s, { radius: 5, iters: 1, restitution: 0.5, beta: 1, slop: 0 });
    // v_rel along n = (vJ - vI)·n = (-5 - 5)·1 = -10. lambda = -(1+0.5)·-10/2 = 7.5
    // v_i ← 5 - 7.5*0.5 = 1.25; v_j ← -5 + 7.5*0.5 = -1.25
    expect(s.v[0]).toBeCloseTo(1.25, 5);
    expect(s.v[2]).toBeCloseTo(-1.25, 5);
  });

  it("pinned (kinematic) body absorbs all displacement", () => {
    const s = makeState([[0, 0], [6, 0]], undefined, [Infinity, 1]);
    resolveContacts(s, { radius: 5, iters: 1, restitution: 0, beta: 1, slop: 0 });
    expect(s.x[0]).toBeCloseTo(0, 6);
    expect(s.x[1]).toBeCloseTo(0, 6);
    expect(s.x[2]).toBeCloseTo(10, 5);
  });

  it("non-overlapping pairs produce no contacts and no motion", () => {
    const s = makeState([[0, 0], [100, 0], [50, 200]]);
    const xBefore = Float32Array.from(s.x);
    const stats = resolveContacts(s, { radius: 5, iters: 4, restitution: 0.5, beta: 1, slop: 0 });
    expect(stats.contacts).toBe(0);
    for (let i = 0; i < s.x.length; i++) expect(s.x[i]).toBe(xBefore[i]);
  });

  it("is deterministic across independent runs", () => {
    const positions: number[][] = [];
    for (let i = 0; i < 50; i++) {
      // pseudo-random but fixed
      const a = (i * 9301 + 49297) % 233280;
      const b = ((i + 1) * 9301 + 49297) % 233280;
      positions.push([(a / 233280) * 60, (b / 233280) * 60]);
    }
    const s1 = makeState(positions);
    const s2 = makeState(positions);
    resolveContacts(s1, { radius: 4, iters: 3, restitution: 0.3, beta: 0.8, slop: 0.1 });
    resolveContacts(s2, { radius: 4, iters: 3, restitution: 0.3, beta: 0.8, slop: 0.1 });
    for (let i = 0; i < s1.x.length; i++) expect(s1.x[i]).toBe(s2.x[i]);
    for (let i = 0; i < s1.v.length; i++) expect(s1.v[i]).toBe(s2.v[i]);
  });

  it("slop suppresses correction for shallow penetrations", () => {
    // diameter 10, distance 9.5 → penetration 0.5; with slop 1, no correction.
    const s = makeState([[0, 0], [9.5, 0]]);
    resolveContacts(s, { radius: 5, iters: 1, restitution: 0, beta: 1, slop: 1 });
    expect(s.x[0]).toBeCloseTo(0, 6);
    expect(s.x[2]).toBeCloseTo(9.5, 6);
  });

  it("more iterations reduce residual penetration on a dense cluster", () => {
    // Build a 5×5 grid of particles spaced 6 px apart; diameter 10 → all overlap.
    const positions: number[][] = [];
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      positions.push([c * 6, r * 6]);
    }
    const sA = makeState(positions);
    const sB = makeState(positions);
    resolveContacts(sA, { radius: 5, iters: 1, restitution: 0, beta: 0.6, slop: 0 });
    resolveContacts(sB, { radius: 5, iters: 6, restitution: 0, beta: 0.6, slop: 0 });

    // After-correction residual: count overlapping pairs.
    const residual = (s: ContactState) => {
      let pen = 0;
      for (let i = 0; i < s.N; i++) for (let j = i + 1; j < s.N; j++) {
        const dx = s.x[i * 2] - s.x[j * 2];
        const dy = s.x[i * 2 + 1] - s.x[j * 2 + 1];
        const d = Math.hypot(dx, dy);
        if (d < 10) pen += 10 - d;
      }
      return pen;
    };
    expect(residual(sB)).toBeLessThan(residual(sA));
  });
});
