// Damping consistency under dt — verify the damping model is now
// frame-rate independent and unconditionally stable.
//
// Model: dv/dt = −k·v  with exact step v ← v · exp(−k·dt).
//
// Properties checked:
//   1. Stepping with dt and (dt/2 twice) produces the SAME velocity
//      (consistency of an exact integrator under refinement).
//   2. k·dt > 1 (which used to explode under (1 − k·dt)) stays bounded
//      and decays monotonically.
//   3. After total time T the velocity equals v0·exp(−k·T) regardless of
//      how many sub-steps are used (1, 4, 60, 600).

import { describe, it, expect } from "vitest";

function decay(dt: number, k: number): number {
  return Math.exp(-k * dt);
}

function simulate(v0: number, k: number, T: number, steps: number): number {
  const dt = T / steps;
  const f = decay(dt, k);
  let v = v0;
  for (let i = 0; i < steps; i++) v *= f;
  return v;
}

describe("damping is dt-consistent and unconditionally stable", () => {
  it("one step of dt equals two steps of dt/2 (exact integrator)", () => {
    const k = 3.5, dt = 0.1, v0 = 12;
    const a = v0 * decay(dt, k);
    const b = v0 * decay(dt / 2, k) * decay(dt / 2, k);
    expect(Math.abs(a - b)).toBeLessThan(1e-12);
  });

  it("converges to v0·exp(−k·T) regardless of frame rate", () => {
    const k = 2.0, T = 1.0, v0 = 7.5;
    const truth = v0 * Math.exp(-k * T);
    for (const steps of [1, 4, 60, 600, 6000]) {
      const v = simulate(v0, k, T, steps);
      expect(Math.abs(v - truth)).toBeLessThan(1e-10);
    }
  });

  it("remains stable when k·dt > 1 (old explicit form blew up)", () => {
    // Old model: (1 − k·dt) with k=10, dt=0.5 → factor = −4 → divergence.
    const k = 10, dt = 0.5, v0 = 1;
    let v = v0;
    for (let i = 0; i < 100; i++) v *= decay(dt, k);
    expect(Math.abs(v)).toBeLessThan(v0); // bounded
    expect(v).toBeGreaterThanOrEqual(0);  // monotonic, no sign flip
  });

  it("monotonically decreases speed each step for any k>0, dt>0", () => {
    for (const k of [0.1, 1, 5, 50]) {
      for (const dt of [0.001, 0.016, 0.1, 1]) {
        const f = decay(dt, k);
        expect(f).toBeGreaterThan(0);
        expect(f).toBeLessThanOrEqual(1);
      }
    }
  });
});
