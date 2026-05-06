import { describe, it, expect } from "vitest";
import {
  makeParticle,
  primeVerlet,
  verletStepSpring,
  analyticalSHO,
  springEnergy,
  type VerletParticle,
} from "./verletSpring";

/**
 * Velocity-Verlet vs. analytical 1D spring.
 *
 * Verlet is 2nd-order symplectic, so:
 *   • Phase-space error ~ O(dt²) over a fixed interval.
 *   • Total energy oscillates but does NOT drift secularly (undamped case).
 *   • With added linear drag F = −c·v, energy decays monotonically (no growth).
 */

const EPS = 1e-9;

function maxStateError(p: VerletParticle, ref: { x: number; v: number }) {
  return Math.max(Math.abs(p.x - ref.x), Math.abs(p.v - ref.v));
}

describe("velocity-Verlet — single 1D spring", () => {
  it("matches the analytical SHO solution within O(dt²) over one period", () => {
    const K = 4;          // spring stiffness
    const m = 1;
    const w = Math.sqrt(K / m);
    const T = (2 * Math.PI) / w; // period
    const x0 = 1.0;
    const v0 = 0.0;
    const dt = T / 2000;        // 2000 steps per period
    const steps = 2000;          // exactly one period

    const p = makeParticle(x0, v0, m);
    primeVerlet(p, K);
    for (let i = 0; i < steps; i++) verletStepSpring(p, dt, K, 0, "explicit");

    const ref = analyticalSHO(x0, v0, K, m, steps * dt);
    const err = maxStateError(p, ref);
    // For dt = T/2000, the local error is ~K·dt²/24 → global ~10⁻⁵ over 1 period.
    expect(err).toBeLessThan(1e-3);
    // Position must come back near x0 after one full period.
    expect(Math.abs(p.x - x0)).toBeLessThan(1e-3);
  });

  it("error scales like O(dt²) (refining dt by 2× shrinks error by ≥3.5×)", () => {
    const K = 9, m = 1;
    const w = Math.sqrt(K / m);
    const T = (2 * Math.PI) / w;
    const x0 = 0.7, v0 = 0.0;

    const runErr = (dt: number) => {
      const steps = Math.round(T / dt);
      const p = makeParticle(x0, v0, m);
      primeVerlet(p, K);
      for (let i = 0; i < steps; i++) verletStepSpring(p, dt, K, 0, "explicit");
      const ref = analyticalSHO(x0, v0, K, m, steps * dt);
      return maxStateError(p, ref) + EPS;
    };

    const e1 = runErr(T / 200);
    const e2 = runErr(T / 400);
    // 2nd-order method → ratio should be ~4 for halving dt; allow ≥3.5 slack.
    expect(e1 / e2).toBeGreaterThan(3.5);
  });

  it("conserves energy without secular drift over many periods (undamped)", () => {
    const K = 16, m = 1;
    const w = Math.sqrt(K / m);
    const T = (2 * Math.PI) / w;
    const dt = T / 500;
    const steps = 500 * 50; // 50 periods

    const p = makeParticle(1.0, 0.0, m);
    primeVerlet(p, K);
    const E0 = springEnergy(p, K);
    let Emin = E0, Emax = E0;
    for (let i = 0; i < steps; i++) {
      verletStepSpring(p, dt, K, 0, "explicit");
      const E = springEnergy(p, K);
      if (E < Emin) Emin = E;
      if (E > Emax) Emax = E;
    }
    // Symplectic: bounded oscillation, no monotonic drift.
    const swing = (Emax - Emin) / E0;
    expect(swing).toBeLessThan(0.01);              // <1% bounded swing
    const Eend = springEnergy(p, K);
    expect(Math.abs(Eend - E0) / E0).toBeLessThan(0.01);
  });
});

describe("velocity-Verlet — drag modes (damped oscillator)", () => {
  it("explicit and exponential drag both decay energy monotonically", () => {
    for (const mode of ["explicit", "exponential", "force"] as const) {
      const K = 4, m = 1, c = 0.5;
      const w0 = Math.sqrt(K / m);
      const dt = (2 * Math.PI) / w0 / 800;
      const steps = 4000;

      const p = makeParticle(1.0, 0.0, m);
      primeVerlet(p, K);
      let prevE = springEnergy(p, K);
      let violations = 0;
      for (let i = 0; i < steps; i++) {
        verletStepSpring(p, dt, K, c, mode);
        const E = springEnergy(p, K);
        // tiny per-step rebound is allowed (Verlet kinetic-energy ripple),
        // but energy must trend strictly down on a coarse window.
        if (i % 200 === 199) {
          if (E > prevE * 1.001) violations++;
          prevE = E;
        }
      }
      expect(violations).toBe(0);
      // After 4000 steps with c=0.5 and ω≈2 → E should drop ≥90%.
      expect(springEnergy(p, K)).toBeLessThan(0.1);
    }
  });

  it("explicit drag stays stable when k·dt < 1 and decays as expected", () => {
    const K = 1, m = 1, c = 0.2;
    const dt = 0.05; // c·dt = 0.01 ≪ 1
    const steps = 2000;
    const p = makeParticle(1.0, 0.0, m);
    primeVerlet(p, K);
    for (let i = 0; i < steps; i++) verletStepSpring(p, dt, K, c, "explicit");
    // Underdamped analytical envelope: |x| ≤ exp(−c·t/(2m)) · √(x₀² + (v₀/ω_d)²).
    // With c=0.2, t=100 → envelope ~ exp(−10) ≈ 4.5e-5; allow generous slack.
    expect(Math.abs(p.x)).toBeLessThan(1e-3);
    expect(Math.abs(p.v)).toBeLessThan(1e-3);
  });
});

describe("velocity-Verlet — small system of two coupled springs", () => {
  // Two masses connected by a spring; center of mass should be stationary
  // when initialized with equal-and-opposite velocities (momentum conserved).
  it("conserves momentum in a 2-body chain with no external force", () => {
    const K = 6, m = 1;
    const x = [-0.5, 0.5];
    const v = [-0.2, 0.2]; // P_total = 0
    const fPrev = [0, 0];
    const f = [0, 0];

    // initial forces (spring between 0 and 1, rest length 1)
    const force = (x: number[]) => {
      const dx = x[1] - x[0];
      const stretch = dx - 1.0;
      const F = K * stretch;
      return [F, -F]; // pulls them toward each other when stretched
    };
    [fPrev[0], fPrev[1]] = force(x);

    const dt = 0.005;
    const steps = 4000;
    let pTotalMax = 0;
    for (let s = 0; s < steps; s++) {
      // drift
      for (let i = 0; i < 2; i++) {
        v[i] += 0.5 * (fPrev[i] / m) * dt;
        x[i] += v[i] * dt;
      }
      // recompute force
      [f[0], f[1]] = force(x);
      // kick
      for (let i = 0; i < 2; i++) {
        v[i] = v[i] + 0.5 * (f[i] / m) * dt;
        fPrev[i] = f[i];
      }
      const P = m * v[0] + m * v[1];
      if (Math.abs(P) > pTotalMax) pTotalMax = Math.abs(P);
    }
    // Newton's 3rd law preserved by the symmetric force pair → P stays ≈0.
    expect(pTotalMax).toBeLessThan(1e-10);
  });
});
