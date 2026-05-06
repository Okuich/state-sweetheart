// Deterministic test scene comparing total force magnitudes between
// gravityMode === "uniform" and gravityMode === "zero" immediately after
// each reset/apply stage.
//
// Mirrors the body-force pipeline used inside PhysicsCanvas:
//   - reset:  s.f.fill(0)
//   - apply:  if gMode !== "zero" && p.gravity !== 0:
//               s.f[i*2 + 1] += p.gravity * s.m[i]
//
// Properties checked (all bit-deterministic):
//   1. After reset, |F| == 0 in both modes.
//   2. After apply with "zero", |F| stays exactly 0.
//   3. After apply with "uniform", sum |F_i| == |g| * Σ m_i and
//      Σ F_y == g * Σ m_i (no x component).
//   4. Difference (uniform − zero) of total force magnitude equals
//      |g| * Σ m_i exactly across multiple reset/apply cycles.

import { describe, it, expect } from "vitest";

type GravityMode = "uniform" | "directional" | "zero";

interface Scene {
  n: number;
  m: Float64Array;
  f: Float64Array; // length 2n
}

function makeScene(masses: number[]): Scene {
  const n = masses.length;
  return {
    n,
    m: Float64Array.from(masses),
    f: new Float64Array(n * 2),
  };
}

function reset(s: Scene) {
  s.f.fill(0);
}

function applyGravity(s: Scene, gravity: number, mode: GravityMode) {
  if (mode === "zero" || gravity === 0) return;
  // uniform: classic +y body force (matches PhysicsCanvas line ~1141)
  for (let i = 0; i < s.n; i++) {
    s.f[i * 2 + 1] += gravity * s.m[i];
  }
}

function totalForceMag(s: Scene): number {
  let sum = 0;
  for (let i = 0; i < s.n; i++) {
    const fx = s.f[i * 2], fy = s.f[i * 2 + 1];
    sum += Math.hypot(fx, fy);
  }
  return sum;
}

function sumFy(s: Scene): number {
  let sum = 0;
  for (let i = 0; i < s.n; i++) sum += s.f[i * 2 + 1];
  return sum;
}

const MASSES = [1, 2, 0.5, 3, 1.25, 0.75, 4, 2.5];
const TOTAL_M = MASSES.reduce((a, b) => a + b, 0);
const G = 9.81;

describe("gravity force scene: uniform vs zero", () => {
  it("reset zeroes forces in both modes", () => {
    const sU = makeScene(MASSES);
    const sZ = makeScene(MASSES);
    // dirty them first
    sU.f.fill(7); sZ.f.fill(-3);
    reset(sU); reset(sZ);
    expect(totalForceMag(sU)).toBe(0);
    expect(totalForceMag(sZ)).toBe(0);
  });

  it("apply with zero leaves |F| at exactly 0", () => {
    const s = makeScene(MASSES);
    reset(s);
    applyGravity(s, G, "zero");
    expect(totalForceMag(s)).toBe(0);
    expect(sumFy(s)).toBe(0);
  });

  it("apply with uniform gives Σ|F_i| = |g|·Σm and Σ F_y = g·Σm", () => {
    const s = makeScene(MASSES);
    reset(s);
    applyGravity(s, G, "uniform");
    expect(totalForceMag(s)).toBeCloseTo(G * TOTAL_M, 12);
    expect(sumFy(s)).toBeCloseTo(G * TOTAL_M, 12);
    // no x component
    for (let i = 0; i < s.n; i++) expect(s.f[i * 2]).toBe(0);
  });

  it("uniform − zero difference equals |g|·Σm across repeated reset/apply cycles", () => {
    const sU = makeScene(MASSES);
    const sZ = makeScene(MASSES);
    for (let cycle = 0; cycle < 5; cycle++) {
      reset(sU); reset(sZ);
      applyGravity(sU, G, "uniform");
      applyGravity(sZ, G, "zero");
      const diff = totalForceMag(sU) - totalForceMag(sZ);
      expect(diff).toBeCloseTo(G * TOTAL_M, 12);
    }
  });

  it("is deterministic: two independent runs produce bit-identical forces", () => {
    const a = makeScene(MASSES);
    const b = makeScene(MASSES);
    reset(a); applyGravity(a, G, "uniform");
    reset(b); applyGravity(b, G, "uniform");
    for (let i = 0; i < a.f.length; i++) expect(a.f[i]).toBe(b.f[i]);
  });

  it("gravity = 0 in uniform mode behaves like zero mode", () => {
    const sU = makeScene(MASSES);
    const sZ = makeScene(MASSES);
    reset(sU); reset(sZ);
    applyGravity(sU, 0, "uniform");
    applyGravity(sZ, G, "zero");
    expect(totalForceMag(sU)).toBe(totalForceMag(sZ));
    expect(totalForceMag(sU)).toBe(0);
  });
});
