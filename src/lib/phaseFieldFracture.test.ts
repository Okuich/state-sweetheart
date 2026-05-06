import { describe, it, expect } from "vitest";
import { initPhaseField, resetPhaseField, stepPhaseField } from "./phaseFieldFracture";

describe("phase-field fracture", () => {
  it("starts pristine (d=0 everywhere)", () => {
    const s = initPhaseField({ W: 32, H: 32 });
    expect(Array.from(s.d).every((v) => v === 0)).toBe(true);
    expect(s.crackSizeHistory).toEqual([]);
  });

  it("damage is monotone non-decreasing across steps", () => {
    const s = initPhaseField({ W: 24, H: 24 });
    for (let k = 0; k < 30; k++) stepPhaseField(s, 0.05);
    const a = Float32Array.from(s.d);
    for (let k = 0; k < 5; k++) stepPhaseField(s, 0); // no extra load
    for (let i = 0; i < a.length; i++) {
      expect(s.d[i]).toBeGreaterThanOrEqual(a[i] - 1e-6);
    }
  });

  it("crack initiates at the notch tip first", () => {
    const s = initPhaseField({ W: 32, H: 32, notch: [0.0, 0.5, 0.3, 0.5] });
    for (let k = 0; k < 40; k++) stepPhaseField(s, 0.04);
    // Find argmax of damage; expect it near the notch tip (x≈0.3, y≈0.5).
    let bestIdx = 0, best = -Infinity;
    for (let i = 0; i < s.d.length; i++) {
      if (s.d[i] > best) { best = s.d[i]; bestIdx = i; }
    }
    const xi = bestIdx % s.W, yj = Math.floor(bestIdx / s.W);
    expect(best).toBeGreaterThan(0);
    expect(Math.abs(xi - 0.3 * s.W)).toBeLessThan(s.W * 0.4);
    expect(Math.abs(yj - 0.5 * s.H)).toBeLessThan(s.H * 0.25);
  });

  it("crack-set size grows monotonically under load", () => {
    const s = initPhaseField({ W: 24, H: 24, dThreshold: 0.5 });
    for (let k = 0; k < 60; k++) stepPhaseField(s, 0.04);
    const hist = s.crackSizeHistory;
    let maxSeen = 0;
    for (const c of hist) {
      expect(c).toBeGreaterThanOrEqual(maxSeen);
      maxSeen = c;
    }
    expect(maxSeen).toBeGreaterThan(0);
  });

  it("reset returns to pristine and clears history", () => {
    const s = initPhaseField();
    for (let k = 0; k < 5; k++) stepPhaseField(s, 0.1);
    resetPhaseField(s);
    expect(s.step).toBe(0);
    expect(s.loadFactor).toBe(0);
    expect(s.crackSizeHistory).toEqual([]);
    expect(Array.from(s.d).every((v) => v === 0)).toBe(true);
  });

  it("deterministic — two runs with same load schedule match", () => {
    const a = initPhaseField({ W: 20, H: 20 });
    const b = initPhaseField({ W: 20, H: 20 });
    for (let k = 0; k < 40; k++) { stepPhaseField(a, 0.03); stepPhaseField(b, 0.03); }
    for (let i = 0; i < a.d.length; i++) expect(a.d[i]).toBeCloseTo(b.d[i], 6);
  });
});
