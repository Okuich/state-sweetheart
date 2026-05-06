import { describe, it, expect } from "vitest";
import {
  advance, checkpoint, hashState, initMaterialState, replay, restore,
  RollbackBuffer, type StepInputs,
} from "./materialCheckpoint";

const baseInp = (E: number, deps: number): StepInputs => ({
  depsTrial: Float32Array.from({ length: E }, () => deps),
  dt: 0.01, yieldStress: 100, hardening: 50, Emod: 1000, tau: 0.05, eta: 10,
});

describe("initMaterialState", () => {
  it("Fp = identity, Sv = 0, alpha = 0", () => {
    const s = initMaterialState(3);
    for (let e = 0; e < 3; e++) {
      expect(s.Fp[e * 4 + 0]).toBe(1);
      expect(s.Fp[e * 4 + 1]).toBe(0);
      expect(s.Fp[e * 4 + 2]).toBe(0);
      expect(s.Fp[e * 4 + 3]).toBe(1);
    }
    expect(Array.from(s.Sv).every((v) => v === 0)).toBe(true);
    expect(Array.from(s.alpha).every((v) => v === 0)).toBe(true);
  });
});

describe("advance — plastic flow updates Fp and alpha", () => {
  it("elastic step (under yield) leaves Fp = I, alpha = 0", () => {
    const s = initMaterialState(2);
    advance(s, baseInp(2, 0.05)); // sigTrial = 50 < yield 100
    for (let e = 0; e < 2; e++) {
      expect(s.Fp[e * 4 + 0]).toBeCloseTo(1, 6);
      expect(s.alpha[e]).toBe(0);
    }
  });

  it("over-yield step grows alpha and Fp[0]", () => {
    const s = initMaterialState(1);
    advance(s, baseInp(1, 0.5)); // sigTrial = 500, yield 100
    expect(s.alpha[0]).toBeGreaterThan(0);
    expect(s.Fp[0]).toBeGreaterThan(1);
  });
});

describe("advance — viscoelastic Sv decays + integrates", () => {
  it("Sv approaches steady state under constant strain rate", () => {
    const s = initMaterialState(1);
    for (let k = 0; k < 200; k++) advance(s, baseInp(1, 0.01));
    // Steady-state: Sv ≈ η·dε / (1 - exp(-dt/τ))  → bounded, > 0.
    expect(s.Sv[0]).toBeGreaterThan(0);
    expect(s.Sv[0]).toBeLessThan(100);
  });

  it("Sv relaxes towards 0 with no strain", () => {
    const s = initMaterialState(1);
    advance(s, baseInp(1, 0.5));      // pump some Sv
    const after = s.Sv[0];
    for (let k = 0; k < 100; k++) advance(s, baseInp(1, 0));
    expect(Math.abs(s.Sv[0])).toBeLessThan(Math.abs(after) * 0.05);
  });
});

describe("checkpoint / restore", () => {
  it("round-trips state and step", () => {
    const s = initMaterialState(4);
    for (let k = 0; k < 5; k++) advance(s, baseInp(4, 0.2));
    const cp = checkpoint(s);
    const h0 = hashState(s);

    // mutate
    for (let k = 0; k < 10; k++) advance(s, baseInp(4, 0.5));
    expect(hashState(s)).not.toBe(h0);

    restore(s, cp);
    expect(s.step).toBe(cp.step);
    expect(hashState(s)).toBe(h0);
  });

  it("checkpoint is independent (doesn't alias state arrays)", () => {
    const s = initMaterialState(2);
    advance(s, baseInp(2, 0.3));
    const cp = checkpoint(s);
    const fpBefore = cp.Fp[0];
    advance(s, baseInp(2, 0.4));
    expect(cp.Fp[0]).toBe(fpBefore); // didn't follow s.Fp
  });
});

describe("RollbackBuffer", () => {
  it("retains only `capacity` checkpoints", () => {
    const buf = new RollbackBuffer(3);
    const s = initMaterialState(1);
    for (let k = 0; k < 5; k++) {
      advance(s, baseInp(1, 0.1));
      buf.push(checkpoint(s));
    }
    expect(buf.size()).toBe(3);
    expect(buf.latest()!.step).toBe(5);
  });

  it("findAtOrBefore returns the closest preceding checkpoint", () => {
    const buf = new RollbackBuffer();
    const s = initMaterialState(1);
    for (let k = 0; k < 4; k++) {
      advance(s, baseInp(1, 0.1));
      buf.push(checkpoint(s));
    }
    expect(buf.findAtOrBefore(2)?.step).toBe(2);
    expect(buf.findAtOrBefore(99)?.step).toBe(4);
    expect(buf.findAtOrBefore(0)).toBeUndefined();
  });
});

describe("deterministic replay", () => {
  it("two replays from the same checkpoint with the same inputs match bit-exactly", () => {
    const E = 8;
    const s = initMaterialState(E);
    for (let k = 0; k < 7; k++) advance(s, baseInp(E, 0.2));
    const cp = checkpoint(s);
    const inputs = Array.from({ length: 30 }, (_, k) => baseInp(E, 0.05 + 0.01 * (k % 5)));
    const a = replay(cp, E, inputs);
    const b = replay(cp, E, inputs);
    expect(a.finalHash).toBe(b.finalHash);
    for (let i = 0; i < a.state.Fp.length; i++) expect(a.state.Fp[i]).toBe(b.state.Fp[i]);
    for (let i = 0; i < a.state.Sv.length; i++) expect(a.state.Sv[i]).toBe(b.state.Sv[i]);
  });

  it("rollback + re-run from buffered checkpoint reproduces the live trajectory", () => {
    const E = 4;
    const s = initMaterialState(E);
    const buf = new RollbackBuffer();
    const inputs: StepInputs[] = [];
    for (let k = 0; k < 12; k++) {
      const inp = baseInp(E, 0.05 * (k + 1));
      inputs.push(inp);
      advance(s, inp);
      buf.push(checkpoint(s));
    }
    const finalHash = hashState(s);
    // Roll back to step 7.
    const cp7 = buf.findAtOrBefore(7)!;
    expect(cp7.step).toBe(7);
    const tail = inputs.slice(7);
    const re = replay(cp7, E, tail);
    expect(re.finalHash).toBe(finalHash);
  });
});
