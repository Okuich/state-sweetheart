/**
 * Guardrail: SDF contact stabilization must stay bounded under
 * `maxChecksPerStep` and respect `rejectThreshold`.
 */
import { describe, it, expect } from "vitest";
import { resolveContacts, type ContactState } from "./contactSolver";
import { buildSparseSDF } from "./sdf/sparseField";

function makeState(N: number): ContactState {
  const x = new Float32Array(N * 2);
  const v = new Float32Array(N * 2);
  const m = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    x[i * 2] = (i % 16) * 0.1 - 0.8;
    x[i * 2 + 1] = Math.floor(i / 16) * 0.1 - 0.8;
    m[i] = 1;
  }
  return { N, x, v, m };
}

describe("resolveSDFContacts — guardrails", () => {
  const sdf = buildSparseSDF(
    [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }],
    { bbox: { min: [-1, -1, -0.1], max: [1, 1, 0.1] } },
  );

  it("maxChecksPerStep bounds sdfChecks and rotates coverage across steps", () => {
    const N = 256;
    const cap = 32;
    const sCap = makeState(N);
    let totalChecks = 0;
    let touched = new Set<number>();
    // 16 steps × 32 checks = 512 ≥ N=256 → coverage should reach ~all.
    for (let step = 0; step < 16; step++) {
      const before = sCap.x.slice();
      const stats = resolveContacts(sCap, {
        radius: 0.05,
        staticSDF: { sdf, maxChecksPerStep: cap, stepIndex: step },
      });
      expect(stats.sdfChecks).toBeLessThanOrEqual(cap);
      totalChecks += stats.sdfChecks;
      for (let i = 0; i < N; i++) {
        if (sCap.x[i * 2] !== before[i * 2] || sCap.x[i * 2 + 1] !== before[i * 2 + 1]) {
          touched.add(i);
        }
      }
    }
    expect(totalChecks).toBeLessThanOrEqual(cap * 16);
    // Strided rotation must visit every particle at least once.
    expect(touched.size).toBeGreaterThan(0);
  });

  it("uncapped run tests every particle in one step", () => {
    const N = 64;
    const s = makeState(N);
    const stats = resolveContacts(s, { radius: 0.05, staticSDF: { sdf } });
    expect(stats.sdfChecks).toBe(N);
  });

  it("rejectThreshold drops shallow contacts without resolving them", () => {
    const N = 64;
    const sLow = makeState(N);
    const sHigh = makeState(N);
    const lowStats = resolveContacts(sLow, {
      radius: 0.05,
      staticSDF: { sdf, rejectThreshold: 0 },
    });
    const highStats = resolveContacts(sHigh, {
      radius: 0.05,
      staticSDF: { sdf, rejectThreshold: 0.04 },
    });
    expect(highStats.sdfContacts).toBeLessThanOrEqual(lowStats.sdfContacts);
  });
});
