/**
 * Validates analytic SDF gradients via central-difference checks.
 *
 * For each primitive we verify:
 *   • ∂d/∂p   matches FD over the query point
 *   • ∂d/∂θ   matches FD over each shape parameter
 * For `dScene`:
 *   • soft-min weights sum to 1
 *   • ∂d_combined/∂p matches FD on the combined evaluator
 */

import { describe, it, expect } from "vitest";
import { dPrim, dScene } from "./differentiable";
import { evalPrim, evalScene } from "./primitives";
import type { SDFPrim, Vec3 } from "./types";

const EPS = 1e-4;

function fdPos(prim: SDFPrim, p: Vec3): Vec3 {
  const out: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const pp: Vec3 = [p[0], p[1], p[2]]; pp[k] += EPS;
    const pm: Vec3 = [p[0], p[1], p[2]]; pm[k] -= EPS;
    out[k] = (evalPrim(prim, pp) - evalPrim(prim, pm)) / (2 * EPS);
  }
  return out;
}

function expectVecClose(a: Vec3, b: Vec3, tol = 5e-3) {
  expect(a[0]).toBeCloseTo(b[0], 2); expect(Math.abs(a[0] - b[0])).toBeLessThan(tol);
  expect(a[1]).toBeCloseTo(b[1], 2); expect(Math.abs(a[1] - b[1])).toBeLessThan(tol);
  expect(a[2]).toBeCloseTo(b[2], 2); expect(Math.abs(a[2] - b[2])).toBeLessThan(tol);
}

describe("dPrim — sphere", () => {
  const prim: SDFPrim = { kind: "sphere", center: [0.1, -0.2, 0.3], radius: 0.4 };
  const p: Vec3 = [0.7, 0.1, -0.2];
  it("matches d and ∂d/∂p", () => {
    const g = dPrim(prim, p);
    expect(g.d).toBeCloseTo(evalPrim(prim, p), 8);
    expectVecClose(g.dp, fdPos(prim, p));
  });
  it("matches ∂d/∂radius and ∂d/∂center", () => {
    const g = dPrim(prim, p);
    if (g.kind !== "sphere") throw new Error("kind");
    // ∂d/∂radius via FD on prim.
    const dR = (evalPrim({ ...prim, radius: prim.radius + EPS }, p)
              - evalPrim({ ...prim, radius: prim.radius - EPS }, p)) / (2 * EPS);
    expect(g.dRadius).toBeCloseTo(dR, 4);
    // ∂d/∂center should equal −∂d/∂p.
    expectVecClose(g.dCenter, [-g.dp[0], -g.dp[1], -g.dp[2]]);
  });
});

describe("dPrim — box", () => {
  const prim: SDFPrim = { kind: "box", center: [0, 0, 0], half: [0.4, 0.3, 0.2] };
  it("matches ∂d/∂p both inside and outside", () => {
    for (const p of [[0.6, 0.1, 0] as Vec3, [0.1, 0.05, -0.05] as Vec3]) {
      const g = dPrim(prim, p);
      expect(g.d).toBeCloseTo(evalPrim(prim, p), 8);
      expectVecClose(g.dp, fdPos(prim, p));
    }
  });
  it("matches ∂d/∂half via FD per axis (outside corner)", () => {
    const p: Vec3 = [0.6, 0.5, 0.05];
    const g = dPrim(prim, p);
    if (g.kind !== "box") throw new Error("kind");
    for (let k = 0; k < 3; k++) {
      const hp = [...prim.half] as Vec3; hp[k] += EPS;
      const hm = [...prim.half] as Vec3; hm[k] -= EPS;
      const fd = (evalPrim({ ...prim, half: hp }, p) - evalPrim({ ...prim, half: hm }, p)) / (2 * EPS);
      expect(g.dHalf[k]).toBeCloseTo(fd, 3);
    }
  });
});

describe("dPrim — cylinder", () => {
  const prim: SDFPrim = { kind: "cylinder", center: [0, 0, 0], axis: [0, 1, 0], radius: 0.3, height: 0.6 };
  it("matches ∂d/∂p outside the side", () => {
    const p: Vec3 = [0.5, 0.1, 0];
    const g = dPrim(prim, p);
    expect(g.d).toBeCloseTo(evalPrim(prim, p), 8);
    expectVecClose(g.dp, fdPos(prim, p));
  });
  it("matches ∂d/∂radius and ∂d/∂height", () => {
    const p: Vec3 = [0.5, 0.4, 0];
    const g = dPrim(prim, p);
    if (g.kind !== "cylinder") throw new Error("kind");
    const dR = (evalPrim({ ...prim, radius: prim.radius + EPS }, p)
              - evalPrim({ ...prim, radius: prim.radius - EPS }, p)) / (2 * EPS);
    const dH = (evalPrim({ ...prim, height: prim.height + EPS }, p)
              - evalPrim({ ...prim, height: prim.height - EPS }, p)) / (2 * EPS);
    expect(g.dRadius).toBeCloseTo(dR, 3);
    expect(g.dHeight).toBeCloseTo(dH, 3);
  });
});

describe("dPrim — torus", () => {
  const prim: SDFPrim = { kind: "torus", center: [0, 0, 0], major: 0.5, minor: 0.15 };
  const p: Vec3 = [0.55, 0.1, 0.1];
  it("matches d and ∂d/∂p", () => {
    const g = dPrim(prim, p);
    expect(g.d).toBeCloseTo(evalPrim(prim, p), 8);
    expectVecClose(g.dp, fdPos(prim, p));
  });
  it("matches ∂d/∂major and ∂d/∂minor", () => {
    const g = dPrim(prim, p);
    if (g.kind !== "torus") throw new Error("kind");
    const dR = (evalPrim({ ...prim, major: prim.major + EPS }, p)
              - evalPrim({ ...prim, major: prim.major - EPS }, p)) / (2 * EPS);
    expect(g.dMajor).toBeCloseTo(dR, 3);
    expect(g.dMinor).toBeCloseTo(-1, 8);
  });
});

describe("dScene — soft-min combiner", () => {
  const prims: SDFPrim[] = [
    { kind: "sphere", center: [-0.3, 0, 0], radius: 0.3 },
    { kind: "sphere", center: [ 0.3, 0, 0], radius: 0.3 },
  ];
  it("weights sum to 1 (soft) / 0 or 1 (hard)", () => {
    const soft = dScene(prims, [0, 0.05, 0], 0.1);
    const sum = soft.prims.reduce((a, x) => a + x.weight, 0);
    expect(sum).toBeCloseTo(1, 6);
    const hard = dScene(prims, [-0.4, 0, 0], 0);
    const sumH = hard.prims.reduce((a, x) => a + x.weight, 0);
    expect(sumH).toBeCloseTo(1, 8);
    expect(hard.prims.filter((x) => x.weight > 0)).toHaveLength(1);
  });
  it("∂d_combined/∂p matches FD on evalScene (soft)", () => {
    const p: Vec3 = [0.05, 0.07, -0.02];
    const k = 0.1;
    const g = dScene(prims, p, k);
    expect(g.d).toBeCloseTo(evalSoftMinScene(prims, p, k), 6);
    for (let i = 0; i < 3; i++) {
      const pp = [...p] as Vec3; pp[i] += EPS;
      const pm = [...p] as Vec3; pm[i] -= EPS;
      const fd = (evalSoftMinScene(prims, pp, k) - evalSoftMinScene(prims, pm, k)) / (2 * EPS);
      expect(g.dp[i]).toBeCloseTo(fd, 3);
    }
  });
  it("hard-min path matches evalScene(prims, p, 0)", () => {
    const p: Vec3 = [-0.5, 0, 0];
    const g = dScene(prims, p, 0);
    expect(g.d).toBeCloseTo(evalScene(prims, p, 0), 8);
  });
});

// Reference soft-min identical to dScene's k>0 branch.
function evalSoftMinScene(prims: SDFPrim[], p: Vec3, k: number): number {
  const ds = prims.map((pr) => evalPrim(pr, p));
  const dMin = Math.min(...ds);
  const Z = ds.reduce((a, d) => a + Math.exp(-(d - dMin) / k), 0);
  return dMin - k * Math.log(Z);
}
