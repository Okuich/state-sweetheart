import { describe, expect, it } from "vitest";
import { buildCSR, type CSRMatrix } from "./sparse";
import { cg } from "./solvers/cg";
import {
  buildPODBasis, buildReducedModel, projectToReduced, liftToFull, solveReduced,
} from "./rom";

/** 1-D Laplacian with Dirichlet pins at the endpoints (n interior unknowns). */
function lap1D(n: number): CSRMatrix {
  const t: Array<[number, number, number]> = [];
  for (let i = 0; i < n; i++) {
    t.push([i, i, 2]);
    if (i > 0) t.push([i, i - 1, -1]);
    if (i < n - 1) t.push([i, i + 1, -1]);
  }
  return buildCSR(n, t);
}

describe("ROM / POD reduced-order surrogate", () => {
  const n = 40;
  const K = lap1D(n);

  function snapshotForFreq(omega: number): Float64Array {
    const f = new Float64Array(n);
    for (let i = 0; i < n; i++) f[i] = Math.sin((omega * (i + 1) * Math.PI) / (n + 1));
    const u = new Float64Array(n);
    cg(K, f, u, { tol: 1e-12, maxIter: 2 * n });
    return u;
  }

  const snaps = [1, 2, 3, 4].map(snapshotForFreq);
  const U = new Float64Array(n * snaps.length);
  snaps.forEach((u, j) => U.set(u, j * n));

  it("basis modes are orthonormal", () => {
    const basis = buildPODBasis(U, n, snaps.length);
    expect(basis.k).toBeGreaterThan(0);
    for (let i = 0; i < basis.k; i++) {
      let nrm = 0;
      for (let p = 0; p < n; p++) nrm += basis.modes[i * n + p] ** 2;
      expect(nrm).toBeCloseTo(1, 8);
    }
  });

  it("project then lift reproduces snapshots in-basis", () => {
    const basis = buildPODBasis(U, n, snaps.length);
    const a = projectToReduced(basis, snaps[0]);
    const back = liftToFull(basis, a);
    let err = 0;
    for (let i = 0; i < n; i++) err += (back[i] - snaps[0][i]) ** 2;
    expect(Math.sqrt(err)).toBeLessThan(1e-8);
  });

  it("Galerkin solve matches full-order CG on in-distribution RHS", () => {
    const basis = buildPODBasis(U, n, snaps.length);
    const model = buildReducedModel(K, basis);
    const f = new Float64Array(n);
    for (let i = 0; i < n; i++) f[i] = Math.sin((2 * (i + 1) * Math.PI) / (n + 1));
    const uFull = new Float64Array(n);
    cg(K, f, uFull, { tol: 1e-12 });
    const { u } = solveReduced(model, f);
    let err = 0, ref = 0;
    for (let i = 0; i < n; i++) { err += (u[i] - uFull[i]) ** 2; ref += uFull[i] ** 2; }
    expect(Math.sqrt(err / ref)).toBeLessThan(1e-6);
  });
});
