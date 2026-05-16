import { describe, it, expect } from "vitest";
import {
  assembleFEMLaplacian, assembleFVMLaplacian, solvePoisson, cg, spmv, buildCSR,
} from "./index";

/** Single unit tet [0,0,0]-[1,0,0]-[0,1,0]-[0,0,1], volume 1/6. */
function unitTet() {
  return {
    vertices: new Float64Array([0,0,0, 1,0,0, 0,1,0, 0,0,1]),
    tets:     new Uint32Array([0,1,2,3]),
  };
}

describe("CSR sparse", () => {
  it("sums duplicate triplets", () => {
    const A = buildCSR(2, [[0,0,1],[0,0,2],[1,1,3]]);
    const y = new Float64Array(2);
    spmv(A, new Float64Array([1,1]), y);
    expect(Array.from(y)).toEqual([3, 3]);
  });
});

describe("FEM Laplacian on a unit tet", () => {
  it("produces a symmetric PSD operator with zero row sums", () => {
    const { K, volume } = assembleFEMLaplacian(unitTet());
    expect(volume).toBeCloseTo(1 / 6, 10);
    // Row sums of pure Laplacian must vanish (null space = constants).
    for (let r = 0; r < K.n; r++) {
      let s = 0;
      for (let k = K.rowPtr[r]; k < K.rowPtr[r + 1]; k++) s += K.values[k];
      expect(Math.abs(s)).toBeLessThan(1e-10);
    }
    // Symmetry.
    const dense = new Float64Array(K.n * K.n);
    for (let r = 0; r < K.n; r++) {
      for (let k = K.rowPtr[r]; k < K.rowPtr[r + 1]; k++) {
        dense[r * K.n + K.colIdx[k]] = K.values[k];
      }
    }
    for (let i = 0; i < K.n; i++) {
      for (let j = 0; j < K.n; j++) {
        expect(Math.abs(dense[i * K.n + j] - dense[j * K.n + i])).toBeLessThan(1e-12);
      }
    }
  });
});

describe("Poisson solve on a unit tet (Laplace, all-Dirichlet)", () => {
  it("recovers the linear harmonic field u(x,y,z) = x", () => {
    const mesh = unitTet();
    const { K, massLumped } = assembleFEMLaplacian(mesh);
    // Pin every vertex to its x coordinate. Solution must reproduce it.
    const dirichlet = [
      { index: 0, value: 0 },
      { index: 1, value: 1 },
      { index: 2, value: 0 },
      { index: 3, value: 0 },
    ];
    const { u, result } = solvePoisson({ K, massLumped, dirichlet });
    expect(result.converged).toBe(true);
    expect(u[0]).toBeCloseTo(0, 8);
    expect(u[1]).toBeCloseTo(1, 8);
    expect(u[2]).toBeCloseTo(0, 8);
    expect(u[3]).toBeCloseTo(0, 8);
  });
});

describe("FVM Laplacian on a 1D-ish strip", () => {
  it("CG solves a 1D Poisson problem to within tolerance", () => {
    const n = 16;
    const A = assembleFVMLaplacian(n, 1, 1, 1 / n);
    const b = new Float64Array(n).fill(1);
    // Pin endpoints to 0.
    const { applyDirichlet } = require("./sparse");
    applyDirichlet(A, b, 0, 0);
    applyDirichlet(A, b, n - 1, 0);
    const x = new Float64Array(n);
    const r = cg(A, b, x, { tol: 1e-10, maxIter: 500 });
    expect(r.converged).toBe(true);
    // Mid-domain value should be the largest (parabolic-ish profile).
    const mid = x[Math.floor(n / 2)];
    expect(mid).toBeGreaterThan(x[1]);
    expect(mid).toBeGreaterThan(x[n - 2]);
  });
});
