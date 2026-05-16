import { describe, it, expect } from "vitest";
import { solveElectrostatic, traceFieldLine } from "./electrostatic";

/** Same bar mesh as thermal tests: x ∈ [0,1], unit y,z, n slabs × 6 tets. */
function buildBarMesh(n: number) {
  const verts: number[] = [];
  const idx = (i: number, j: number, k: number) =>
    (k * 2 + j) * (n + 1) + i;
  for (let k = 0; k < 2; k++) {
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i <= n; i++) verts.push(i / n, j, k);
    }
  }
  const tets: number[] = [];
  for (let i = 0; i < n; i++) {
    const v000 = idx(i, 0, 0), v100 = idx(i + 1, 0, 0);
    const v010 = idx(i, 1, 0), v110 = idx(i + 1, 1, 0);
    const v001 = idx(i, 0, 1), v101 = idx(i + 1, 0, 1);
    const v011 = idx(i, 1, 1), v111 = idx(i + 1, 1, 1);
    tets.push(
      v000, v100, v110, v111,
      v000, v110, v010, v111,
      v000, v010, v011, v111,
      v000, v011, v001, v111,
      v000, v001, v101, v111,
      v000, v101, v100, v111,
    );
  }
  return { vertices: new Float64Array(verts), tets: new Uint32Array(tets) };
}

describe("Electrostatic field engine", () => {
  it("parallel-plate capacitor: linear V, uniform E = −(ΔV/L)x̂", () => {
    const n = 8;
    const mesh = buildBarMesh(n);
    const dirichlet: { index: number; value: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: 10 });
    }
    const sol = solveElectrostatic({
      mesh, dirichlet, cg: { tol: 1e-10, maxIter: 2000 },
    });
    expect(sol.solve.result.converged).toBe(true);
    // V(x) = 10x must be exact at nodes.
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      expect(Math.abs(sol.V[v] - 10 * x)).toBeLessThan(1e-6);
    }
    // E = −∇V = (−10, 0, 0) uniform across all tets.
    for (let t = 0; t < sol.fieldMagnitudeTet.length; t++) {
      expect(sol.EPerTet[t * 3]).toBeLessThan(-9.99);
      expect(Math.abs(sol.EPerTet[t * 3 + 1])).toBeLessThan(1e-6);
      expect(Math.abs(sol.EPerTet[t * 3 + 2])).toBeLessThan(1e-6);
      expect(sol.fieldMagnitudeTet[t]).toBeGreaterThan(9.99);
    }
    // Intensity normalized: max should be 1.
    let maxI = 0;
    for (let i = 0; i < sol.intensity.length; i++) maxI = Math.max(maxI, sol.intensity[i]);
    expect(maxI).toBeCloseTo(1, 6);
  });

  it("anisotropic ε rescales D but not E direction", () => {
    const n = 4;
    const mesh = buildBarMesh(n);
    const nTets = mesh.tets.length / 4;
    // εxx = 5, εyy = εzz = 1 — D should be 5× larger than E in x.
    const epsilonTensor = new Float64Array(nTets * 6);
    for (let t = 0; t < nTets; t++) {
      epsilonTensor[t * 6]     = 5;
      epsilonTensor[t * 6 + 1] = 1;
      epsilonTensor[t * 6 + 2] = 1;
    }
    const dirichlet: { index: number; value: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: 1 });
    }
    const sol = solveElectrostatic({ mesh, epsilonTensor, dirichlet });
    expect(sol.solve.result.converged).toBe(true);
    for (let t = 0; t < nTets; t++) {
      const ex = sol.EPerTet[t * 3];
      const dx = sol.DPerTet[t * 3];
      expect(dx / ex).toBeCloseTo(5, 4);
    }
  });

  it("RK4 field-line tracer follows a uniform field", () => {
    // Uniform E = (1,0,0); a particle at the origin should march to (1,0,0)
    // after 100 steps of size 0.01.
    const sampleE = () => [1, 0, 0] as const;
    const line = traceFieldLine([0, 0, 0], sampleE, { stepSize: 0.01, maxSteps: 100 });
    // 101 points (seed + 100 steps), final x ≈ 1.
    expect(line.length).toBe(101 * 3);
    expect(line[line.length - 3]).toBeCloseTo(1, 6);
    expect(line[line.length - 2]).toBeCloseTo(0, 10);
    expect(line[line.length - 1]).toBeCloseTo(0, 10);
  });
});
