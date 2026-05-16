import { describe, it, expect } from "vitest";
import { solvePotentialFlow, makeVelocitySampler } from "./potentialFlow";

function buildBarMesh(n: number) {
  const verts: number[] = [];
  const idx = (i: number, j: number, k: number) => (k * 2 + j) * (n + 1) + i;
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

describe("Potential-flow engine", () => {
  it("uniform flow: φ = U·x, v = (U,0,0), Cp = 0 at reference", () => {
    const n = 6;
    const mesh = buildBarMesh(n);
    const U = 2.5;
    const dirichlet: { index: number; value: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: U });
    }
    const sol = solvePotentialFlow({
      mesh, dirichlet, referenceSpeed: U,
      cg: { tol: 1e-10, maxIter: 2000 },
    });
    expect(sol.solve.result.converged).toBe(true);
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      expect(Math.abs(sol.phi[v] - U * x)).toBeLessThan(1e-6);
    }
    for (let t = 0; t < sol.speedTet.length; t++) {
      expect(Math.abs(sol.velocityPerTet[t * 3] - U)).toBeLessThan(1e-6);
      expect(Math.abs(sol.velocityPerTet[t * 3 + 1])).toBeLessThan(1e-6);
      expect(Math.abs(sol.velocityPerTet[t * 3 + 2])).toBeLessThan(1e-6);
      expect(Math.abs(sol.speedTet[t] - U)).toBeLessThan(1e-6);
    }
    for (let v = 0; v < sol.cp.length; v++) {
      expect(Math.abs(sol.cp[v])).toBeLessThan(1e-6);
    }
    expect(sol.referenceSpeed).toBeCloseTo(U, 6);
  });

  it("pinGauge alone with zero flux yields φ ≡ 0", () => {
    const mesh = buildBarMesh(3);
    const sol = solvePotentialFlow({ mesh, pinGauge: true });
    for (let v = 0; v < sol.phi.length; v++) {
      expect(Math.abs(sol.phi[v])).toBeLessThan(1e-9);
    }
  });

  it("velocity sampler returns the per-tet vector near a centroid", () => {
    const mesh = buildBarMesh(4);
    const U = 1.0;
    const dirichlet: { index: number; value: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: U });
    }
    const sol = solvePotentialFlow({ mesh, dirichlet });
    const sample = makeVelocitySampler(mesh, sol.velocityPerTet);
    const v = sample(0.5, 0.5, 0.5);
    expect(v).not.toBeNull();
    expect(Math.abs(v![0] - U)).toBeLessThan(1e-6);
  });
});
