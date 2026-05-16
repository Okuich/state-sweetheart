import { describe, it, expect } from "vitest";
import { solveThermal } from "./thermal";

/**
 * Build a 1D "bar" as a stack of tets along x ∈ [0,1] with unit y,z extent.
 * Each slab [x_i, x_{i+1}] × [0,1]² is split into 6 tets (standard cube-to-tet).
 */
function buildBarMesh(n: number) {
  const verts: number[] = [];
  const idx = (i: number, j: number, k: number) =>
    (k * 2 + j) * (n + 1) + i;
  for (let k = 0; k < 2; k++) {
    for (let j = 0; j < 2; j++) {
      for (let i = 0; i <= n; i++) {
        verts.push(i / n, j, k);
      }
    }
  }
  // 6-tet decomposition of each cube (Delaunay-ish, consistent winding).
  const tets: number[] = [];
  for (let i = 0; i < n; i++) {
    const v000 = idx(i,     0, 0), v100 = idx(i + 1, 0, 0);
    const v010 = idx(i,     1, 0), v110 = idx(i + 1, 1, 0);
    const v001 = idx(i,     0, 1), v101 = idx(i + 1, 0, 1);
    const v011 = idx(i,     1, 1), v111 = idx(i + 1, 1, 1);
    tets.push(
      v000, v100, v010, v001,
      v100, v110, v010, v111,
      v100, v010, v001, v111,
      v010, v011, v001, v111,
      v100, v101, v001, v111,
      v010, v100, v110, v111,
    );
  }
  return {
    vertices: new Float64Array(verts),
    tets:     new Uint32Array(tets),
    nx: n + 1,
  };
}

describe("Thermal field engine", () => {
  it("recovers a linear conduction profile T(x) = x with hot/cold ends", () => {
    const n = 8;
    const mesh = buildBarMesh(n);
    const dirichlet: { index: number; value: number }[] = [];
    // Pin every x=0 vertex to 0 and every x=1 vertex to 1.
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: 1 });
    }
    const sol = solveThermal({ mesh, dirichlet });
    expect(sol.solve.result.converged).toBe(true);
    // Interior nodes must satisfy T ≈ x (Laplace solution is linear).
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      expect(Math.abs(sol.T[v] - x)).toBeLessThan(1e-6);
    }
    // Flux magnitude must be ~1 in every tet, pointing in −x.
    for (let t = 0; t < sol.fluxMagnitude.length; t++) {
      expect(sol.fluxMagnitude[t]).toBeGreaterThan(0.99);
      expect(sol.fluxMagnitude[t]).toBeLessThan(1.01);
      expect(sol.fluxPerTet[t * 3]).toBeLessThan(-0.99);
    }
  });

  it("hotspot map peaks at the heated end", () => {
    const n = 6;
    const mesh = buildBarMesh(n);
    const dirichlet: { index: number; value: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: 100 });
    }
    const sol = solveThermal({ mesh, dirichlet, referenceTemperature: 0 });
    // Vertex at x=1 should have hotspot = 1 (max).
    const hotEnd = sol.hotspot.findIndex((h, v) =>
      h > 0.99 && Math.abs(mesh.vertices[v * 3] - 1) < 1e-9,
    );
    expect(hotEnd).toBeGreaterThanOrEqual(0);
    // Cold end should have hotspot ≈ 0.
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      if (Math.abs(mesh.vertices[v * 3]) < 1e-9) {
        expect(sol.hotspot[v]).toBeLessThan(1e-6);
      }
    }
  });

  it("anisotropic conductivity steers flux direction", () => {
    const n = 4;
    const mesh = buildBarMesh(n);
    // Make conduction strong in x, weak in y/z — but apply the gradient in x.
    // Flux should align with x.
    const nTets = mesh.tets.length / 4;
    const kappaTensor = new Float64Array(nTets * 6);
    for (let t = 0; t < nTets; t++) {
      kappaTensor[t * 6]     = 10;   // κxx
      kappaTensor[t * 6 + 1] = 0.1;  // κyy
      kappaTensor[t * 6 + 2] = 0.1;  // κzz
    }
    const dirichlet: { index: number; value: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (Math.abs(x) < 1e-9)          dirichlet.push({ index: v, value: 0 });
      else if (Math.abs(x - 1) < 1e-9) dirichlet.push({ index: v, value: 1 });
    }
    const sol = solveThermal({ mesh, kappaTensor, dirichlet });
    expect(sol.solve.result.converged).toBe(true);
    // |qx| dominates.
    for (let t = 0; t < nTets; t++) {
      const qx = Math.abs(sol.fluxPerTet[t * 3]);
      const qy = Math.abs(sol.fluxPerTet[t * 3 + 1]);
      const qz = Math.abs(sol.fluxPerTet[t * 3 + 2]);
      expect(qx).toBeGreaterThan(qy * 10 - 1e-6);
      expect(qx).toBeGreaterThan(qz * 10 - 1e-6);
    }
  });
});
