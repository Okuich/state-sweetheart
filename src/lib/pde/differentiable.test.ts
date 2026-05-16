import { describe, it, expect } from "vitest";
import { solveThermal } from "./thermal";
import {
  differentiateThermal, targetTemperatureLoss, fluxMagnitudeLoss,
  inverseDesignKappa,
} from "./differentiable";

/** Same 1-D bar mesh used by `thermal.test.ts`. */
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
  return {
    vertices: new Float64Array(verts),
    tets: new Uint32Array(tets),
  };
}

function makeDirichlet(mesh: { vertices: Float64Array }, hot: number, cold: number) {
  const out: { index: number; value: number }[] = [];
  for (let v = 0; v < mesh.vertices.length / 3; v++) {
    const x = mesh.vertices[v * 3];
    if (Math.abs(x) < 1e-9) out.push({ index: v, value: cold });
    else if (Math.abs(x - 1) < 1e-9) out.push({ index: v, value: hot });
  }
  return out;
}

/**
 * Centered finite difference for one perturbed parameter.
 * Loss is `Σ ½ w_p (T_p − T*)²` evaluated against current temperatures.
 */
function fdGradKappa(
  base: () => Float64Array, // returns kappa array we mutate
  evalLoss: () => number,
  t: number,
  h: number,
): number {
  const kappa = base();
  const orig = kappa[t];
  kappa[t] = orig + h;
  const Lp = evalLoss();
  kappa[t] = orig - h;
  const Lm = evalLoss();
  kappa[t] = orig;
  return (Lp - Lm) / (2 * h);
}

describe("Differentiable thermal solves", () => {
  it("adjoint dL/dκ matches centered finite differences (temperature loss)", () => {
    const mesh = buildBarMesh(4);
    const nTets = mesh.tets.length / 4;
    const kappa = new Float64Array(nTets);
    for (let t = 0; t < nTets; t++) kappa[t] = 1 + 0.1 * (t % 5); // mildly heterogeneous
    const dirichlet = makeDirichlet(mesh, 1, 0);

    // Probes: a few interior nodes targeting an arbitrary profile.
    const probes: { index: number; target: number; weight?: number }[] = [];
    for (let v = 0; v < mesh.vertices.length / 3; v++) {
      const x = mesh.vertices[v * 3];
      if (x > 0.1 && x < 0.9) probes.push({ index: v, target: x * x, weight: 1 });
    }

    const fwd = solveThermal({ mesh, kappa, dirichlet, cg: { tol: 1e-12, maxIter: 5000 } });
    const { dLdT } = targetTemperatureLoss(fwd.T, probes);
    const grads = differentiateThermal(
      { mesh, kappa, dirichlet, cg: { tol: 1e-12, maxIter: 5000 } },
      { dLdT },
      fwd,
    );

    const evalLoss = () => {
      const sol = solveThermal({ mesh, kappa, dirichlet, cg: { tol: 1e-12, maxIter: 5000 } });
      return targetTemperatureLoss(sol.T, probes).loss;
    };
    const tCheck = [0, 3, 7, 11, 17, nTets - 1].filter((t) => t < nTets);
    for (const t of tCheck) {
      const fd = fdGradKappa(() => kappa, evalLoss, t, 1e-4);
      const adj = grads.dLdKappa[t];
      const rel = Math.abs(fd - adj) / Math.max(1e-8, Math.abs(fd) + Math.abs(adj));
      expect(rel).toBeLessThan(5e-3);
    }
  });

  it("adjoint dL/dκ matches FD for combined T + flux loss", () => {
    const mesh = buildBarMesh(3);
    const nTets = mesh.tets.length / 4;
    const kappa = new Float64Array(nTets);
    for (let t = 0; t < nTets; t++) kappa[t] = 0.8 + 0.05 * t;
    const dirichlet = makeDirichlet(mesh, 2, 0);

    const probes = [{ index: 5, target: 0.5, weight: 1 }];
    const fluxW = new Float64Array(nTets);
    for (let t = 0; t < nTets; t++) fluxW[t] = 0.1;

    const fwd = solveThermal({ mesh, kappa, dirichlet, cg: { tol: 1e-12, maxIter: 5000 } });
    const tLoss = targetTemperatureLoss(fwd.T, probes);
    const qLoss = fluxMagnitudeLoss(fwd.fluxPerTet, fluxW);

    const grads = differentiateThermal(
      { mesh, kappa, dirichlet, cg: { tol: 1e-12, maxIter: 5000 } },
      { dLdT: tLoss.dLdT, dLdFluxPerTet: qLoss.dLdFluxPerTet },
      fwd,
    );

    const evalLoss = () => {
      const s = solveThermal({ mesh, kappa, dirichlet, cg: { tol: 1e-12, maxIter: 5000 } });
      const a = targetTemperatureLoss(s.T, probes).loss;
      const b = fluxMagnitudeLoss(s.fluxPerTet, fluxW).loss;
      return a + b;
    };
    for (const t of [0, 2, 5, 9, nTets - 1].filter((t) => t < nTets)) {
      const fd = fdGradKappa(() => kappa, evalLoss, t, 1e-4);
      const adj = grads.dLdKappa[t];
      const rel = Math.abs(fd - adj) / Math.max(1e-8, Math.abs(fd) + Math.abs(adj));
      expect(rel).toBeLessThan(5e-3);
    }
  });

  it("gradient descent on κ recovers a target temperature profile", () => {
    const mesh = buildBarMesh(4);
    const nV = mesh.vertices.length / 3;
    const nTets = mesh.tets.length / 4;
    // Pin both ends to 0 and drive interior with a volumetric source so the
    // solution genuinely depends on κ (with zero source the BCs alone fix T).
    const dirichlet = makeDirichlet(mesh, 0, 0);
    const source = new Float64Array(nV).fill(1);

    const kappaGT = new Float64Array(nTets);
    for (let t = 0; t < nTets; t++) kappaGT[t] = 0.4 + 1.2 * (t % 5) / 5;
    const truth = solveThermal({
      mesh, kappa: kappaGT, source, dirichlet, cg: { tol: 1e-12, maxIter: 5000 },
    });
    const probes: { index: number; target: number; weight: number }[] = [];
    for (let v = 0; v < nV; v++) {
      const x = mesh.vertices[v * 3];
      if (x > 0.05 && x < 0.95) probes.push({ index: v, target: truth.T[v], weight: 1 });
    }

    const kappa0 = new Float64Array(nTets).fill(1);
    const initialLoss = targetTemperatureLoss(
      solveThermal({ mesh, kappa: kappa0, source, dirichlet, cg: { tol: 1e-12 } }).T,
      probes,
    ).loss;
    expect(initialLoss).toBeGreaterThan(1e-6);

    const result = inverseDesignKappa(
      { mesh, kappa: kappa0, source, dirichlet, cg: { tol: 1e-10, maxIter: 5000 } },
      probes,
      { steps: 200, learningRate: 0.5 },
    );
    const finalLoss = result.history[result.history.length - 1].loss;
    // The kappa→T map is many-to-one on this bar (only the x-profile of κ
    // matters), so we don't expect exact recovery — only that the adjoint
    // gradient drives a non-trivial descent.
    expect(finalLoss).toBeLessThan(initialLoss * 0.5);
  });
});
