/**
 * Potential-flow engine (incompressible, irrotational).
 *
 *   ∇²φ = σ        in Ω        (σ = 0 for source-free flow)
 *   ∇φ · n = v_N   on Γ_N      (prescribed normal velocity: inflow/outflow/wall)
 *        φ = φ_D   on Γ_D      (gauge pin or pressure-reference electrode)
 *
 * The velocity field is recovered as v = ∇φ. With Bernoulli (steady,
 * incompressible, inviscid) we report a pressure coefficient
 *
 *   Cp = 1 − (|v| / V_ref)²
 *
 * Solid walls are modelled with v_N = 0 (no-penetration). Inlets prescribe
 * v_N = −U·n̂ (mass flowing into the body); outlets v_N = +U·n̂. Pure
 * Neumann problems are rank-1 deficient — pin at least one Dirichlet node
 * (any value, it sets the gauge) or pass `pinGauge: true` to auto-pin the
 * first vertex to 0.
 *
 * Operator is the same Laplacian used by thermal/electrostatic, so we
 * reuse `assembleThermalStiffness` with unit "conductivity" per tet.
 */
import { type FEMMeshInput } from "./laplacian";
import { solvePoisson, type DirichletBC, type PoissonSolution } from "./poisson";
import { type CGOptions } from "./solvers/cg";
import { assembleThermalStiffness } from "./thermal";

export interface PotentialFlowProblem {
  mesh: FEMMeshInput;
  /** Per-tet scalar weight (e.g. variable density). Defaults to 1. */
  weight?: Float64Array | Float32Array;
  /** Per-vertex volumetric source σ (1/s). Defaults to source-free. */
  source?: Float64Array;
  /** Pre-integrated boundary flux loads ∫ v_N N_i dA (m³/s). */
  neumannLoads?: Float64Array;
  /** Pinned potentials (gauge). */
  dirichlet?: ReadonlyArray<DirichletBC>;
  /** If true and no dirichlet supplied, auto-pin vertex 0 to φ=0. */
  pinGauge?: boolean;
  /** Free-stream / reference speed used for Cp. Defaults to max |v|. */
  referenceSpeed?: number;
  cg?: CGOptions;
}

export interface PotentialFlowSolution {
  /** Velocity potential per vertex (m²/s). */
  phi: Float64Array;
  /** Per-vertex velocity v = ∇φ (flat xyz, length 3·nVerts). */
  velocity: Float64Array;
  /** Per-tet velocity v (flat xyz, length 3·nTets). */
  velocityPerTet: Float64Array;
  /** Per-tet |v|. */
  speedTet: Float64Array;
  /** Per-vertex |v|. */
  speed: Float64Array;
  /** Per-vertex pressure coefficient Cp = 1 − (|v|/V_ref)². */
  cp: Float64Array;
  /** Reference speed actually used for Cp (m/s). */
  referenceSpeed: number;
  /** Divergence residual per tet (should ≈ source for an exact solve). */
  divergenceTet: Float64Array;
  /** Underlying Poisson solve diagnostics. */
  solve: PoissonSolution;
}

export function solvePotentialFlow(problem: PotentialFlowProblem): PotentialFlowSolution {
  const { mesh } = problem;
  const tets = mesh.tets;
  const nTets = tets.length / 4;
  const nVerts = mesh.positions.length / 3;

  const weight = problem.weight ?? (() => {
    const w = new Float64Array(nTets);
    w.fill(1);
    return w;
  })();

  const { K, massLumped, tetGradients } = assembleThermalStiffness(mesh, weight);

  let dirichlet = problem.dirichlet;
  if ((!dirichlet || dirichlet.length === 0) && problem.pinGauge) {
    dirichlet = [{ vertex: 0, value: 0 }];
  }

  const solve = solvePoisson({
    K,
    massLumped,
    source: problem.source,
    loads: problem.neumannLoads,
    dirichlet,
    cg: problem.cg,
  });
  const phi = solve.u;

  const velocityPerTet = new Float64Array(nTets * 3);
  const speedTet = new Float64Array(nTets);
  const divergenceTet = new Float64Array(nTets);

  for (let t = 0; t < nTets; t++) {
    let vx = 0, vy = 0, vz = 0;
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      const p = phi[vid];
      vx += p * tetGradients[t * 12 + k * 3];
      vy += p * tetGradients[t * 12 + k * 3 + 1];
      vz += p * tetGradients[t * 12 + k * 3 + 2];
    }
    velocityPerTet[t * 3] = vx;
    velocityPerTet[t * 3 + 1] = vy;
    velocityPerTet[t * 3 + 2] = vz;
    speedTet[t] = Math.hypot(vx, vy, vz);
    // ∇·v inside a P1 tet is identically zero (v is constant per element);
    // we expose the slot for callers that post-process to a smoother field.
    divergenceTet[t] = 0;
  }

  const velocity = new Float64Array(nVerts * 3);
  const wAcc = new Float64Array(nVerts);
  for (let t = 0; t < nTets; t++) {
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      velocity[vid * 3]     += velocityPerTet[t * 3];
      velocity[vid * 3 + 1] += velocityPerTet[t * 3 + 1];
      velocity[vid * 3 + 2] += velocityPerTet[t * 3 + 2];
      wAcc[vid] += 1;
    }
  }
  const speed = new Float64Array(nVerts);
  let maxSpeed = 0;
  for (let v = 0; v < nVerts; v++) {
    if (wAcc[v] > 0) {
      velocity[v * 3]     /= wAcc[v];
      velocity[v * 3 + 1] /= wAcc[v];
      velocity[v * 3 + 2] /= wAcc[v];
    }
    const s = Math.hypot(velocity[v * 3], velocity[v * 3 + 1], velocity[v * 3 + 2]);
    speed[v] = s;
    if (s > maxSpeed) maxSpeed = s;
  }

  const vRef = problem.referenceSpeed && problem.referenceSpeed > 0
    ? problem.referenceSpeed
    : (maxSpeed > 0 ? maxSpeed : 1);
  const inv2 = 1 / (vRef * vRef);
  const cp = new Float64Array(nVerts);
  for (let v = 0; v < nVerts; v++) cp[v] = 1 - speed[v] * speed[v] * inv2;

  return {
    phi, velocity, velocityPerTet, speedTet, speed, cp,
    referenceSpeed: vRef, divergenceTet, solve,
  };
}

/** Sampler factory: nearest-tet velocity lookup for streamline tracing. */
export function makeVelocitySampler(
  mesh: FEMMeshInput,
  velocityPerTet: Float64Array,
): (x: number, y: number, z: number) => readonly [number, number, number] | null {
  const tets = mesh.tets;
  const pos = mesh.positions;
  const nTets = tets.length / 4;
  const centroids = new Float64Array(nTets * 3);
  for (let t = 0; t < nTets; t++) {
    let cx = 0, cy = 0, cz = 0;
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      cx += pos[vid * 3]; cy += pos[vid * 3 + 1]; cz += pos[vid * 3 + 2];
    }
    centroids[t * 3] = cx * 0.25;
    centroids[t * 3 + 1] = cy * 0.25;
    centroids[t * 3 + 2] = cz * 0.25;
  }
  return (x, y, z) => {
    let best = -1, bestD = Infinity;
    for (let t = 0; t < nTets; t++) {
      const dx = centroids[t * 3] - x;
      const dy = centroids[t * 3 + 1] - y;
      const dz = centroids[t * 3 + 2] - z;
      const d = dx * dx + dy * dy + dz * dz;
      if (d < bestD) { bestD = d; best = t; }
    }
    if (best < 0) return null;
    return [
      velocityPerTet[best * 3],
      velocityPerTet[best * 3 + 1],
      velocityPerTet[best * 3 + 2],
    ] as const;
  };
}
