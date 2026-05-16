/**
 * Electrostatic field engine.
 *
 *   −∇·(ε ∇V) = ρ        in Ω
 *            V = V_D     on Γ_D    (electrode potentials)
 *       (ε∇V)·n = σ_N    on Γ_N    (prescribed surface charge / flux)
 *
 * Built on the Phase-1 PDE core. Supports isotropic OR anisotropic
 * (symmetric 3×3) permittivity per tet, per-vertex charge density ρ
 * (C/m³), Dirichlet electrode potentials, and pre-integrated Neumann
 * surface-charge nodal loads (C).
 *
 * Outputs the potential V, per-vertex electric field E = −∇V, per-tet
 * electric displacement D = ε·E, and a normalized field-intensity map
 * useful for breakdown screening, sensor placement, EMI shielding, and
 * PCB layout. Also exposes a streamline tracer for E-field-line maps.
 *
 * Implementation note: the operator −∇·(σ∇·) is structurally identical
 * to thermal conduction, so we reuse `assembleThermalStiffness` and
 * rebrand the outputs.
 */
import { type FEMMeshInput } from "./laplacian";
import { solvePoisson, type DirichletBC, type PoissonSolution } from "./poisson";
import { type CGOptions } from "./solvers/cg";
import { assembleThermalStiffness } from "./thermal";

export interface ElectrostaticProblem {
  mesh: FEMMeshInput;
  /** Isotropic permittivity per tet (F/m). Ignored if `epsilonTensor` is given. */
  epsilon?: Float64Array | Float32Array;
  /** Anisotropic permittivity tensor per tet, flat length = 6 · nTets. */
  epsilonTensor?: Float64Array | Float32Array;
  /** Per-vertex charge density ρ (C/m³). */
  chargeDensity?: Float64Array;
  /** Pre-integrated surface-charge nodal loads (C). */
  surfaceCharge?: Float64Array;
  /** Electrode potentials. */
  dirichlet?: ReadonlyArray<DirichletBC>;
  cg?: CGOptions;
}

export interface ElectrostaticSolution {
  /** Electric potential per vertex (V). */
  V: Float64Array;
  /** Per-vertex electric field E = −∇V (flat xyz, length 3·nVerts). */
  E: Float64Array;
  /** Per-tet electric displacement D = ε·E (flat xyz, length 3·nTets). */
  DPerTet: Float64Array;
  /** Per-tet |E|. */
  fieldMagnitudeTet: Float64Array;
  /** Per-vertex |E|. */
  fieldMagnitude: Float64Array;
  /** Normalized [0,1] field-intensity map per vertex. */
  intensity: Float64Array;
  /** Underlying Poisson solve diagnostics. */
  solve: PoissonSolution;
}

export function solveElectrostatic(problem: ElectrostaticProblem): ElectrostaticSolution {
  const { K, massLumped, tetGradients } = assembleThermalStiffness(
    problem.mesh, problem.epsilon, problem.epsilonTensor,
  );
  const solve = solvePoisson({
    K,
    massLumped,
    source: problem.chargeDensity,
    loads: problem.surfaceCharge,
    dirichlet: problem.dirichlet,
    cg: problem.cg,
  });
  const V = solve.u;

  const tets = problem.mesh.tets;
  const nTets = tets.length / 4;
  const nVerts = V.length;
  const DPerTet = new Float64Array(nTets * 3);
  const fieldMagnitudeTet = new Float64Array(nTets);
  const tetE = new Float64Array(nTets * 3);

  for (let t = 0; t < nTets; t++) {
    // ∇V per tet (constant within P1 element).
    let gx = 0, gy = 0, gz = 0;
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      const Vk = V[vid];
      gx += Vk * tetGradients[t * 12 + k * 3];
      gy += Vk * tetGradients[t * 12 + k * 3 + 1];
      gz += Vk * tetGradients[t * 12 + k * 3 + 2];
    }
    // E = −∇V
    const ex = -gx, ey = -gy, ez = -gz;
    tetE[t * 3] = ex; tetE[t * 3 + 1] = ey; tetE[t * 3 + 2] = ez;

    // D = ε · E (isotropic or anisotropic).
    let exx = 1, eyy = 1, ezz = 1, exy = 0, exz = 0, eyz = 0;
    if (problem.epsilonTensor) {
      const eT = problem.epsilonTensor;
      exx = eT[t * 6];     eyy = eT[t * 6 + 1];
      ezz = eT[t * 6 + 2]; exy = eT[t * 6 + 3];
      exz = eT[t * 6 + 4]; eyz = eT[t * 6 + 5];
    } else if (problem.epsilon) {
      const e = problem.epsilon[t];
      exx = eyy = ezz = e;
    }
    DPerTet[t * 3]     = exx * ex + exy * ey + exz * ez;
    DPerTet[t * 3 + 1] = exy * ex + eyy * ey + eyz * ez;
    DPerTet[t * 3 + 2] = exz * ex + eyz * ey + ezz * ez;
    fieldMagnitudeTet[t] = Math.sqrt(ex * ex + ey * ey + ez * ez);
  }

  // Vertex-averaged E.
  const E = new Float64Array(nVerts * 3);
  const wAcc = new Float64Array(nVerts);
  for (let t = 0; t < nTets; t++) {
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      E[vid * 3]     += tetE[t * 3];
      E[vid * 3 + 1] += tetE[t * 3 + 1];
      E[vid * 3 + 2] += tetE[t * 3 + 2];
      wAcc[vid] += 1;
    }
  }
  const fieldMagnitude = new Float64Array(nVerts);
  let maxMag = 0;
  for (let v = 0; v < nVerts; v++) {
    if (wAcc[v] > 0) {
      E[v * 3]     /= wAcc[v];
      E[v * 3 + 1] /= wAcc[v];
      E[v * 3 + 2] /= wAcc[v];
    }
    const m = Math.hypot(E[v * 3], E[v * 3 + 1], E[v * 3 + 2]);
    fieldMagnitude[v] = m;
    if (m > maxMag) maxMag = m;
  }
  const intensity = new Float64Array(nVerts);
  const inv = maxMag > 0 ? 1 / maxMag : 0;
  for (let v = 0; v < nVerts; v++) intensity[v] = fieldMagnitude[v] * inv;

  return { V, E, DPerTet, fieldMagnitudeTet, fieldMagnitude, intensity, solve };
}

/**
 * Trace an electric field line starting at `seed` by integrating E with
 * RK4 over the per-tet constant field. `sampleE` returns the field at a
 * world-space point (typically a closure over a kd-tree or octree
 * lookup); we keep tracer signature decoupled so callers can plug in
 * any spatial accelerator. Returns the polyline as a flat xyz array.
 */
export interface FieldLineOptions {
  /** Step size in world units. */
  stepSize: number;
  /** Max number of steps before stopping. */
  maxSteps: number;
  /** Stop when |E| < this. */
  minMagnitude?: number;
  /** Direction: +1 follows E (from + to − electrode), −1 reverses. */
  direction?: 1 | -1;
}

export function traceFieldLine(
  seed: readonly [number, number, number],
  sampleE: (x: number, y: number, z: number) => readonly [number, number, number] | null,
  opts: FieldLineOptions,
): Float64Array {
  const pts: number[] = [seed[0], seed[1], seed[2]];
  const dir = opts.direction ?? 1;
  const minMag = opts.minMagnitude ?? 1e-12;
  let x = seed[0], y = seed[1], z = seed[2];
  for (let s = 0; s < opts.maxSteps; s++) {
    const k1 = sampleE(x, y, z);
    if (!k1) break;
    const m1 = Math.hypot(k1[0], k1[1], k1[2]);
    if (m1 < minMag) break;
    const half = opts.st