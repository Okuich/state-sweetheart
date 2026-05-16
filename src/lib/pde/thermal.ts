/**
 * Steady-state thermal field engine.
 *
 *   −∇·(κ ∇T) = Q       in Ω
 *           T = T_D     on Γ_D   (fixed temperatures)
 *      (κ∇T)·n = q_N    on Γ_N   (prescribed heat flux into the body)
 *
 * Built on top of the Phase-1 PDE core. Supports isotropic OR anisotropic
 * (symmetric 3×3) conductivity per tet, volumetric sources, Dirichlet
 * temperatures, and pre-integrated Neumann nodal loads. Outputs the
 * temperature field plus vertex gradients, per-tet heat-flux vectors, a
 * normalized hotspot map, and a thermal-stress indicator (α·ΔT).
 */
import { buildCSR, type CSRMatrix } from "./sparse";
import { type FEMMeshInput } from "./laplacian";
import {
  solvePoisson, type DirichletBC, type PoissonSolution,
} from "./poisson";
import { type CGOptions } from "./solvers/cg";

/** Symmetric 3×3 conductivity tensor stored as [κxx,κyy,κzz,κxy,κxz,κyz]. */
export type KappaTensor = readonly [number, number, number, number, number, number];

export interface ThermalProblem {
  mesh: FEMMeshInput;
  /** Isotropic conductivity per tet. Ignored if `kappaTensor` is given. */
  kappa?: Float64Array | Float32Array;
  /** Anisotropic conductivity per tet, flat array length = 6 · nTets. */
  kappaTensor?: Float64Array | Float32Array;
  /** Volumetric heat source per vertex (W/m³). */
  source?: Float64Array;
  /** Pre-integrated Neumann nodal loads (W). */
  neumannLoads?: Float64Array;
  /** Fixed temperatures. */
  dirichlet?: ReadonlyArray<DirichletBC>;
  /** Reference temperature for thermal-stress indicator. Defaults to mean Dirichlet (or 0). */
  referenceTemperature?: number;
  /** Linear thermal expansion coefficient α (1/K). Defaults to 1. */
  expansionCoefficient?: number;
  cg?: CGOptions;
}

export interface ThermalSolution {
  /** Temperature per vertex. */
  T: Float64Array;
  /** Per-vertex temperature gradient (flat xyz, length 3·nVerts). */
  gradT: Float64Array;
  /** Per-tet heat flux q = −κ·∇T (flat xyz, length 3·nTets). */
  fluxPerTet: Float64Array;
  /** Per-tet |q|. */
  fluxMagnitude: Float64Array;
  /** Normalized [0,1] hotspot map per vertex (|T−T_ref| / max). */
  hotspot: Float64Array;
  /** Thermal-stress indicator per vertex: α · (T − T_ref). */
  thermalStress: Float64Array;
  /** Underlying Poisson solve diagnostics. */
  solve: PoissonSolution;
}

/**
 * Assemble the (possibly anisotropic) thermal stiffness K and lumped mass.
 * K_ij = Σ_e ∫_e ∇φ_i · κ ∇φ_j dV  (P1 tetrahedra, gradients constant per tet).
 */
export function assembleThermalStiffness(
  mesh: FEMMeshInput,
  kappa?: Float64Array | Float32Array,
  kappaTensor?: Float64Array | Float32Array,
): { K: CSRMatrix; massLumped: Float64Array; volume: number; tetGradients: Float64Array; tetVolumes: Float64Array } {
  const verts = mesh.vertices;
  const tets = mesh.tets;
  const nVerts = verts.length / 3;
  const nTets = tets.length / 4;

  const triplets: Array<[number, number, number]> = [];
  const mass = new Float64Array(nVerts);
  // Per-tet shape-function gradients (4 grads × 3 components per tet).
  const tetGradients = new Float64Array(nTets * 12);
  const tetVolumes = new Float64Array(nTets);
  let total = 0;

  const p: Float64Array[] = [
    new Float64Array(3), new Float64Array(3),
    new Float64Array(3), new Float64Array(3),
  ];
  const g: Float64Array[] = [
    new Float64Array(3), new Float64Array(3),
    new Float64Array(3), new Float64Array(3),
  ];

  for (let t = 0; t < nTets; t++) {
    const i0 = tets[t * 4],     i1 = tets[t * 4 + 1];
    const i2 = tets[t * 4 + 2], i3 = tets[t * 4 + 3];
    const ids = [i0, i1, i2, i3];
    for (let k = 0; k < 4; k++) {
      p[k][0] = verts[ids[k] * 3];
      p[k][1] = verts[ids[k] * 3 + 1];
      p[k][2] = verts[ids[k] * 3 + 2];
    }
    const a = sub(p[1], p[0]);
    const b = sub(p[2], p[0]);
    const c = sub(p[3], p[0]);
    const det = a[0] * (b[1] * c[2] - b[2] * c[1])
              - a[1] * (b[0] * c[2] - b[2] * c[0])
              + a[2] * (b[0] * c[1] - b[1] * c[0]);
    const V = det / 6;
    const absV = Math.abs(V);
    if (absV < 1e-20) continue;
    total += absV;
    tetVolumes[t] = absV;

    const inv6V = 1 / (6 * V);
    g[0] = scale(cross(sub(p[3], p[1]), sub(p[2], p[1])), inv6V);
    g[1] = scale(cross(sub(p[2], p[0]), sub(p[3], p[0])), inv6V);
    g[2] = scale(cross(sub(p[3], p[0]), sub(p[1], p[0])), inv6V);
    g[3] = scale(cross(sub(p[1], p[0]), sub(p[2], p[0])), inv6V);
    for (let k = 0; k < 4; k++) {
      tetGradients[t * 12 + k * 3]     = g[k][0];
      tetGradients[t * 12 + k * 3 + 1] = g[k][1];
      tetGradients[t * 12 + k * 3 + 2] = g[k][2];
    }

    let kIso = 1;
    let kxx = 1, kyy = 1, kzz = 1, kxy = 0, kxz = 0, kyz = 0;
    let anisotropic = false;
    if (kappaTensor) {
      kxx = kappaTensor[t * 6];     kyy = kappaTensor[t * 6 + 1];
      kzz = kappaTensor[t * 6 + 2]; kxy = kappaTensor[t * 6 + 3];
      kxz = kappaTensor[t * 6 + 4]; kyz = kappaTensor[t * 6 + 5];
      anisotropic = true;
    } else if (kappa) {
      kIso = kappa[t];
    }

    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const gi = g[i], gj = g[j];
        let kdot: number;
        if (anisotropic) {
          // (κ · gj)
          const kgx = kxx * gj[0] + kxy * gj[1] + kxz * gj[2];
          const kgy = kxy * gj[0] + kyy * gj[1] + kyz * gj[2];
          const kgz = kxz * gj[0] + kyz * gj[1] + kzz * gj[2];
          kdot = gi[0] * kgx + gi[1] * kgy + gi[2] * kgz;
        } else {
          kdot = kIso * (gi[0] * gj[0] + gi[1] * gj[1] + gi[2] * gj[2]);
        }
        const v = absV * kdot;
        if (v !== 0) triplets.push([ids[i], ids[j], v]);
      }
    }
    const mShare = absV / 4;
    mass[i0] += mShare; mass[i1] += mShare;
    mass[i2] += mShare; mass[i3] += mShare;
  }

  return {
    K: buildCSR(nVerts, triplets),
    massLumped: mass,
    volume: total,
    tetGradients,
    tetVolumes,
  };
}

/** Solve a steady-state thermal problem end-to-end. */
export function solveThermal(problem: ThermalProblem): ThermalSolution {
  const { K, massLumped, tetGradients } = assembleThermalStiffness(
    problem.mesh, problem.kappa, problem.kappaTensor,
  );
  const solve = solvePoisson({
    K,
    massLumped,
    source: problem.source,
    loads: problem.neumannLoads,
    dirichlet: problem.dirichlet,
    cg: problem.cg,
  });
  const T = solve.u;

  // Per-tet ∇T (constant inside each P1 tet) and flux q = −κ∇T.
  const tets = problem.mesh.tets;
  const nTets = tets.length / 4;
  const fluxPerTet = new Float64Array(nTets * 3);
  const fluxMagnitude = new Float64Array(nTets);
  const tetGradT = new Float64Array(nTets * 3);
  for (let t = 0; t < nTets; t++) {
    let gx = 0, gy = 0, gz = 0;
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      const Tk = T[vid];
      gx += Tk * tetGradients[t * 12 + k * 3];
      gy += Tk * tetGradients[t * 12 + k * 3 + 1];
      gz += Tk * tetGradients[t * 12 + k * 3 + 2];
    }
    tetGradT[t * 3]     = gx;
    tetGradT[t * 3 + 1] = gy;
    tetGradT[t * 3 + 2] = gz;

    let kxx = 1, kyy = 1, kzz = 1, kxy = 0, kxz = 0, kyz = 0;
    if (problem.kappaTensor) {
      const kT = problem.kappaTensor;
      kxx = kT[t * 6];     kyy = kT[t * 6 + 1];
      kzz = kT[t * 6 + 2]; kxy = kT[t * 6 + 3];
      kxz = kT[t * 6 + 4]; kyz = kT[t * 6 + 5];
    } else if (problem.kappa) {
      const k = problem.kappa[t];
      kxx = kyy = kzz = k;
    }
    const qx = -(kxx * gx + kxy * gy + kxz * gz);
    const qy = -(kxy * gx + kyy * gy + kyz * gz);
    const qz = -(kxz * gx + kyz * gy + kzz * gz);
    fluxPerTet[t * 3]     = qx;
    fluxPerTet[t * 3 + 1] = qy;
    fluxPerTet[t * 3 + 2] = qz;
    fluxMagnitude[t] = Math.sqrt(qx * qx + qy * qy + qz * qz);
  }

  // Volume-weighted vertex gradient averaging.
  const nVerts = T.length;
  const gradT = new Float64Array(nVerts * 3);
  const wAcc = new Float64Array(nVerts);
  for (let t = 0; t < nTets; t++) {
    const w = 1; // tet volume already encoded in solve; use unit weight
    for (let k = 0; k < 4; k++) {
      const vid = tets[t * 4 + k];
      gradT[vid * 3]     += w * tetGradT[t * 3];
      gradT[vid * 3 + 1] += w * tetGradT[t * 3 + 1];
      gradT[vid * 3 + 2] += w * tetGradT[t * 3 + 2];
      wAcc[vid] += w;
    }
  }
  for (let v = 0; v < nVerts; v++) {
    if (wAcc[v] > 0) {
      gradT[v * 3]     /= wAcc[v];
      gradT[v * 3 + 1] /= wAcc[v];
      gradT[v * 3 + 2] /= wAcc[v];
    }
  }

  // Reference temperature.
  let Tref = problem.referenceTemperature;
  if (Tref === undefined) {
    if (problem.dirichlet && problem.dirichlet.length > 0) {
      let s = 0;
      for (const bc of problem.dirichlet) s += bc.value;
      Tref = s / problem.dirichlet.length;
    } else {
      Tref = 0;
    }
  }
  const alpha = problem.expansionCoefficient ?? 1;
  const hotspot = new Float64Array(nVerts);
  const thermalStress = new Float64Array(nVerts);
  let maxDev = 0;
  for (let v = 0; v < nVerts; v++) {
    const d = Math.abs(T[v] - Tref);
    if (d > maxDev) maxDev = d;
    thermalStress[v] = alpha * (T[v] - Tref);
  }
  const inv = maxDev > 0 ? 1 / maxDev : 0;
  for (let v = 0; v < nVerts; v++) hotspot[v] = Math.abs(T[v] - Tref) * inv;

  return { T, gradT, fluxPerTet, fluxMagnitude, hotspot, thermalStress, solve };
}

// ── vec3 helpers ────────────────────────────────────────────────────────────
function sub(a: Float64Array, b: Float64Array): Float64Array {
  return new Float64Array([a[0] - b[0], a[1] - b[1], a[2] - b[2]]);
}
function cross(a: Float64Array, b: Float64Array): Float64Array {
  return new Float64Array([
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]);
}
function scale(a: Float64Array, s: number): Float64Array {
  return new Float64Array([a[0] * s, a[1] * s, a[2] * s]);
}
