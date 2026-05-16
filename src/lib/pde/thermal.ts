/**
 * Steady-state thermal field engine.
 *
 *   −∇·(κ ∇T) = Q       in Ω
 *           T = T_D     on Γ_D   (fixed temperatures)
 *      (κ∇T)·n = q_N    on Γ_N   (prescribed heat flux into the body)
 *
 * Built on top of the Phase-1 PDE core. Supports:
 *   • isotropic conductivity (scalar κ per tet)
 *   • anisotropic conductivity (symmetric 3×3 tensor per tet)
 *   • volumetric heat source Q (W/m³) per vertex
 *   • Dirichlet temperatures and integrated Neumann nodal loads
 *
 * Outputs the temperature field T plus derived quantities — vertex
 * gradients ∇T, per-tet heat-flux vectors q = −κ∇T, a normalized hotspot
 * map, and a thermal-stress indicator (α·ΔT) ready for downstream
 * structural screening.
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
  /** Isotropic conductivity per tet (length = nTets). Mutually exclusive with `kappaTensor`. */
  kappa?: Float64Array | Float32Array;
  /** Anisotropic conductivity per tet, flat array length = 6 · nTets. */
  kappaTensor?: Float64Array | Float32Array;
  /** Volumetric heat source per vertex (W/m³). */
  source?: Float64Array;
  /** Pre-integrated Neumann nodal loads (W). */
  neumannLoads?: Float64Array;
  /** Fixed temperatures. */
  dirichlet?: ReadonlyArray<DirichletBC>;
  /** Reference temperature for thermal-stress indicator (defaults to mean Dirichlet). */
  referenceTemperature?: number;
  /** Linear thermal expansion coefficient α (1/K) for the stress indicator. Defaults 1. */
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
  /** Normalized [0,1] hotspot map per vertex (relative to max |T−T_ref|). */
  hotspot: Float64Array;
  /** Thermal-stress indicator per vertex: α · (T − T_ref). */
  thermalStress: Float64Array;
  /** Underlying Poisson solve diagnostics. */
  solve: PoissonSolution;
}

/**
 * Assemble the (possibly anisotropic) thermal stiffness K such that
 *   K · T ≈ −∫ ∇φ_i · κ ∇φ_j  for vertices i, j.
 */
export function assembleThermalStiffness(
  mesh: FEMMeshInput,
  kappa?: Float64Array | Float32Array,
  kappaTensor?: Float64Array | Float32Array,
): { K: CSRMatrix; massLumped: Float64Array; volume: number } {
  const verts = mesh.vertices;
  const tets = mesh.tets;
  const nVerts = verts.length / 3;
  const nTets = tets.length / 4;

  const triplets: Array<[number, number, number]> = [];
  const mass = new Float64Array(nVerts);
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

    const inv6V = 1 / (6 * V);
    g[0] = scale(cross(sub(p[2], p[1]), sub(p[3], p[1])), inv6V);
    g[1] = scale(cross(sub(p[3], p[0]), sub(p[2], p[0])), inv6V);
    g[2] = scale(cross(sub(p[1], p[0]), sub(p[3], p[0])), inv6V);
    g[3] = scale(cross(sub(p[2], p[0]), sub(p[1], p[0])), inv6V);

    // Conductivity for this tet.
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
          // gi · (κ · gj) with κ symmetric.
          