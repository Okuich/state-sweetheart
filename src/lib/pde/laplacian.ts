/**
 * Laplace / Poisson operator assembly.
 *
 * Two paths are provided:
 *   • FEM (P1 linear tetrahedra) — primary path, consumes Geometry OS
 *     `OctreeMesh`-style {vertices, tets} buffers directly.
 *   • FVM (two-point flux on a structured grid) — used as a fast preview
 *     path and to seed multigrid coarse operators.
 *
 * Output is a CSR stiffness matrix K such that K · u ≈ −∫ ∇φ · κ ∇u dV
 * (the discrete Laplacian, with `κ` a scalar conductivity). Mass-lumped
 * load assembly lives in `poisson.ts`.
 */
import { buildCSR, type CSRMatrix } from "./sparse";

export interface FEMMeshInput {
  /** Flat xyz, length = 3 · nVerts. */
  vertices: Float64Array | Float32Array;
  /** Flat tet indices, length = 4 · nTets. */
  tets: Uint32Array | Int32Array;
  /** Optional per-tet scalar conductivity κ. Defaults to 1. */
  kappa?: Float64Array | Float32Array;
}

export interface AssemblyResult {
  /** Stiffness matrix K. */
  K: CSRMatrix;
  /** Lumped mass diagonal M (vertex volumes). */
  massLumped: Float64Array;
  /** Total volume (sanity check). */
  volume: number;
}

/**
 * Assemble the FEM Laplacian for P1 tetrahedra.
 *
 * For each tet with vertices v0..v3, the local 4×4 stiffness is
 *   K^e_ij = κ · V · (g_i · g_j)
 * where g_i are the gradients of the barycentric shape functions.
 */
export function assembleFEMLaplacian(mesh: FEMMeshInput): AssemblyResult {
  const verts = mesh.vertices;
  const tets = mesh.tets;
  const nVerts = verts.length / 3;
  const nTets = tets.length / 4;
  const kappa = mesh.kappa;

  const triplets: Array<[number, number, number]> = [];
  const mass = new Float64Array(nVerts);
  let totalVol = 0;

  const p: Float64Array[] = [new Float64Array(3), new Float64Array(3), new Float64Array(3), new Float64Array(3)];
  const g: Float64Array[] = [new Float64Array(3), new Float64Array(3), new Float64Array(3), new Float64Array(3)];

  for (let t = 0; t < nTets; t++) {
    const i0 = tets[t * 4];
    const i1 = tets[t * 4 + 1];
    const i2 = tets[t * 4 + 2];
    const i3 = tets[t * 4 + 3];
    const ids = [i0, i1, i2, i3];
    for (let k = 0; k < 4; k++) {
      p[k][0] = verts[ids[k] * 3];
      p[k][1] = verts[ids[k] * 3 + 1];
      p[k][2] = verts[ids[k] * 3 + 2];
    }
    // Tet volume = (1/6) |det([p1-p0, p2-p0, p3-p0])|.
    const a = sub(p[1], p[0]);
    const b = sub(p[2], p[0]);
    const c = sub(p[3], p[0]);
    const det = a[0] * (b[1] * c[2] - b[2] * c[1])
              - a[1] * (b[0] * c[2] - b[2] * c[0])
              + a[2] * (b[0] * c[1] - b[1] * c[0]);
    const V = det / 6;
    const absV = Math.abs(V);
    if (absV < 1e-20) continue;
    totalVol += absV;

    // Shape-function gradients: g_i = (1/(6V)) · (cross of opposite edges).
    // Using the standard formula for linear tet basis on physical coords.
    const inv6V = 1 / (6 * V);
    g[0] = scale(cross(sub(p[3], p[1]), sub(p[2], p[1])), inv6V);
    g[1] = scale(cross(sub(p[2], p[0]), sub(p[3], p[0])), inv6V);
    g[2] = scale(cross(sub(p[3], p[0]), sub(p[1], p[0])), inv6V);
    g[3] = scale(cross(sub(p[1], p[0]), sub(p[2], p[0])), inv6V);

    const k = (kappa ? kappa[t] : 1) * absV;
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        const v = k * dot3(g[i], g[j]);
        if (v !== 0) triplets.push([ids[i], ids[j], v]);
      }
    }
    const mShare = absV / 4;
    mass[i0] += mShare;
    mass[i1] += mShare;
    mass[i2] += mShare;
    mass[i3] += mShare;
  }

  return { K: buildCSR(nVerts, triplets), massLumped: mass, volume: totalVol };
}

// ── tiny vec3 helpers (local; mirror style used elsewhere in the codebase) ──
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
function dot3(a: Float64Array, b: Float64Array): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * Assemble a Finite-Volume 7-point Laplacian on a uniform structured grid.
 * `nx*ny*nz` cells, spacing `h`. Used as a fast preview / multigrid coarse
 * operator. Boundary cells get reflecting (Neumann-zero) stencils unless
 * Dirichlet is applied later.
 */
export function assembleFVMLaplacian(
  nx: number, ny: number, nz: number, h: number,
): CSRMatrix {
  const n = nx * ny * nz;
  const triplets: Array<[number, number, number]> = [];
  const invH2 = 1 / (h * h);
  const id = (i: number, j: number, k: number) => (k * ny + j) * nx + i;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const c = id(i, j, k);
        let diag = 0;
        const neigh: Array<[number, number, number]> = [
          [i - 1, j, k], [i + 1, j, k],
          [i, j - 1, k], [i, j + 1, k],
          [i, j, k - 1], [i, j, k + 1],
        ];
        for (const [ii, jj, kk] of neigh) {
          if (ii < 0 || ii >= nx || jj < 0 || jj >= ny || kk < 0 || kk >= nz) continue;
          triplets.push([c, id(ii, jj, kk), -invH2]);
          diag += invH2;
        }
        triplets.push([c, c, diag]);
      }
    }
  }
  return buildCSR(n, triplets);
}
