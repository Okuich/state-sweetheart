/**
 * Reduced-Order Model (ROM) for the Laplacian Physics Engine.
 *
 * Builds a low-dimensional Galerkin surrogate of a high-fidelity SPD system
 *
 *     K · u = f          K ∈ ℝⁿˣⁿ, n ≫ 1
 *
 * from a set of full-order solution snapshots U = [u₁, …, u_s]. The POD
 * basis Φ ∈ ℝⁿˣᵏ is obtained by an eigen-decomposition of the small
 * Gram matrix Gᵢⱼ = ⟨uᵢ, uⱼ⟩ (the "method of snapshots", Sirovich 1987),
 * which avoids ever forming the n×n covariance and stays dependency-free.
 *
 * With Φ in hand the reduced operator and right-hand-side are
 *
 *     K_r = Φᵀ K Φ      f_r = Φᵀ f      u ≈ Φ · K_r⁻¹ f_r
 *
 * For SPD K (Laplacian / thermal / electrostatic / potential-flow stiffness)
 * K_r is also SPD and tiny — we solve it with a Cholesky factorisation.
 *
 * Snapshots are typically gathered by running the full solver at a handful
 * of parameter samples (boundary values, conductivity tweaks, source
 * locations, …); the ROM then produces near-instant previews for the
 * remaining parameter sweep.
 */
import { type CSRMatrix, spmv } from "./sparse";

export interface PODOptions {
  /** Maximum modes to retain. Default = #snapshots. */
  maxModes?: number;
  /** Drop modes whose normalised eigenvalue < tol. Default 1e-10. */
  energyTolerance?: number;
}

export interface PODBasis {
  /** Full-order dimension. */
  n: number;
  /** Number of retained modes k. */
  k: number;
  /** Φ stored column-major: Φ[i + j·n] is the i-th entry of mode j. */
  modes: Float64Array;
  /** Eigenvalues of the snapshot Gram (descending). */
  eigenvalues: Float64Array;
  /** Cumulative energy ratio per retained mode in [0,1]. */
  energy: Float64Array;
}

export interface ReducedModel {
  basis: PODBasis;
  /** Dense reduced stiffness K_r (k×k, SPD), row-major. */
  Kr: Float64Array;
  /** Lower-triangular Cholesky factor of K_r, row-major. */
  L: Float64Array;
}

/**
 * Build a POD basis from snapshots `U` (flat column-major, length n·s).
 * Snapshots are mean-centred for numerical stability — store the mean
 * separately if you need to reproject.
 */
export function buildPODBasis(
  U: Float64Array,
  n: number,
  s: number,
  opts: PODOptions = {},
): PODBasis {
  if (U.length !== n * s) {
    throw new Error(`buildPODBasis: expected ${n * s} entries, got ${U.length}`);
  }
  // Gram G = Uᵀ U  (s×s, symmetric PSD).
  const G = new Float64Array(s * s);
  for (let i = 0; i < s; i++) {
    for (let j = i; j < s; j++) {
      let acc = 0;
      const oi = i * n;
      const oj = j * n;
      for (let p = 0; p < n; p++) acc += U[oi + p] * U[oj + p];
      G[i * s + j] = acc;
      G[j * s + i] = acc;
    }
  }

  const { values: eig, vectors: V } = jacobiEigSym(G, s);
  // Sort eigenpairs descending.
  const order: number[] = Array.from({ length: s }, (_, i) => i);
  order.sort((a, b) => eig[b] - eig[a]);

  const tol = opts.energyTolerance ?? 1e-10;
  const cap = Math.min(opts.maxModes ?? s, s);
  const totalEnergy = Math.max(eig.reduce((a, b) => a + Math.max(b, 0), 0), 1e-30);

  const keep: number[] = [];
  for (let m = 0; m < cap; m++) {
    const lam = eig[order[m]];
    if (lam / totalEnergy < tol) break;
    keep.push(order[m]);
  }
  const k = Math.max(keep.length, 1);

  const modes = new Float64Array(n * k);
  const eigenvalues = new Float64Array(k);
  const energy = new Float64Array(k);
  let cum = 0;
  for (let j = 0; j < k; j++) {
    const idx = keep[j] ?? order[j];
    const lam = Math.max(eig[idx], 0);
    eigenvalues[j] = lam;
    cum += lam;
    energy[j] = cum / totalEnergy;
    // Φ_j = (1/√λ_j) · U · v_j
    const scale = lam > 1e-30 ? 1 / Math.sqrt(lam) : 0;
    const off = j * n;
    for (let p = 0; p < n; p++) {
      let acc = 0;
      for (let q = 0; q < s; q++) acc += U[q * n + p] * V[q * s + idx];
      modes[off + p] = acc * scale;
    }
    // Normalise in case of round-off.
    let nrm = 0;
    for (let p = 0; p < n; p++) nrm += modes[off + p] * modes[off + p];
    nrm = Math.sqrt(nrm);
    if (nrm > 0) {
      const inv = 1 / nrm;
      for (let p = 0; p < n; p++) modes[off + p] *= inv;
    }
  }
  return { n, k, modes, eigenvalues, energy };
}

/** Project a full-order vector x ∈ ℝⁿ onto the basis: a = Φᵀ x. */
export function projectToReduced(basis: PODBasis, x: Float64Array): Float64Array {
  const { n, k, modes } = basis;
  const a = new Float64Array(k);
  for (let j = 0; j < k; j++) {
    let s = 0;
    const off = j * n;
    for (let p = 0; p < n; p++) s += modes[off + p] * x[p];
    a[j] = s;
  }
  return a;
}

/** Lift a reduced vector a ∈ ℝᵏ back to full order: x = Φ · a. */
export function liftToFull(basis: PODBasis, a: Float64Array): Float64Array {
  const { n, k, modes } = basis;
  const x = new Float64Array(n);
  for (let j = 0; j < k; j++) {
    const aj = a[j];
    if (aj === 0) continue;
    const off = j * n;
    for (let p = 0; p < n; p++) x[p] += modes[off + p] * aj;
  }
  return x;
}

/** Assemble the reduced operator K_r = Φᵀ K Φ and its Cholesky factor. */
export function buildReducedModel(K: CSRMatrix, basis: PODBasis): ReducedModel {
  const { n, k, modes } = basis;
  if (K.n !== n) throw new Error(`buildReducedModel: K is ${K.n}×${K.n}, basis n=${n}`);
  // KΦ column-by-column (n×k).
  const KPhi = new Float64Array(n * k);
  const tmp = new Float64Array(n);
  for (let j = 0; j < k; j++) {
    const off = j * n;
    spmv(K, modes.subarray(off, off + n) as Float64Array, tmp);
    KPhi.set(tmp, off);
  }
  // K_r = Φᵀ (KΦ), row-major k×k.
  const Kr = new Float64Array(k * k);
  for (let i = 0; i < k; i++) {
    const oi = i * n;
    for (let j = 0; j < k; j++) {
      const oj = j * n;
      let acc = 0;
      for (let p = 0; p < n; p++) acc += modes[oi + p] * KPhi[oj + p];
      Kr[i * k + j] = acc;
    }
  }
  // Symmetrise (kill round-off) and factor.
  for (let i = 0; i < k; i++) {
    for (let j = i + 1; j < k; j++) {
      const m = 0.5 * (Kr[i * k + j] + Kr[j * k + i]);
      Kr[i * k + j] = m;
      Kr[j * k + i] = m;
    }
  }
  const L = choleskyFactor(Kr, k);
  return { basis, Kr, L };
}

/** Solve K · u ≈ Φ · (K_r⁻¹ Φᵀ f) using the surrogate. */
export function solveReduced(model: ReducedModel, f: Float64Array): {
  u: Float64Array;
  reducedCoefficients: Float64Array;
} {
  const { basis, L } = model;
  const fr = projectToReduced(basis, f);
  const ar = choleskySolve(L, basis.k, fr);
  const u = liftToFull(basis, ar);
  return { u, reducedCoefficients: ar };
}

// ── Linear algebra helpers ──────────────────────────────────────────────────

/**
 * Cyclic Jacobi eigen-decomposition of a symmetric n×n matrix (row-major,
 * not mutated). Returns eigenvalues and orthonormal eigenvectors stored
 * column-major (vectors[i + j·n] is the i-th entry of eigenvector j).
 */
function jacobiEigSym(A: Float64Array, n: number): {
  values: Float64Array;
  vectors: Float64Array;
} {
  const a = new Float64Array(A); // copy, mutated in place
  const V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1; // V starts as identity (col-major)

  const maxSweeps = 80;
  const tol = 1e-14;
  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        off += apq * apq;
      }
    }
    if (off < tol) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = a[p * n + q];
        if (Math.abs(apq) < 1e-18) continue;
        const app = a[p * n + p];
        const aqq = a[q * n + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(1 + theta * theta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        a[p * n + p] = app - t * apq;
        a[q * n + q] = aqq + t * apq;
        a[p * n + q] = 0;
        a[q * n + p] = 0;
        for (let i = 0; i < n; i++) {
          if (i !== p && i !== q) {
            const aip = a[i * n + p];
            const aiq = a[i * n + q];
            a[i * n + p] = c * aip - s * aiq;
            a[p * n + i] = a[i * n + p];
            a[i * n + q] = s * aip + c * aiq;
            a[q * n + i] = a[i * n + q];
          }
          // V columns p and q
          const vip = V[i + p * n];
          const viq = V[i + q * n];
          V[i + p * n] = c * vip - s * viq;
          V[i + q * n] = s * vip + c * viq;
        }
      }
    }
  }
  const values = new Float64Array(n);
  for (let i = 0; i < n; i++) values[i] = a[i * n + i];
  return { values, vectors: V };
}

/** Row-major Cholesky: A = L Lᵀ. Returns L (lower-triangular, row-major). */
function choleskyFactor(A: Float64Array, n: number): Float64Array {
  const L = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * n + j];
      for (let k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
      if (i === j) {
        if (sum <= 0) {
          // Tiny regularisation for numerically-singular reduced operators.
          sum = Math.max(sum, 1e-14);
        }
        L[i * n + j] = Math.sqrt(sum);
      } else {
        L[i * n + j] = sum / L[j * n + j];
      }
    }
  }
  return L;
}

/** Solve L Lᵀ x = b given the Cholesky factor L. */
function choleskySolve(L: Float64Array, n: number, b: Float64Array): Float64Array {
  const y = new Float64Array(n);
  const x = new Float64Array(n);
  // L y = b
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * n + k] * y[k];
    y[i] = s / L[i * n + i];
  }
  // Lᵀ x = y
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k];
    x[i] = s / L[i * n + i];
  }
  return x;
}
