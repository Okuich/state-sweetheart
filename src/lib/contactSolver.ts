/**
 * contactSolver.ts
 *
 * Narrow-phase → constraint-solver bridge for the unified
 * rigid / cloth / particle state used in PhysicsCanvas.
 *
 * Design
 * ──────
 * The existing `gpuNarrowPhase.ts` works over an abstract `BodySet`
 * (positions, velocities, invMass, kind, restitution). The live sim
 * however carries its state as a flat `{ x, v, m }` SoA (`State`).
 * Rather than copy state into a `BodySet` each frame, we run the same
 * pairwise contact test directly on the live arrays — broad-phase via
 * a uniform spatial grid (already mirrored elsewhere in the file), then
 * a sequential-impulse + Baumgarte-style position-correction pass.
 *
 * Per contact (with normal n from i → j, depth d > 0):
 *   • normal impulse  λ = max(0, −(1+e)·v_rel·n) / (wᵢ + wⱼ)
 *       v_i ← v_i − λ wᵢ n,   v_j ← v_j + λ wⱼ n
 *   • position split  x_i ← x_i − (d wᵢ)/(wᵢ+wⱼ) · n,
 *                     x_j ← x_j + (d wⱼ)/(wᵢ+wⱼ) · n
 *
 * Solver iterations are Gauss-Seidel: each contact reads the latest
 * velocity / position written by the previous contact in the same
 * iteration. `iters > 1` reduces residual penetration on dense piles.
 *
 * Deterministic: contacts are emitted in (i, j) ascending order from a
 * fixed cell-traversal pattern; given identical state and params the
 * output is bit-stable across runs.
 */

import { sphereCollide, penetration } from "@/lib/sdf/queries";
import type { SparseSDF } from "@/lib/sdf/sparseField";

export interface ContactStats {
  /** Number of (i, j) overlapping pairs detected this call. */
  contacts: number;
  /** Number of solver iterations actually executed. */
  iters: number;
  /** Sum of penetration depths across detected contacts. */
  totalPenetration: number;
  /** Max single-contact penetration depth before correction. */
  maxPenetration: number;
  /** Particles found penetrating the static SDF collider this call. */
  sdfContacts: number;
  /** Sum of SDF penetration depths before correction. */
  sdfTotalPenetration: number;
  /** Max single-particle SDF penetration depth before correction. */
  sdfMaxPenetration: number;
  /** Particles actually tested against the SDF (≤ N, capped by guardrail). */
  sdfChecks: number;
}

/** Minimum state shape this solver requires. */
export interface ContactState {
  N: number;
  x: Float32Array | Float64Array;
  v: Float32Array | Float64Array;
  m: Float32Array | Float64Array;
}

/** Static SDF collider lifted to the 2D simulation plane (XY at z=worldZ). */
export interface SDFColliderOptions {
  sdf: SparseSDF;
  /** World-space Z slice the 2D sim lives on. Default 0. */
  worldZ?: number;
  /**
   * Performance guardrail: cap the number of particles tested against
   * the SDF in a single call. When `N > maxChecksPerStep`, the solver
   * processes a strided subset of particles. The stride offset rotates
   * with `stepIndex` so every particle is covered over consecutive
   * frames — contact stabilization stays bounded even on huge piles.
   * Undefined or 0 disables the cap (test every particle).
   */
  maxChecksPerStep?: number;
  /**
   * Cheap reject threshold (world units). A particle is skipped before
   * the more expensive `sphereCollide` (gradient) call if its
   * `penetration + radius` is ≤ this value. Default 0 (only skip
   * particles strictly outside the sphere of influence). Set to a
   * small positive value (e.g. `0.25 * radius`) to drop shallow
   * grazing contacts that don't justify a Newton-style resolve.
   */
  rejectThreshold?: number;
  /** Frame counter used to rotate the strided subset. Default 0. */
  stepIndex?: number;
}

export interface ContactOptions {
  /** Per-particle contact radius. */
  radius: number;
  iters?: number;
  restitution?: number;
  beta?: number;
  slop?: number;
  pinnedMass?: number;
  /** Optional static SDF collider. When provided, every particle is
   *  tested with `sphereCollide` and resolved using the SDF gradient as
   *  the contact normal. */
  staticSDF?: SDFColliderOptions;
}

/**
 * Detects pairwise contacts and applies sequential impulses + a
 * position-correction split. Operates in place on `s.x` and `s.v`.
 */
export function resolveContacts(
  s: ContactState,
  opts: ContactOptions,
): ContactStats {
  const N = s.N | 0;
  const radius = Math.max(0, opts.radius);
  const iters = Math.max(1, opts.iters ?? 1);
  const e = Math.min(1, Math.max(0, opts.restitution ?? 0));
  const beta = Math.min(1, Math.max(0, opts.beta ?? 1));
  const slop = Math.max(0, opts.slop ?? 0);
  const pinned = opts.pinnedMass ?? Infinity;

  if (N < 2 || radius <= 0) {
    const empty = emptyStats();
    if (opts.staticSDF) resolveSDFContacts(s, radius, e, beta, slop, pinned, opts.staticSDF, empty);
    return empty;
  }

  const diam = 2 * radius;
  const diam2 = diam * diam;

  // ── Broad-phase: uniform grid keyed on cell = diam ──────────────────
  // Bounding box from current positions.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < N; i++) {
    const xi = s.x[i * 2], yi = s.x[i * 2 + 1];
    if (xi < minX) minX = xi; if (xi > maxX) maxX = xi;
    if (yi < minY) minY = yi; if (yi > maxY) maxY = yi;
  }
  // Guard degenerate.
  if (!isFinite(minX) || !isFinite(minY)) {
    return emptyStats();
  }
  const gw = Math.max(1, Math.ceil((maxX - minX) / diam) + 1);
  const gh = Math.max(1, Math.ceil((maxY - minY) / diam) + 1);
  const nCells = gw * gh;
  // Hard cap to avoid O(N²) memory in pathological spreads.
  if (nCells > N * 16 + 1024) {
    return resolveAllPairs(s, opts);
  }

  const cellOf = new Int32Array(N);
  const cellCount = new Int32Array(nCells);
  for (let i = 0; i < N; i++) {
    const cx = Math.min(gw - 1, Math.max(0, ((s.x[i * 2] - minX) / diam) | 0));
    const cy = Math.min(gh - 1, Math.max(0, ((s.x[i * 2 + 1] - minY) / diam) | 0));
    const c = cy * gw + cx;
    cellOf[i] = c;
    cellCount[c]++;
  }
  const cellStart = new Int32Array(nCells + 1);
  for (let c = 0; c < nCells; c++) cellStart[c + 1] = cellStart[c] + cellCount[c];
  const cursor = new Int32Array(nCells);
  const order = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    const c = cellOf[i];
    order[cellStart[c] + cursor[c]++] = i;
  }

  // Collect candidate pairs deterministically (i < j).
  const pairsI: number[] = [];
  const pairsJ: number[] = [];
  for (let cy = 0; cy < gh; cy++) {
    for (let cx = 0; cx < gw; cx++) {
      const c = cy * gw + cx;
      const aS = cellStart[c], aE = cellStart[c + 1];
      if (aS === aE) continue;
      for (let dyc = 0; dyc <= 1; dyc++) {
        for (let dxc = -1; dxc <= 1; dxc++) {
          if (dyc === 0 && dxc < 0) continue; // dedupe
          const nxi = cx + dxc, nyi = cy + dyc;
          if (nxi < 0 || nxi >= gw || nyi < 0 || nyi >= gh) continue;
          const cn = nyi * gw + nxi;
          const bS = cellStart[cn], bE = cellStart[cn + 1];
          if (bS === bE) continue;
          const same = c === cn;
          for (let ai = aS; ai < aE; ai++) {
            const i = order[ai];
            const startB = same ? ai + 1 : bS;
            for (let bi = startB; bi < bE; bi++) {
              const j = order[bi];
              const a = i < j ? i : j;
              const b = i < j ? j : i;
              const dx = s.x[a * 2] - s.x[b * 2];
              const dy = s.x[a * 2 + 1] - s.x[b * 2 + 1];
              if (dx * dx + dy * dy < diam2) {
                pairsI.push(a);
                pairsJ.push(b);
              }
            }
          }
        }
      }
    }
  }

  const stats = solveSequential(s, pairsI, pairsJ, diam, iters, e, beta, slop, pinned);
  if (opts.staticSDF) resolveSDFContacts(s, radius, e, beta, slop, pinned, opts.staticSDF, stats);
  return stats;
}

function resolveAllPairs(s: ContactState, opts: ContactOptions): ContactStats {
  const N = s.N | 0;
  const radius = Math.max(0, opts.radius);
  const iters = Math.max(1, opts.iters ?? 1);
  const e = Math.min(1, Math.max(0, opts.restitution ?? 0));
  const beta = Math.min(1, Math.max(0, opts.beta ?? 1));
  const slop = Math.max(0, opts.slop ?? 0);
  const pinned = opts.pinnedMass ?? Infinity;
  const diam = 2 * radius;
  const diam2 = diam * diam;
  const pairsI: number[] = [];
  const pairsJ: number[] = [];
  for (let i = 0; i < N; i++) {
    for (let j = i + 1; j < N; j++) {
      const dx = s.x[i * 2] - s.x[j * 2];
      const dy = s.x[i * 2 + 1] - s.x[j * 2 + 1];
      if (dx * dx + dy * dy < diam2) {
        pairsI.push(i);
        pairsJ.push(j);
      }
    }
  }
  const stats = solveSequential(s, pairsI, pairsJ, diam, iters, e, beta, slop, pinned);
  if (opts.staticSDF) resolveSDFContacts(s, radius, e, beta, slop, pinned, opts.staticSDF, stats);
  return stats;
}

function invMass(m: number, pinned: number): number {
  if (!isFinite(m) || m <= 0) return 0;
  if (m >= pinned) return 0;
  return 1 / m;
}

function solveSequential(
  s: ContactState,
  pairsI: number[],
  pairsJ: number[],
  diam: number,
  iters: number,
  e: number,
  beta: number,
  slop: number,
  pinned: number,
): ContactStats {
  const P = pairsI.length;
  let totalPen = 0;
  let maxPen = 0;

  // Pre-measure depth (before any correction) for stats.
  for (let p = 0; p < P; p++) {
    const i = pairsI[p], j = pairsJ[p];
    const dx = s.x[j * 2] - s.x[i * 2];
    const dy = s.x[j * 2 + 1] - s.x[i * 2 + 1];
    const d = Math.hypot(dx, dy);
    const pen = diam - d;
    if (pen > 0) {
      totalPen += pen;
      if (pen > maxPen) maxPen = pen;
    }
  }

  for (let it = 0; it < iters; it++) {
    for (let p = 0; p < P; p++) {
      const i = pairsI[p], j = pairsJ[p];
      const wi = invMass(s.m[i], pinned);
      const wj = invMass(s.m[j], pinned);
      const wsum = wi + wj;
      if (wsum <= 0) continue;

      const dx = s.x[j * 2] - s.x[i * 2];
      const dy = s.x[j * 2 + 1] - s.x[i * 2 + 1];
      const dist = Math.hypot(dx, dy);
      const pen = diam - dist;
      if (pen <= 0) continue;

      // Stable normal even when bodies coincide.
      const nx = dist > 1e-9 ? dx / dist : 1;
      const ny = dist > 1e-9 ? dy / dist : 0;

      // Velocity impulse (normal, no friction).
      const rvx = s.v[j * 2] - s.v[i * 2];
      const rvy = s.v[j * 2 + 1] - s.v[i * 2 + 1];
      const vn = rvx * nx + rvy * ny;
      if (vn < 0) {
        // Approaching — apply restitution impulse.
        const lambda = -(1 + e) * vn / wsum;
        s.v[i * 2]     -= lambda * wi * nx;
        s.v[i * 2 + 1] -= lambda * wi * ny;
        s.v[j * 2]     += lambda * wj * nx;
        s.v[j * 2 + 1] += lambda * wj * ny;
      }

      // Position correction (Baumgarte split by inverse mass).
      const corrDepth = Math.max(0, pen - slop) * beta;
      if (corrDepth > 0) {
        const di = (corrDepth * wi) / wsum;
        const dj = (corrDepth * wj) / wsum;
        s.x[i * 2]     -= di * nx;
        s.x[i * 2 + 1] -= di * ny;
        s.x[j * 2]     += dj * nx;
        s.x[j * 2 + 1] += dj * ny;
      }
    }
  }

  return { contacts: P, iters, totalPenetration: totalPen, maxPenetration: maxPen,
    sdfContacts: 0, sdfTotalPenetration: 0, sdfMaxPenetration: 0, sdfChecks: 0 };
}

function emptyStats(): ContactStats {
  return { contacts: 0, iters: 0, totalPenetration: 0, maxPenetration: 0,
    sdfContacts: 0, sdfTotalPenetration: 0, sdfMaxPenetration: 0, sdfChecks: 0 };
}

/**
 * Static SDF collision pass. Each particle is tested with `sphereCollide`
 * against the supplied SDF; on contact, the SDF gradient is used as the
 * surface normal and the particle is corrected against it (kinematic
 * collider — invMass = 0 on the SDF side, so the full impulse / position
 * correction is applied to the particle).
 *
 * The 2D simulation plane is lifted to z = `worldZ` (default 0) before
 * sampling. Only the (x,y) components of the gradient drive the response;
 * a strongly axial-Z normal indicates the particle is grazing the slice
 * tangentially and contributes only depth statistics.
 */
function resolveSDFContacts(
  s: ContactState,
  radius: number,
  e: number,
  beta: number,
  slop: number,
  pinned: number,
  collider: SDFColliderOptions,
  stats: ContactStats,
): void {
  const N = s.N | 0;
  const z = collider.worldZ ?? 0;
  const r = Math.max(0, radius);
  // Performance guardrail: cap how many particles we test, walking a
  // strided subset whose offset rotates each call so coverage is
  // amortized across frames. `cap = 0 / undefined / cap >= N` → test all.
  const cap = collider.maxChecksPerStep && collider.maxChecksPerStep > 0
    ? Math.min(N, collider.maxChecksPerStep | 0) : N;
  const stride = cap >= N ? 1 : Math.max(1, Math.floor(N / cap));
  const offset = cap >= N ? 0 : (((collider.stepIndex ?? 0) % stride) + stride) % stride;
  // Cheap reject threshold: skip particles whose `pen + radius` is
  // ≤ rejectThreshold without invoking the gradient. Default 0
  // preserves the original behaviour exactly.
  const rejectAt = collider.rejectThreshold ?? 0;
  let checks = 0;
  for (let i = offset; i < N; i += stride) {
    const wi = invMass(s.m[i], pinned);
    if (wi <= 0) continue;
    const px = s.x[i * 2], py = s.x[i * 2 + 1];
    checks++;
    // Cheap reject via penetration() before invoking the gradient.
    const pen = penetration(collider.sdf, [px, py, z]) + r;
    if (pen <= rejectAt) continue;
    const hit = sphereCollide(collider.sdf, [px, py, z], r);
    if (!hit.hit) continue;

    stats.sdfContacts++;
    stats.sdfTotalPenetration += hit.depth;
    if (hit.depth > stats.sdfMaxPenetration) stats.sdfMaxPenetration = hit.depth;

    // Project gradient normal onto the simulation plane.
    let nx = hit.normal[0], ny = hit.normal[1];
    const nl = Math.hypot(nx, ny);
    if (nl < 1e-6) continue; // normal is purely Z — no in-plane response
    nx /= nl; ny /= nl;

    // Velocity impulse (collider is kinematic → 1/wsum = 1/wi).
    const vn = s.v[i * 2] * nx + s.v[i * 2 + 1] * ny;
    if (vn < 0) {
      const lambda = -(1 + e) * vn; // wsum = wi → wi/wsum = 1
      s.v[i * 2]     += lambda * nx;
      s.v[i * 2 + 1] += lambda * ny;
    }
    // Position correction.
    const corr = Math.max(0, hit.depth - slop) * beta;
    if (corr > 0) {
      s.x[i * 2]     += corr * nx;
      s.x[i * 2 + 1] += corr * ny;
    }
  }
  stats.sdfChecks += checks;
}
