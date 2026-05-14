/**
 * Differentiable SDF queries.
 *
 * Returns analytical (or, where messy, central-difference) gradients of
 * the signed distance with respect to:
 *   • the query point  p  (∂d/∂p ∈ R³)
 *   • the shape parameters of each primitive (∂d/∂θ)
 *
 * Intended for topology-/shape-optimization: a downstream loss L(d)
 * can be back-propagated through `dScene` to update primitive params
 * via dL/dθ_i = (∂L/∂d) · w_i · (∂d_i/∂θ_i)  where w_i is the
 * soft-min weight returned per primitive.
 *
 * For the *baked* SparseSDF, only ∂d/∂p is meaningful (the shape
 * parameters are no longer present in the cached voxels). Callers that
 * want full ∂d/∂θ should evaluate the *analytic* scene via `dScene`.
 */

import { sampleSDF, type SparseSDF } from "./sparseField";
import { gradient } from "./queries";
import type { SDFPrim, Vec3 } from "./types";

// ── Per-primitive derivative records ─────────────────────────────────
export type PrimGradients =
  | { kind: "sphere";   d: number; dp: Vec3; dCenter: Vec3; dRadius: number }
  | { kind: "box";      d: number; dp: Vec3; dCenter: Vec3; dHalf: Vec3 }
  | { kind: "cylinder"; d: number; dp: Vec3; dCenter: Vec3; dRadius: number; dHeight: number; dAxis: Vec3 }
  | { kind: "plane";    d: number; dp: Vec3; dPoint: Vec3; dNormal: Vec3 }
  | { kind: "torus";    d: number; dp: Vec3; dCenter: Vec3; dMajor: number; dMinor: number };

export interface DiffSceneResult {
  /** Combined (soft-min) signed distance. */
  d: number;
  /** ∂d/∂p — gradient of the *combined* distance wrt the query point. */
  dp: Vec3;
  /**
   * Per-primitive contributions. `weight` is the soft-min weight w_i
   * (∂d_combined/∂d_i); multiply each `prim` derivative by `weight` to
   * get the contribution to ∂d_combined/∂θ_i.
   */
  prims: Array<{ index: number; weight: number; grads: PrimGradients }>;
}

// ── Vec3 helpers (kept local to avoid pulling in primitives.ts privates) ──
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);

/**
 * Differentiable single-primitive SDF: returns d and full gradients
 * wrt p and shape parameters.
 */
export function dPrim(prim: SDFPrim, p: Vec3): PrimGradients {
  switch (prim.kind) {
    case "sphere":   return diffSphere(prim, p);
    case "box":      return diffBox(prim, p);
    case "cylinder": return diffCylinder(prim, p);
    case "plane":    return diffPlane(prim, p);
    case "torus":    return diffTorus(prim, p);
  }
}

function diffSphere(prim: Extract<SDFPrim, { kind: "sphere" }>, p: Vec3): PrimGradients {
  const rel = sub(p, prim.center);
  const L = len(rel);
  const inv = L > 1e-12 ? 1 / L : 0;
  const n: Vec3 = [rel[0] * inv, rel[1] * inv, rel[2] * inv];
  return {
    kind: "sphere",
    d: L - prim.radius,
    dp:      n,
    dCenter: [-n[0], -n[1], -n[2]],
    dRadius: -1,
  };
}

function diffBox(prim: Extract<SDFPrim, { kind: "box" }>, p: Vec3): PrimGradients {
  // s_i = sign(p_i − c_i), q_i = |p_i − c_i| − h_i.
  const s: Vec3 = [
    p[0] >= prim.center[0] ? 1 : -1,
    p[1] >= prim.center[1] ? 1 : -1,
    p[2] >= prim.center[2] ? 1 : -1,
  ];
  const q: Vec3 = [
    Math.abs(p[0] - prim.center[0]) - prim.half[0],
    Math.abs(p[1] - prim.center[1]) - prim.half[1],
    Math.abs(p[2] - prim.center[2]) - prim.half[2],
  ];
  const qPos: Vec3 = [Math.max(q[0], 0), Math.max(q[1], 0), Math.max(q[2], 0)];
  const outside = Math.hypot(qPos[0], qPos[1], qPos[2]);
  const maxQ = Math.max(q[0], Math.max(q[1], q[2]));
  const inside = Math.min(maxQ, 0);
  const d = outside + inside;

  // ∂d/∂q_i: outside contribution is qPos_i / outside; inside contribution
  // applies only to argmax(q) when all q<0.
  const dDq: Vec3 = [0, 0, 0];
  if (outside > 1e-12) {
    dDq[0] = qPos[0] / outside;
    dDq[1] = qPos[1] / outside;
    dDq[2] = qPos[2] / outside;
  }
  if (maxQ < 0) {
    const i = q[0] >= q[1] && q[0] >= q[2] ? 0 : q[1] >= q[2] ? 1 : 2;
    dDq[i] = 1;
  }

  return {
    kind: "box",
    d,
    dp:      [s[0] * dDq[0], s[1] * dDq[1], s[2] * dDq[2]],
    dCenter: [-s[0] * dDq[0], -s[1] * dDq[1], -s[2] * dDq[2]],
    dHalf:   [-dDq[0], -dDq[1], -dDq[2]],
  };
}

function diffCylinder(prim: Extract<SDFPrim, { kind: "cylinder" }>, p: Vec3): PrimGradients {
  // Analytic for p, c, r, h. Axis derivative via central difference
  // (closed form is messy under axis renormalization).
  const aLen = Math.max(1e-12, len(prim.axis));
  const a: Vec3 = [prim.axis[0] / aLen, prim.axis[1] / aLen, prim.axis[2] / aLen];
  const rel = sub(p, prim.center);
  const t = dot(rel, a);
  const radial: Vec3 = [rel[0] - a[0] * t, rel[1] - a[1] * t, rel[2] - a[2] * t];
  const radL = len(radial);
  const dr = radL - prim.radius;
  const dh = Math.abs(t) - prim.height * 0.5;
  const outside = Math.hypot(Math.max(dr, 0), Math.max(dh, 0));
  const inside  = Math.min(Math.max(dr, dh), 0);
  const d = outside + inside;

  // ∂d/∂dr, ∂d/∂dh (mirrors box logic on a 2D corner).
  let dDdr = 0, dDdh = 0;
  if (outside > 1e-12) {
    dDdr = Math.max(dr, 0) / outside;
    dDdh = Math.max(dh, 0) / outside;
  }
  if (Math.max(dr, dh) < 0) {
    if (dr >= dh) dDdr = 1; else dDdh = 1;
  }

  // ∂dr/∂p_i = radial_i / radL  (orthogonal projection grad)
  // ∂dh/∂p_i = sign(t) * a_i
  const radInv = radL > 1e-12 ? 1 / radL : 0;
  const sgnT = t >= 0 ? 1 : -1;
  const dp: Vec3 = [
    dDdr * radial[0] * radInv + dDdh * sgnT * a[0],
    dDdr * radial[1] * radInv + dDdh * sgnT * a[1],
    dDdr * radial[2] * radInv + dDdh * sgnT * a[2],
  ];
  const dCenter: Vec3 = [-dp[0], -dp[1], -dp[2]];
  const dRadius = -dDdr;
  const dHeight = -0.5 * dDdh;

  // Axis derivative: small central-difference around prim.axis (unit-renorm
  // happens inside the evaluator, so we step the *raw* axis components).
  const dAxis = fdAxis(prim, p);

  return { kind: "cylinder", d, dp, dCenter, dRadius, dHeight, dAxis };
}

function fdAxis(prim: Extract<SDFPrim, { kind: "cylinder" }>, p: Vec3): Vec3 {
  const eps = 1e-4 * Math.max(1, len(prim.axis));
  const out: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const ax: Vec3 = [prim.axis[0], prim.axis[1], prim.axis[2]];
    ax[k] += eps;
    const dp = evalCylinderAxis(prim, p, ax);
    ax[k] -= 2 * eps;
    const dm = evalCylinderAxis(prim, p, ax);
    out[k] = (dp - dm) / (2 * eps);
  }
  return out;
}

function evalCylinderAxis(prim: Extract<SDFPrim, { kind: "cylinder" }>, p: Vec3, axis: Vec3): number {
  const aLen = Math.max(1e-12, len(axis));
  const a: Vec3 = [axis[0] / aLen, axis[1] / aLen, axis[2] / aLen];
  const rel = sub(p, prim.center);
  const t = dot(rel, a);
  const radial: Vec3 = [rel[0] - a[0] * t, rel[1] - a[1] * t, rel[2] - a[2] * t];
  const dr = len(radial) - prim.radius;
  const dh = Math.abs(t) - prim.height * 0.5;
  return Math.hypot(Math.max(dr, 0), Math.max(dh, 0)) + Math.min(Math.max(dr, dh), 0);
}

function diffPlane(prim: Extract<SDFPrim, { kind: "plane" }>, p: Vec3): PrimGradients {
  const nL = Math.max(1e-12, len(prim.normal));
  const n: Vec3 = [prim.normal[0] / nL, prim.normal[1] / nL, prim.normal[2] / nL];
  const rel = sub(p, prim.point);
  const d = dot(rel, n);
  // ∂d/∂point = -n. ∂d/∂normal_i: derivative of (rel · normal/|normal|).
  // = rel_i / |n| − (rel·n_unit) * n_unit_i / |n|
  const proj = d; // rel · n_unit
  const dNormal: Vec3 = [
    (rel[0] - proj * n[0]) / nL,
    (rel[1] - proj * n[1]) / nL,
    (rel[2] - proj * n[2]) / nL,
  ];
  return {
    kind: "plane",
    d,
    dp: n,
    dPoint: [-n[0], -n[1], -n[2]],
    dNormal,
  };
}

function diffTorus(prim: Extract<SDFPrim, { kind: "torus" }>, p: Vec3): PrimGradients {
  const rel = sub(p, prim.center);
  const rho = Math.hypot(rel[0], rel[2]);
  const q1 = rho - prim.major;
  const q2 = rel[1];
  const m = Math.hypot(q1, q2);
  const mInv = m > 1e-12 ? 1 / m : 0;
  const rhoInv = rho > 1e-12 ? 1 / rho : 0;
  const dp: Vec3 = [
    q1 * mInv * rel[0] * rhoInv,
    q2 * mInv,
    q1 * mInv * rel[2] * rhoInv,
  ];
  return {
    kind: "torus",
    d: m - prim.minor,
    dp,
    dCenter: [-dp[0], -dp[1], -dp[2]],
    dMajor: -q1 * mInv,
    dMinor: -1,
  };
}

/**
 * Differentiable scene SDF using a smooth-min combiner.
 *
 * For k > 0: log-sum-exp soft-min with sharpness 1/k.
 *   d = −k · log Σ exp(−d_i / k)
 *   ∂d/∂d_i = w_i = exp(−d_i / k) / Σ exp(−d_j / k)
 *
 * For k ≤ 0: hard min — w_i = 1 at argmin, 0 elsewhere (sub-gradient).
 *
 * The returned `prims` list pairs each primitive's full gradients with
 * its weight w_i, so callers can chain-rule: dL/dθ_i = (∂L/∂d) · w_i · ∂d_i/∂θ_i.
 */
export function dScene(prims: SDFPrim[], p: Vec3, k = 0): DiffSceneResult {
  if (prims.length === 0) {
    return { d: Number.POSITIVE_INFINITY, dp: [0, 0, 0], prims: [] };
  }
  const grads: PrimGradients[] = prims.map((pr) => dPrim(pr, p));

  // Soft-min weights.
  let weights: number[];
  let dCombined: number;
  if (k > 0) {
    const dMin = grads.reduce((a, g) => Math.min(a, g.d), Infinity);
    let Z = 0;
    const ex = grads.map((g) => {
      const e = Math.exp(-(g.d - dMin) / k);
      Z += e;
      return e;
    });
    weights = ex.map((e) => e / Z);
    dCombined = dMin - k * Math.log(Z);
  } else {
    let argmin = 0;
    for (let i = 1; i < grads.length; i++) if (grads[i].d < grads[argmin].d) argmin = i;
    weights = grads.map((_, i) => (i === argmin ? 1 : 0));
    dCombined = grads[argmin].d;
  }

  // Combined ∂d/∂p = Σ w_i · ∂d_i/∂p.
  const dp: Vec3 = [0, 0, 0];
  for (let i = 0; i < grads.length; i++) {
    const w = weights[i];
    if (w === 0) continue;
    dp[0] += w * grads[i].dp[0];
    dp[1] += w * grads[i].dp[1];
    dp[2] += w * grads[i].dp[2];
  }

  return {
    d: dCombined,
    dp,
    prims: grads.map((g, i) => ({ index: i, weight: weights[i], grads: g })),
  };
}

/**
 * Differentiable query against a *baked* SparseSDF.
 *
 * Only ∂d/∂p is recovered (via the existing central-difference
 * `gradient`); shape-parameter gradients are not preserved through
 * baking — use `dScene` against the source primitives for those.
 */
export function dField(sdf: SparseSDF, p: Vec3): { d: number; dp: Vec3 } {
  return { d: sampleSDF(sdf, p), dp: gradient(sdf, p) };
}
