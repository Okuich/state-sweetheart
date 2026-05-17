/**
 * Differentiable thermal solves via the adjoint method.
 *
 * For a linear, symmetric system K(p)·u = b(p) the gradient of any scalar
 * loss L(u(p)) is recovered with ONE additional linear solve (the adjoint),
 * not by backpropagating through PCG iterations:
 *
 *       Forward:    K · u = b
 *       Adjoint:    K · λ = ∂L/∂u
 *       Then:       ∂L/∂p = −λᵀ · (∂K/∂p) · u + λᵀ · (∂b/∂p)
 *
 * Because K is SPD (and symmetric), forward and adjoint share the same
 * operator. We re-use `solvePoisson` for the adjoint solve, pinning the
 * Dirichlet nodes to zero — the symmetric `applyDirichlet` then enforces
 * λ_Γ = 0, leaving a free-node system on the interior.
 *
 * Supported design parameters:
 *   • per-tet scalar conductivity κ_t          (isotropic only here)
 *   • per-vertex volumetric source f_v
 *   • per-vertex pre-integrated Neumann load g_v
 *
 * Supported losses are arbitrary smooth scalars whose ∂L/∂u (per vertex)
 * AND ∂L/∂q (per tet, optional) the caller provides. Common helpers for
 * "target temperature at probe nodes" and "minimize peak flux" are
 * exported below.
 *
 * This module is verified end-to-end against centered finite differences
 * in `differentiable.test.ts` to guarantee gradient correctness — the
 * essential trust contract for any inverse-design / training loop.
 */
import { type CSRMatrix } from "./sparse";
import { solvePoisson, type DirichletBC } from "./poisson";
import {
  assembleThermalStiffness, solveThermal,
  type ThermalSolution, type ThermalProblem,
} from "./thermal";
import { type FEMMeshInput } from "./laplacian";
import { type CGOptions } from "./solvers/cg";

export interface DifferentiableThermalProblem {
  mesh: FEMMeshInput;
  /** Per-tet isotropic conductivity (length = nTets). Required for grads. */
  kappa: Float64Array;
  /** Optional per-vertex source density (length = nVerts). */
  source?: Float64Array;
  /** Optional per-vertex Neumann nodal loads (length = nVerts). */
  neumannLoads?: Float64Array;
  dirichlet?: ReadonlyArray<DirichletBC>;
  cg?: CGOptions;
}

export interface ThermalSensitivities {
  /** ∂L/∂T per vertex (length = nVerts). Optional. */
  dLdT?: Float64Array;
  /** ∂L/∂q per tet, flat xyz, length = 3·nTets. Optional. */
  dLdFluxPerTet?: Float64Array;
}

export interface ThermalGradients {
  /** ∂L/∂κ per tet (length = nTets). */
  dLdKappa: Float64Array;
  /** ∂L/∂f per vertex (length = nVerts). */
  dLdSource: Float64Array;
  /** ∂L/∂g per vertex (length = nVerts). Dirichlet nodes are zero. */
  dLdLoads: Float64Array;
  /** Forward solution snapshot used to compute the gradient. */
  forward: ThermalSolution;
  /** Adjoint field λ per vertex. */
  adjoint: Float64Array;
}

/**
 * Run forward + adjoint to obtain analytic gradients of `L = L_T + L_q`
 * with respect to (κ, source, loads).
 */
export function differentiateThermal(
  problem: DifferentiableThermalProblem,
  sens: ThermalSensitivities,
  forward?: ThermalSolution,
): ThermalGradients {
  // ── Forward solve (re-use if caller already has one) ────────────────────
  const fwd = forward ?? solveThermal(toThermalProblem(problem));
  const u = fwd.T;
  const nV = u.length;

  // ── Assemble K and per-tet grads ────────────────────────────────────────
  const asm = assembleThermalStiffness(problem.mesh, problem.kappa);
  const K: CSRMatrix = asm.K;
  const tetGrads = asm.tetGradients;   // (nTets · 4 · 3)
  const tetVols = asm.tetVolumes;
  const tets = problem.mesh.tets;
  const nTets = tets.length / 4;

  // ── Build adjoint RHS = ∂L/∂u (in nodal coordinates) ────────────────────
  // Direct vertex-loss term plus flux backprop:
  //   ∂q_t,d / ∂u_v  =  −κ_t · g_{v,d}  (v ∈ tet t)
  const adjRhs = new Float64Array(nV);
  if (sens.dLdT) {
    for (let i = 0; i < nV; i++) adjRhs[i] += sens.dLdT[i];
  }
  const dLdq = sens.dLdFluxPerTet;
  if (dLdq) {
    for (let t = 0; t < nTets; t++) {
      const kt = problem.kappa[t];
      const dx = dLdq[t * 3], dy = dLdq[t * 3 + 1], dz = dLdq[t * 3 + 2];
      for (let k = 0; k < 4; k++) {
        const vid = tets[t * 4 + k];
        const gx = tetGrads[t * 12 + k * 3];
        const gy = tetGrads[t * 12 + k * 3 + 1];
        const gz = tetGrads[t * 12 + k * 3 + 2];
        adjRhs[vid] += -kt * (dx * gx + dy * gy + dz * gz);
      }
    }
  }

  // Pin Dirichlet nodes of the adjoint to zero (same symmetric elimination).
  const zeroDirichlet: DirichletBC[] = (problem.dirichlet ?? []).map((bc) => ({
    index: bc.index, value: 0,
  }));

  // Use solvePoisson with loads = adjRhs (already nodal). massLumped is
  // required by the interface but unused when source is omitted.
  const adjSolve = solvePoisson({
    K,
    massLumped: asm.massLumped,
    loads: adjRhs,
    dirichlet: zeroDirichlet,
    cg: problem.cg,
  });
  const lambda = adjSolve.u;

  // ── Gradient w.r.t. per-tet κ ───────────────────────────────────────────
  // ∂L/∂κ_t (operator term):  −λ_local · K_t^e(κ=1) · u_local
  // where K_t^e(κ=1)_ij = V_t · g_i · g_j.
  // Plus direct flux term:    Σ_d (∂L/∂q_t,d) · (q_t,d / κ_t)
  //                           = −Σ_d dLdq · g_local · u_local  (G u = −q/κ)
  const dLdKappa = new Float64Array(nTets);
  for (let t = 0; t < nTets; t++) {
    const V = tetVols[t];
    const id0 = tets[t * 4],     id1 = tets[t * 4 + 1];
    const id2 = tets[t * 4 + 2], id3 = tets[t * 4 + 3];
    const ids = [id0, id1, id2, id3];

    // Local arrays for clarity.
    const ul = [u[id0], u[id1], u[id2], u[id3]];
    const ll = [lambda[id0], lambda[id1], lambda[id2], lambda[id3]];

    // Operator term: − Σ_{i,j} λ_i · (V · g_i·g_j) · u_j
    let opGrad = 0;
    for (let i = 0; i < 4; i++) {
      const gix = tetGrads[t * 12 + i * 3];
      const giy = tetGrads[t * 12 + i * 3 + 1];
      const giz = tetGrads[t * 12 + i * 3 + 2];
      for (let j = 0; j < 4; j++) {
        const gjx = tetGrads[t * 12 + j * 3];
        const gjy = tetGrads[t * 12 + j * 3 + 1];
        const gjz = tetGrads[t * 12 + j * 3 + 2];
        opGrad += ll[i] * V * (gix * gjx + giy * gjy + giz * gjz) * ul[j];
      }
    }
    let g = -opGrad;

    // Direct flux term, if present.
    if (dLdq) {
      const dx = dLdq[t * 3], dy = dLdq[t * 3 + 1], dz = dLdq[t * 3 + 2];
      // (∂q/∂κ) = −G u  →   contribution = (∂L/∂q) · (−G u)
      let Gux = 0, Guy = 0, Guz = 0;
      for (let k = 0; k < 4; k++) {
        Gux += tetGrads[t * 12 + k * 3]     * ul[k];
        Guy += tetGrads[t * 12 + k * 3 + 1] * ul[k];
        Guz += tetGrads[t * 12 + k * 3 + 2] * ul[k];
      }
      g += -(dx * Gux + dy * Guy + dz * Guz);
    }
    dLdKappa[t] = g;
    // touch ids to silence "declared but not used" — used implicitly above
    void ids;
  }

  // ── Gradient w.r.t. per-vertex source: ∂rhs/∂f_i = M_i (lumped mass)
  const dLdSource = new Float64Array(nV);
  for (let i = 0; i < nV; i++) dLdSource[i] = asm.massLumped[i] * lambda[i];

  // ── Gradient w.r.t. per-vertex Neumann load: ∂rhs/∂g_i = 1
  const dLdLoads = new Float64Array(nV);
  for (let i = 0; i < nV; i++) dLdLoads[i] = lambda[i];
  // Dirichlet nodes have lambda = 0 already → automatically zero.

  return { dLdKappa, dLdSource, dLdLoads, forward: fwd, adjoint: lambda };
}

function toThermalProblem(p: DifferentiableThermalProblem): ThermalProblem {
  return {
    mesh: p.mesh,
    kappa: p.kappa,
    source: p.source,
    neumannLoads: p.neumannLoads,
    dirichlet: p.dirichlet,
    cg: p.cg,
  };
}

// ─── Loss helpers ────────────────────────────────────────────────────────────

/**
 * L = ½ · Σ_v w_v · (T_v − T*_v)²   →   ∂L/∂T_v = w_v · (T_v − T*_v).
 * `weights` defaults to 1 for indices present in `probes`, 0 elsewhere.
 */
export function targetTemperatureLoss(
  T: Float64Array,
  probes: ReadonlyArray<{ index: number; target: number; weight?: number }>,
): { loss: number; dLdT: Float64Array } {
  const dLdT = new Float64Array(T.length);
  let loss = 0;
  for (const p of probes) {
    const w = p.weight ?? 1;
    const d = T[p.index] - p.target;
    loss += 0.5 * w * d * d;
    dLdT[p.index] += w * d;
  }
  return { loss, dLdT };
}

/**
 * L = ½ · Σ_t w_t · |q_t|²    →    ∂L/∂q_t = w_t · q_t.
 * Encourages flux minimization (e.g. insulating designs).
 */
export function fluxMagnitudeLoss(
  fluxPerTet: Float64Array,
  weights?: Float64Array,
): { loss: number; dLdFluxPerTet: Float64Array } {
  const nTets = fluxPerTet.length / 3;
  const dLdq = new Float64Array(fluxPerTet.length);
  let loss = 0;
  for (let t = 0; t < nTets; t++) {
    const w = weights ? weights[t] : 1;
    const qx = fluxPerTet[t * 3];
    const qy = fluxPerTet[t * 3 + 1];
    const qz = fluxPerTet[t * 3 + 2];
    loss += 0.5 * w * (qx * qx + qy * qy + qz * qz);
    dLdq[t * 3]     = w * qx;
    dLdq[t * 3 + 1] = w * qy;
    dLdq[t * 3 + 2] = w * qz;
  }
  return { loss, dLdFluxPerTet: dLdq };
}

// ─── Inverse-design driver ───────────────────────────────────────────────────

export interface InverseDesignOptions {
  /** Number of outer optimization steps. */
  steps?: number;
  /** Learning rate on log(κ). Default 0.05. */
  learningRate?: number;
  /** Lower bound on κ (defaults to 1e-3). */
  kappaMin?: number;
  /** Upper bound on κ (defaults to 1e3). */
  kappaMax?: number;
  /** Tikhonov smoothness on log(κ) deltas vs initial. Default 0. */
  regularization?: number;
  /** Optional progress callback per outer step. */
  onStep?: (step: number, loss: number, kappa: Float64Array) => void;
}

export interface InverseDesignResult {
  kappa: Float64Array;
  history: Array<{ step: number; loss: number; gradNorm: number }>;
  finalSolution: ThermalSolution;
}

/**
 * Plain log-space projected gradient descent on per-tet κ to minimize a
 * temperature-target loss. Demonstrates the differentiable path end-to-end
 * and is the building block the panel's "Optimize κ" mode calls into.
 *
 * This is intentionally simple (no line search, no L-BFGS) — the value
 * proposition is the adjoint gradient itself, which any external optimizer
 * (Adam, L-BFGS, neural-net training loop) can consume directly.
 */
export function inverseDesignKappa(
  problem: DifferentiableThermalProblem,
  probes: ReadonlyArray<{ index: number; target: number; weight?: number }>,
  options: InverseDesignOptions = {},
): InverseDesignResult {
  const steps = options.steps ?? 20;
  const lr = options.learningRate ?? 0.05;
  const kMin = options.kappaMin ?? 1e-3;
  const kMax = options.kappaMax ?? 1e3;
  const reg = options.regularization ?? 0;

  const kappa = new Float64Array(problem.kappa);
  const logK0 = new Float64Array(kappa.length);
  for (let t = 0; t < kappa.length; t++) logK0[t] = Math.log(kappa[t]);

  const history: InverseDesignResult["history"] = [];
  let lastSolution: ThermalSolution | undefined;

  for (let step = 0; step < steps; step++) {
    const fwd = solveThermal({ ...toThermalProblem(problem), kappa });
    lastSolution = fwd;
    const { loss, dLdT } = targetTemperatureLoss(fwd.T, probes);

    // Regularization: ½ · reg · Σ (log κ_t − log κ_t^0)².
    let regLoss = 0;
    const dRegdLogK = new Float64Array(kappa.length);
    if (reg > 0) {
      for (let t = 0; t < kappa.length; t++) {
        const d = Math.log(kappa[t]) - logK0[t];
        regLoss += 0.5 * reg * d * d;
        dRegdLogK[t] = reg * d;
      }
    }

    const grads = differentiateThermal(
      { ...problem, kappa },
      { dLdT },
      fwd,
    );
    // Convert dL/dκ → dL/d(log κ) = dL/dκ · κ, then add regularizer (already in log-space).
    const dLdLogK = new Float64Array(kappa.length);
    let gNorm2 = 0;
    for (let t = 0; t < kappa.length; t++) {
      dLdLogK[t] = grads.dLdKappa[t] * kappa[t] + dRegdLogK[t];
      gNorm2 += dLdLogK[t] * dLdLogK[t];
    }
    const gradNorm = Math.sqrt(gNorm2);
    history.push({ step, loss: loss + regLoss, gradNorm });
    options.onStep?.(step, loss + regLoss, kappa);

    // Projected gradient step in log-space.
    for (let t = 0; t < kappa.length; t++) {
      let lk = Math.log(kappa[t]) - lr * dLdLogK[t];
      if (lk < Math.log(kMin)) lk = Math.log(kMin);
      if (lk > Math.log(kMax)) lk = Math.log(kMax);
      kappa[t] = Math.exp(lk);
    }
  }

  const finalSolution = lastSolution ?? solveThermal({ ...toThermalProblem(problem), kappa });
  return { kappa, history, finalSolution };
}

// ─── Joint inverse design over (κ, source, loads) ────────────────────────────

export interface InverseDesignTargets {
  /** Optimize per-tet log(κ). Bounds: [kappaMin, kappaMax]. Default false. */
  kappa?: boolean;
  /** Optimize per-vertex source density f. Default false. */
  source?: boolean;
  /** Optimize per-vertex Neumann load g. Dirichlet nodes are skipped. Default false. */
  loads?: boolean;
}

export interface InverseDesignAllOptions {
  steps?: number;
  /** Per-channel learning rates. Defaults: κ=0.05, source=0.05, loads=0.05. */
  learningRate?: { kappa?: number; source?: number; loads?: number };
  kappaMin?: number;       // default 1e-3
  kappaMax?: number;       // default 1e3
  sourceMin?: number;      // default -Infinity
  sourceMax?: number;      // default +Infinity
  loadsMin?: number;       // default -Infinity
  loadsMax?: number;       // default +Infinity
  /**
   * Tikhonov regularization vs the initial value, per channel.
   *   κ:      ½·reg.kappa  · Σ (log κ_t  − log κ_t^0)²
   *   source: ½·reg.source · Σ (f_i      − f_i^0)²
   *   loads:  ½·reg.loads  · Σ (g_i      − g_i^0)²
   */
  regularization?: { kappa?: number; source?: number; loads?: number };
  onStep?: (
    step: number,
    loss: number,
    state: { kappa: Float64Array; source: Float64Array; loads: Float64Array },
  ) => void;
}

export interface InverseDesignAllResult {
  kappa: Float64Array;
  source: Float64Array;
  loads: Float64Array;
  history: Array<{ step: number; loss: number; gradNorm: number }>;
  finalSolution: ThermalSolution;
}

/**
 * Projected gradient descent that can jointly optimize any subset of
 * (per-tet κ, per-vertex source f, per-vertex Neumann load g) against a
 * temperature-target loss at probe vertices.
 *
 * κ is updated in log-space (multiplicative, positivity preserved); source
 * and loads are updated in linear space with optional box constraints.
 * Dirichlet nodes are left untouched for loads (their adjoint is zero, so
 * the gradient is already zero — we additionally pin the value to its
 * initial scalar so projection is a no-op there).
 *
 * One outer step = one forward solve + one adjoint solve, then a single
 * vectorized update per active channel. No line search.
 */
export function inverseDesignThermal(
  problem: DifferentiableThermalProblem,
  probes: ReadonlyArray<{ index: number; target: number; weight?: number }>,
  targets: InverseDesignTargets,
  options: InverseDesignAllOptions = {},
): InverseDesignAllResult {
  const steps = options.steps ?? 20;
  const lrK = options.learningRate?.kappa  ?? 0.05;
  const lrF = options.learningRate?.source ?? 0.05;
  const lrG = options.learningRate?.loads  ?? 0.05;
  const kMin = options.kappaMin ?? 1e-3;
  const kMax = options.kappaMax ?? 1e3;
  const fMin = options.sourceMin ?? -Infinity;
  const fMax = options.sourceMax ?? +Infinity;
  const gMin = options.loadsMin  ?? -Infinity;
  const gMax = options.loadsMax  ?? +Infinity;
  const regK = options.regularization?.kappa  ?? 0;
  const regF = options.regularization?.source ?? 0;
  const regG = options.regularization?.loads  ?? 0;

  const nV = problem.mesh.vertices.length / 3;
  const nT = problem.kappa.length;

  // Live design state (cloned from problem so caller's buffers are untouched).
  const kappa  = new Float64Array(problem.kappa);
  const source = problem.source
    ? new Float64Array(problem.source)
    : new Float64Array(nV);
  const loads  = problem.neumannLoads
    ? new Float64Array(problem.neumannLoads)
    : new Float64Array(nV);

  // Initial references for regularization.
  const logK0 = new Float64Array(nT);
  for (let t = 0; t < nT; t++) logK0[t] = Math.log(kappa[t]);
  const f0 = new Float64Array(source);
  const g0 = new Float64Array(loads);

  // Mask of Dirichlet vertex ids (no update on loads / source there).
  const dirichletMask = new Uint8Array(nV);
  if (problem.dirichlet) {
    for (const bc of problem.dirichlet) dirichletMask[bc.index] = 1;
  }

  const history: InverseDesignAllResult["history"] = [];
  let lastSolution: ThermalSolution | undefined;

  for (let step = 0; step < steps; step++) {
    // ── Forward solve with current design.
    const fwd = solveThermal({
      mesh: problem.mesh,
      kappa,
      source,
      neumannLoads: loads,
      dirichlet: problem.dirichlet,
      cg: problem.cg,
    });
    lastSolution = fwd;

    const { loss: dataLoss, dLdT } = targetTemperatureLoss(fwd.T, probes);

    // Regularizers (closed-form, also contribute to displayed loss).
    let regLoss = 0;
    const dRegLogK = new Float64Array(nT);
    if (regK > 0 && targets.kappa) {
      for (let t = 0; t < nT; t++) {
        const d = Math.log(kappa[t]) - logK0[t];
        regLoss += 0.5 * regK * d * d;
        dRegLogK[t] = regK * d;
      }
    }
    const dRegF = new Float64Array(nV);
    if (regF > 0 && targets.source) {
      for (let i = 0; i < nV; i++) {
        const d = source[i] - f0[i];
        regLoss += 0.5 * regF * d * d;
        dRegF[i] = regF * d;
      }
    }
    const dRegG = new Float64Array(nV);
    if (regG > 0 && targets.loads) {
      for (let i = 0; i < nV; i++) {
        if (dirichletMask[i]) continue;
        const d = loads[i] - g0[i];
        regLoss += 0.5 * regG * d * d;
        dRegG[i] = regG * d;
      }
    }

    // ── Adjoint solve via differentiateThermal (one extra linear solve).
    const grads = differentiateThermal(
      { ...problem, kappa, source, neumannLoads: loads },
      { dLdT },
      fwd,
    );

    let gNorm2 = 0;

    // ── κ step (log-space).
    if (targets.kappa) {
      for (let t = 0; t < nT; t++) {
        const dLogK = grads.dLdKappa[t] * kappa[t] + dRegLogK[t];
        gNorm2 += dLogK * dLogK;
        let lk = Math.log(kappa[t]) - lrK * dLogK;
        if (lk < Math.log(kMin)) lk = Math.log(kMin);
        if (lk > Math.log(kMax)) lk = Math.log(kMax);
        kappa[t] = Math.exp(lk);
      }
    }

    // ── source step (linear, box constraints).
    if (targets.source) {
      for (let i = 0; i < nV; i++) {
        const d = grads.dLdSource[i] + dRegF[i];
        gNorm2 += d * d;
        let v = source[i] - lrF * d;
        if (v < fMin) v = fMin;
        if (v > fMax) v = fMax;
        source[i] = v;
      }
    }

    // ── loads step (linear, box constraints, skip Dirichlet).
    if (targets.loads) {
      for (let i = 0; i < nV; i++) {
        if (dirichletMask[i]) continue;
        const d = grads.dLdLoads[i] + dRegG[i];
        gNorm2 += d * d;
        let v = loads[i] - lrG * d;
        if (v < gMin) v = gMin;
        if (v > gMax) v = gMax;
        loads[i] = v;
      }
    }

    const totalLoss = dataLoss + regLoss;
    history.push({ step, loss: totalLoss, gradNorm: Math.sqrt(gNorm2) });
    options.onStep?.(step, totalLoss, { kappa, source, loads });
  }

  const finalSolution = lastSolution ?? solveThermal({
    mesh: problem.mesh, kappa, source, neumannLoads: loads,
    dirichlet: problem.dirichlet, cg: problem.cg,
  });
  return { kappa, source, loads, history, finalSolution };
}

/**
 * Convenience wrapper: optimize only per-vertex source f against probe targets.
 */
export function inverseDesignSource(
  problem: DifferentiableThermalProblem,
  probes: ReadonlyArray<{ index: number; target: number; weight?: number }>,
  options: InverseDesignAllOptions = {},
): InverseDesignAllResult {
  return inverseDesignThermal(problem, probes, { source: true }, options);
}

/**
 * Convenience wrapper: optimize only per-vertex Neumann load g against probe
 * targets. Dirichlet nodes are pinned (their gradient is zero).
 */
export function inverseDesignLoads(
  problem: DifferentiableThermalProblem,
  probes: ReadonlyArray<{ index: number; target: number; weight?: number }>,
  options: InverseDesignAllOptions = {},
): InverseDesignAllResult {
  return inverseDesignThermal(problem, probes, { loads: true }, options);
}
