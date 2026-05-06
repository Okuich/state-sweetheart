/**
 * materialModel.ts
 * Differentiable material parameter set + analytic gradient stubs for
 * five constitutive models. Pure functions; consumed by MaterialEditorPanel.
 */

export type MaterialKind =
  | "hookean"
  | "neo-hookean"
  | "viscoelastic"
  | "plastic"
  | "fracture";

export interface MaterialParams {
  /** Lamé first parameter (shear modulus). All models. */
  mu: number;
  /** Lamé second parameter. Hookean / Neo-Hookean. */
  lambda: number;
  /** Viscous damping coefficient (Pa·s). Viscoelastic. */
  eta: number;
  /** Relaxation time (s). Viscoelastic. */
  tau: number;
  /** Yield stress (Pa). Plastic. */
  yieldStress: number;
  /** Strain-hardening modulus. Plastic. */
  hardening: number;
  /** Critical strain at fracture initiation. Fracture. */
  epsFrac: number;
  /** Fracture energy release rate (J/m²). Fracture. */
  Gc: number;
}

export const DEFAULT_PARAMS: MaterialParams = {
  mu:          1.0e4,
  lambda:      1.5e4,
  eta:         50,
  tau:         0.05,
  yieldStress: 1.0e3,
  hardening:   200,
  epsFrac:     0.15,
  Gc:          50,
};

/** Which params are "active" (i.e. influence ψ) for each model. */
export const ACTIVE_PARAMS: Record<MaterialKind, ReadonlyArray<keyof MaterialParams>> = {
  "hookean":     ["mu", "lambda"],
  "neo-hookean": ["mu", "lambda"],
  "viscoelastic":["mu", "lambda", "eta", "tau"],
  "plastic":     ["mu", "lambda", "yieldStress", "hardening"],
  "fracture":    ["mu", "lambda", "epsFrac", "Gc"],
};

/**
 * Strain energy density ψ(ε) for the chosen model, evaluated at a
 * scalar 1D strain `eps` (sufficient for a UI preview / sparkline).
 *
 * Hookean:       ψ = 0.5·(2μ+λ)·ε²
 * Neo-Hookean:   ψ = 0.5·μ·(F²−1) − μ·ln|F| + 0.5·λ·(ln|F|)²,  F = 1+ε
 * Viscoelastic:  ψ_h + 0.5·η/τ·ε²              (Maxwell-like preview)
 * Plastic:       elastic up to yield, then hardening slope
 * Fracture:      ψ_h · (1 − d), d = min(1, max(0,(ε−ε_f)·E/G_c))
 */
export function strainEnergy(kind: MaterialKind, p: MaterialParams, eps: number): number {
  const E = 2 * p.mu + p.lambda;
  switch (kind) {
    case "hookean":
      return 0.5 * E * eps * eps;
    case "neo-hookean": {
      const F = 1 + eps;
      if (F <= 0) return Number.POSITIVE_INFINITY;
      const lnF = Math.log(F);
      return 0.5 * p.mu * (F * F - 1) - p.mu * lnF + 0.5 * p.lambda * lnF * lnF;
    }
    case "viscoelastic":
      return 0.5 * E * eps * eps + 0.5 * (p.eta / Math.max(p.tau, 1e-9)) * eps * eps;
    case "plastic": {
      const epsY = p.yieldStress / Math.max(E, 1e-9);
      if (Math.abs(eps) <= epsY) return 0.5 * E * eps * eps;
      const ep = Math.abs(eps) - epsY;
      return 0.5 * E * epsY * epsY + p.yieldStress * ep + 0.5 * p.hardening * ep * ep;
    }
    case "fracture": {
      const psi = 0.5 * E * eps * eps;
      const d = Math.min(1, Math.max(0, (Math.abs(eps) - p.epsFrac) * E / Math.max(p.Gc, 1e-9)));
      return psi * (1 - d);
    }
  }
}

/**
 * Analytic gradient ∂ψ/∂θ for each active parameter θ.
 * These are the building blocks a differentiable simulator would feed
 * into adjoint backprop. Returns 0 for inactive params.
 */
export function gradient(
  kind: MaterialKind,
  p: MaterialParams,
  eps: number,
): Partial<Record<keyof MaterialParams, number>> {
  const out: Partial<Record<keyof MaterialParams, number>> = {};
  const e2 = eps * eps;
  switch (kind) {
    case "hookean":
      out.mu     = e2;          // ∂(0.5(2μ+λ)ε²)/∂μ
      out.lambda = 0.5 * e2;
      break;
    case "neo-hookean": {
      const F = 1 + eps;
      if (F > 0) {
        const lnF = Math.log(F);
        out.mu     = 0.5 * (F * F - 1) - lnF;
        out.lambda = 0.5 * lnF * lnF;
      }
      break;
    }
    case "viscoelastic": {
      const t = Math.max(p.tau, 1e-9);
      out.mu     = e2;
      out.lambda = 0.5 * e2;
      out.eta    = 0.5 * e2 / t;
      out.tau    = -0.5 * p.eta * e2 / (t * t);
      break;
    }
    case "plastic": {
      const E = 2 * p.mu + p.lambda;
      const epsY = p.yieldStress / Math.max(E, 1e-9);
      if (Math.abs(eps) <= epsY) {
        out.mu     = e2;
        out.lambda = 0.5 * e2;
      } else {
        const ep = Math.abs(eps) - epsY;
        out.mu          = epsY * epsY;
        out.lambda      = 0.5 * epsY * epsY;
        out.yieldStress = ep;
        out.hardening   = 0.5 * ep * ep;
      }
      break;
    }
    case "fracture": {
      const E = 2 * p.mu + p.lambda;
      const arg = (Math.abs(eps) - p.epsFrac) * E / Math.max(p.Gc, 1e-9);
      const d = Math.min(1, Math.max(0, arg));
      const psi = 0.5 * E * eps * eps;
      out.mu     = e2 * (1 - d);
      out.lambda = 0.5 * e2 * (1 - d);
      if (arg > 0 && arg < 1) {
        out.epsFrac = psi * (E / Math.max(p.Gc, 1e-9));
        out.Gc      = psi * ((Math.abs(eps) - p.epsFrac) * E / (p.Gc * p.Gc));
      }
      break;
    }
  }
  return out;
}
