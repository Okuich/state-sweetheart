/**
 * verletSpring.ts — reference velocity-Verlet integrator + 1D spring oracle.
 *
 * Mirrors the two-step Verlet update used by PhysicsCanvas.tsx:
 *
 *   verletDrift (uses a_old, cached in fPrev):
 *     v ← v + ½·a_old·dt
 *     x ← x + v·dt
 *
 *   recompute forces → a_new
 *
 *   verletKick (uses a_new):
 *     v ← (v + ½·a_new·dt) · drag_decay
 *     fPrev ← f
 *
 * Drag decay matches the canvas's three drag modes:
 *   "explicit"    → (1 − k·dt)
 *   "exponential" → exp(−k·dt)
 *   "force"       → 1 (drag is added into f as −k·m·v before kick)
 *
 * Exposed for unit tests so we can assert the integrator reproduces the
 * analytical solution of the 1D simple-harmonic-oscillator
 *
 *     m·ẍ = −K·x − c·ẋ
 *
 * for small dt (and small K·dt² / c·dt).
 */

export type DragMode = "explicit" | "exponential" | "force";

export type VerletParticle = {
  x: number;     // 1D position
  v: number;     // 1D velocity
  m: number;     // mass
  f: number;     // current force (recomputed each step)
  fPrev: number; // force from previous step (used for first half-kick)
};

export function makeParticle(x: number, v: number, m = 1): VerletParticle {
  return { x, v, m, f: 0, fPrev: 0 };
}

/** Verlet first half-kick + drift, using cached a_old = fPrev/m. */
export function verletDrift(p: VerletParticle, dt: number): void {
  const aOld = p.fPrev / p.m;
  p.v += 0.5 * aOld * dt;
  p.x += p.v * dt;
}

/**
 * Verlet second half-kick using the freshly-computed force, with the same
 * drag decay the canvas applies.
 */
export function verletKick(
  p: VerletParticle,
  dt: number,
  damping: number,
  drag: DragMode = "explicit",
): void {
  const decay =
    drag === "force"        ? 1 :
    drag === "exponential"  ? Math.exp(-damping * dt) :
                              Math.max(0, 1 - damping * dt);
  const aNew = p.f / p.m;
  p.v = (p.v + 0.5 * aNew * dt) * decay;
  p.fPrev = p.f;
}

/**
 * One full Verlet step for a 1D spring −K·x (optionally with body-force drag).
 * Force is recomputed between drift and kick.
 */
export function verletStepSpring(
  p: VerletParticle,
  dt: number,
  K: number,
  damping: number,
  drag: DragMode = "explicit",
): void {
  verletDrift(p, dt);
  // recompute forces at new position
  p.f = -K * p.x;
  if (drag === "force" && damping > 0) {
    p.f += -damping * p.m * p.v;
  }
  verletKick(p, dt, damping, drag);
}

/**
 * Bootstrap fPrev so the very first verletDrift uses the correct a₀ = f(x₀)/m.
 * The canvas does this implicitly by initializing fPrev=0 and then converging
 * after one step; tests use this to compare cleanly against the analytical
 * solution from t=0.
 */
export function primeVerlet(p: VerletParticle, K: number): void {
  p.f = -K * p.x;
  p.fPrev = p.f;
}

/**
 * Analytical solution to m·ẍ = −K·x  (undamped SHO):
 *
 *   x(t) = x₀·cos(ω·t) + (v₀/ω)·sin(ω·t)
 *   v(t) = -x₀·ω·sin(ω·t) + v₀·cos(ω·t)
 *
 * with ω = √(K/m).
 */
export function analyticalSHO(
  x0: number, v0: number, K: number, m: number, t: number,
): { x: number; v: number } {
  const w = Math.sqrt(K / m);
  return {
    x: x0 * Math.cos(w * t) + (v0 / w) * Math.sin(w * t),
    v: -x0 * w * Math.sin(w * t) + v0 * Math.cos(w * t),
  };
}

/**
 * Total mechanical energy of a 1D spring system,
 *
 *     E = ½·m·v² + ½·K·x²
 *
 * Velocity-Verlet is symplectic — for the undamped oscillator E should
 * oscillate within an O(dt²) bound but not drift secularly.
 */
export function springEnergy(p: VerletParticle, K: number): number {
  return 0.5 * p.m * p.v * p.v + 0.5 * K * p.x * p.x;
}
