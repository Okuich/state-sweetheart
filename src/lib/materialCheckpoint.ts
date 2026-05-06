/**
 * materialCheckpoint.ts
 * Save/restore the per-element history variables that make plastic and
 * viscoelastic constitutive laws path-dependent:
 *
 *   • Fp[e]  — plastic deformation gradient (2×2 in 2D, stored as 4 floats
 *              per element in column-major order).
 *   • Sv[e]  — viscoelastic internal stress (Maxwell branch), 3 floats
 *              per element representing the symmetric 2×2 tensor
 *              components (xx, yy, xy).
 *   • alpha[e] — scalar isotropic hardening variable (equivalent plastic
 *                strain). Drives the yield surface expansion.
 *
 * A `MaterialState` is a struct-of-arrays. A `Checkpoint` is a deep
 * (copied) snapshot stamped with a step id and a content hash. The
 * `RollbackBuffer` keeps a bounded ring of checkpoints so the user can
 * step backwards, branch, or kick off a deterministic replay.
 *
 * Determinism: all advance/replay code reads ONLY from the state and
 * the supplied scalar load history. There is no Math.random, no Date,
 * no Map iteration order dependency. Two replays with identical inputs
 * produce identical states (verified by tests).
 */

export interface MaterialState {
  E: number;                // element count
  Fp:    Float32Array;      // length E*4 (col-major 2×2 per element)
  Sv:    Float32Array;      // length E*3 (xx, yy, xy)
  alpha: Float32Array;      // length E
  /** Step counter advanced by `advance()`. */
  step: number;
}

export interface Checkpoint {
  step: number;
  /** djb2 hash of the state arrays — for the determinism harness. */
  hash: number;
  Fp:    Float32Array;
  Sv:    Float32Array;
  alpha: Float32Array;
}

/** Construct a pristine state: Fp = I, Sv = 0, alpha = 0. */
export function initMaterialState(E: number): MaterialState {
  const Fp = new Float32Array(E * 4);
  for (let e = 0; e < E; e++) {
    Fp[e * 4 + 0] = 1; // F11
    Fp[e * 4 + 3] = 1; // F22
  }
  return {
    E,
    Fp,
    Sv: new Float32Array(E * 3),
    alpha: new Float32Array(E),
    step: 0,
  };
}

/** djb2 over the three buffers. Order-stable, position-sensitive. */
export function hashState(s: MaterialState): number {
  let h = 5381 >>> 0;
  const acc = (a: Float32Array) => {
    const view = new Uint32Array(a.buffer, a.byteOffset, a.length);
    for (let i = 0; i < view.length; i++) {
      h = (((h << 5) + h) ^ view[i]) >>> 0;
    }
  };
  acc(s.Fp); acc(s.Sv); acc(s.alpha);
  return h >>> 0;
}

/** Take a deep checkpoint of the state. */
export function checkpoint(s: MaterialState): Checkpoint {
  return {
    step: s.step,
    hash: hashState(s),
    Fp:    new Float32Array(s.Fp),
    Sv:    new Float32Array(s.Sv),
    alpha: new Float32Array(s.alpha),
  };
}

/** Restore the state from a checkpoint (in place). */
export function restore(s: MaterialState, cp: Checkpoint): void {
  if (cp.Fp.length !== s.Fp.length || cp.Sv.length !== s.Sv.length || cp.alpha.length !== s.alpha.length) {
    throw new Error("restore: checkpoint dimensions mismatch state");
  }
  s.Fp.set(cp.Fp);
  s.Sv.set(cp.Sv);
  s.alpha.set(cp.alpha);
  s.step = cp.step;
}

// ── Single-step constitutive update (deterministic) ─────────────────────

/**
 * Per-element scalar inputs applied this step. The caller (a real
 * simulator) would compute these from the displacement field; here we
 * accept them directly so the module is testable in isolation.
 */
export interface StepInputs {
  /** Trial strain increment per element, scalar (1D projection). */
  depsTrial: Float32Array;     // length E
  /** Time step. */
  dt: number;
  /** Yield stress σ_y. */
  yieldStress: number;
  /** Hardening modulus H. */
  hardening: number;
  /** Elastic modulus E (used to scale stress / yield check). */
  Emod: number;
  /** Viscoelastic relaxation time τ. */
  tau: number;
  /** Viscoelastic shear coupling η. */
  eta: number;
}

/**
 * Advance the material state by one step. Mutates `state` in place.
 *
 *   • Plastic:        radial return on a 1D projection of σ trial.
 *                     Fp[0] is updated as F11 ← F11 · exp(Δε_p).
 *                     alpha ← alpha + |Δε_p|.
 *   • Viscoelastic:   Sv ← Sv·exp(-dt/τ) + η·dε   (Maxwell).
 *
 * The 1D scalar projection is intentionally simple — this module exists
 * to test the *checkpoint/replay* contract, not to be a constitutive
 * law beauty contest.
 */
export function advance(state: MaterialState, inp: StepInputs): void {
  const { Fp, Sv, alpha, E } = state;
  const decay = Math.exp(-inp.dt / Math.max(inp.tau, 1e-9));
  for (let e = 0; e < E; e++) {
    const deps = inp.depsTrial[e];

    // Trial elastic stress increment.
    const sigTrial = inp.Emod * deps;
    const yieldNow = inp.yieldStress + inp.hardening * alpha[e];

    let depsP = 0;
    if (Math.abs(sigTrial) > yieldNow) {
      // Plastic flow: project back to yield surface.
      const overshoot = (Math.abs(sigTrial) - yieldNow) / (inp.Emod + inp.hardening);
      depsP = Math.sign(sigTrial) * overshoot;
      alpha[e] += Math.abs(depsP);
      // Update Fp[0] (F11) multiplicatively. F22 / F12 / F21 unchanged
      // in this 1D projection.
      Fp[e * 4 + 0] *= Math.exp(depsP);
    }

    // Viscoelastic Maxwell branch: relaxes prior stress + new strain rate.
    Sv[e * 3 + 0] = Sv[e * 3 + 0] * decay + inp.eta * deps;
    // Companion components track the same scalar projection in 2D.
    Sv[e * 3 + 1] = Sv[e * 3 + 1] * decay;
    Sv[e * 3 + 2] = Sv[e * 3 + 2] * decay;
  }
  state.step++;
}

// ── Rollback buffer ─────────────────────────────────────────────────────

export class RollbackBuffer {
  readonly capacity: number;
  private buf: Checkpoint[] = [];

  constructor(capacity = 32) {
    this.capacity = Math.max(1, capacity);
  }

  push(cp: Checkpoint): void {
    this.buf.push(cp);
    if (this.buf.length > this.capacity) this.buf.shift();
  }

  /** Most recent checkpoint, or undefined. */
  latest(): Checkpoint | undefined {
    return this.buf[this.buf.length - 1];
  }

  /** Pop the latest checkpoint and return it (one-step rollback). */
  pop(): Checkpoint | undefined {
    return this.buf.pop();
  }

  /** Find the last checkpoint at or before `step`. */
  findAtOrBefore(step: number): Checkpoint | undefined {
    for (let i = this.buf.length - 1; i >= 0; i--) {
      if (this.buf[i].step <= step) return this.buf[i];
    }
    return undefined;
  }

  size(): number { return this.buf.length; }
  clear(): void  { this.buf.length = 0; }
  toArray(): readonly Checkpoint[] { return this.buf.slice(); }
}

// ── Replay harness ──────────────────────────────────────────────────────

/**
 * Replay a recorded sequence of step inputs from a checkpoint. Returns
 * the final state hash so callers can verify bit-identical reproduction.
 */
export function replay(
  cp: Checkpoint,
  E: number,
  inputs: StepInputs[],
): { finalHash: number; state: MaterialState } {
  const s = initMaterialState(E);
  restore(s, cp);
  for (const inp of inputs) advance(s, inp);
  return { finalHash: hashState(s), state: s };
}
