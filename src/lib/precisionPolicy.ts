/**
 * precisionPolicy.ts
 * WebGPU has no native f64 — all storage/uniform/compute math is f32.
 * Rather than fail loudly when a caller hands us a Float64Array, this
 * module either:
 *   • throws (mode "strict")
 *   • silently truncates to f32 and tracks the worst observed loss
 *     (mode "auto-downgrade")
 *
 * The downgrade reports include enough info for a UI banner:
 *   • whether anything was downgraded this session
 *   • per-conversion: array length, max abs value, max ULP loss,
 *     whether any value overflowed f32 range
 *   • a rolling list of the most recent N events
 */

export type PrecisionMode = "strict" | "auto-downgrade";

export interface PrecisionLossEvent {
  /** Caller-supplied tag, e.g. "vertices", "Fp", "Sv". */
  label: string;
  /** Source array length. */
  length: number;
  /** Maximum |x| in the source array. */
  maxAbs: number;
  /** Largest |f64 - f32| observed (absolute units). */
  maxAbsLoss: number;
  /** Largest relative loss = |f64-f32| / max(|f64|, eps). */
  maxRelLoss: number;
  /** True iff any element exceeded ±3.4e38 (f32 max). */
  overflow: boolean;
  /** ms since epoch (UI sorting, not for determinism). */
  at: number;
}

export interface PrecisionStats {
  mode: PrecisionMode;
  /** Whether the host device is f32-only (true on WebGPU). */
  hostIsF32Only: boolean;
  /** Total downgrade calls since last reset. */
  conversions: number;
  /** Total elements downgraded. */
  elementsConverted: number;
  /** Worst-ever relative loss across all events. */
  worstRelLoss: number;
  /** Whether any conversion overflowed f32 range. */
  anyOverflow: boolean;
  /** Most recent events (oldest → newest), bounded by `historySize`. */
  events: PrecisionLossEvent[];
}

export interface PrecisionPolicyOptions {
  mode?: PrecisionMode;
  /** Override host detection — useful for tests. */
  hostIsF32Only?: boolean;
  historySize?: number;
}

const F32_MAX = 3.4028234663852886e38;

export class PrecisionPolicy {
  mode: PrecisionMode;
  readonly hostIsF32Only: boolean;
  readonly historySize: number;
  private stats: PrecisionStats;

  constructor(opts: PrecisionPolicyOptions = {}) {
    this.mode = opts.mode ?? "auto-downgrade";
    this.hostIsF32Only = opts.hostIsF32Only ?? detectF32Only();
    this.historySize = Math.max(1, opts.historySize ?? 16);
    this.stats = this.freshStats();
  }

  setMode(mode: PrecisionMode): void { this.mode = mode; this.stats.mode = mode; }
  reset(): void { this.stats = this.freshStats(); }
  snapshot(): PrecisionStats {
    return { ...this.stats, events: this.stats.events.slice() };
  }

  /**
   * Convert (or pass through) an arbitrary numeric array intended for
   * the GPU. Returns a Float32Array.
   *
   * - If input is already a Float32Array → returned as-is (no copy).
   * - If host is NOT f32-only AND mode === "strict" with a Float64Array
   *   → still coerced to Float32Array (caller asked for GPU upload).
   *   The "strict" mode just throws when downgrade WOULD lose precision.
   */
  toGpuFloat32(input: ArrayLike<number> & { BYTES_PER_ELEMENT?: number }, label: string): Float32Array {
    if (input instanceof Float32Array) return input;
    const src = input as ArrayLike<number>;

    let maxAbs = 0, maxAbsLoss = 0, maxRelLoss = 0, overflow = false;
    const out = new Float32Array(src.length);
    const probe = new Float32Array(1);
    for (let i = 0; i < src.length; i++) {
      const x = src[i];
      probe[0] = x;
      const x32 = probe[0];
      out[i] = x32;
      const a = Math.abs(x);
      if (a > maxAbs) maxAbs = a;
      if (a > F32_MAX) overflow = true;
      const loss = Math.abs(x - x32);
      if (loss > maxAbsLoss) maxAbsLoss = loss;
      const rel = loss / Math.max(a, 1e-300);
      if (Number.isFinite(rel) && rel > maxRelLoss) maxRelLoss = rel;
    }

    const event: PrecisionLossEvent = {
      label, length: src.length, maxAbs, maxAbsLoss, maxRelLoss, overflow,
      at: Date.now(),
    };

    const meaningfulLoss = maxAbsLoss > 0 || overflow;

    if (this.mode === "strict" && meaningfulLoss) {
      throw new Error(
        `precisionPolicy[strict]: refused to downgrade "${label}" — ` +
        `maxRelLoss=${maxRelLoss.toExponential(2)}, overflow=${overflow}`
      );
    }

    // Track only when an actual downgrade happened (input not f32).
    this.stats.conversions++;
    this.stats.elementsConverted += src.length;
    if (maxRelLoss > this.stats.worstRelLoss) this.stats.worstRelLoss = maxRelLoss;
    if (overflow) this.stats.anyOverflow = true;
    this.stats.events.push(event);
    if (this.stats.events.length > this.historySize) this.stats.events.shift();

    return out;
  }

  /** True iff the user should see the warning banner. */
  shouldShowBanner(): boolean {
    if (!this.hostIsF32Only) return false;
    return this.stats.conversions > 0 &&
      (this.stats.worstRelLoss > 1e-7 || this.stats.anyOverflow);
  }

  private freshStats(): PrecisionStats {
    return {
      mode: this.mode,
      hostIsF32Only: this.hostIsF32Only,
      conversions: 0,
      elementsConverted: 0,
      worstRelLoss: 0,
      anyOverflow: false,
      events: [],
    };
  }
}

function detectF32Only(): boolean {
  // WebGPU never exposes f64 — if it is present, we are f32-only.
  // (The "shader-f16" feature shrinks further but doesn't widen.)
  if (typeof navigator !== "undefined" && (navigator as Navigator & { gpu?: unknown }).gpu) return true;
  return false;
}

/** Module-level singleton most callers can share. */
let _shared: PrecisionPolicy | null = null;
export function getPrecisionPolicy(): PrecisionPolicy {
  if (!_shared) _shared = new PrecisionPolicy();
  return _shared;
}
