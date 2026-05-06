/**
 * pairDedup.ts
 * ──────────────────────────────────────────────────────────────────────────
 * Deterministic, overflow-safe deduplication pipeline for candidate pairs
 * emitted by a broad-phase (uniform grid, BVH, sort-and-sweep).
 *
 * Why this exists
 * ---------------
 * Most broad-phases over-emit:
 *   • Uniform grid: a pair (i, j) is emitted once per *shared cell*, so
 *     bodies overlapping multiple cells generate up to 4 duplicates in 2-D.
 *   • BVH / SAP: with thick AABBs or hierarchy overlap, the same (i, j)
 *     can be re-discovered along multiple traversal paths.
 *   • Workers/GPU writeback: parallel emitters race to the same output
 *     buffer; even with atomics, the resulting list can carry duplicates
 *     because each worker filters only its own slice.
 *
 * Feeding a duplicated stream into the narrow-phase wastes work AND
 * (worse) breaks contact-resolution determinism — applying λ twice is
 * not idempotent. The narrow-phase therefore needs a stream of UNIQUE,
 * canonically-ordered pairs.
 *
 * Determinism contract
 * --------------------
 *   1. Output ordering is canonical: pairs sorted ascending by
 *      `(min(i,j), max(i,j))`. Insertion order does NOT affect output.
 *   2. Each pair (i, j) appears AT MOST ONCE, with i < j (we always
 *      normalize so the smaller index is first — collapses the (i,j)
 *      vs (j,i) ambiguity that broad-phases emit).
 *   3. Self-pairs (i == i) are dropped.
 *   4. Out-of-range indices (i ≥ N or j ≥ N) are dropped.
 *   5. Saturation behavior is explicit: when the unique-pair count would
 *      exceed `maxPairs`, the FIRST `maxPairs` pairs (in canonical order)
 *      are kept and `overflow.dropped > 0` is reported. This guarantees
 *      stable truncation across runs — the same input ALWAYS produces
 *      the same kept-set, even when truncated.
 *
 * Overflow safety
 * ---------------
 *   • The hash key uses `BigInt`: `(BigInt(a) << 32n) | BigInt(b)`.
 *     For N up to 2³¹, both halves fit in 32 bits and the key is unique.
 *     Pure JS numbers would lose mantissa precision above 2⁵³ ≈ N²
 *     scenarios; BigInt is the only safe choice.
 *   • `Set<bigint>` is O(1) average insert/lookup — the dedupe step is
 *     O(P) over input pair count, independent of N.
 *   • Output is packed into a single `Uint32Array` of length 2·M (no
 *     per-pair object allocation) to keep GC out of the hot path.
 *
 * Output format
 * -------------
 * Returns a `DedupResult { pairs: Uint32Array, count, overflow }` where
 * `pairs` is i₀, j₀, i₁, j₁, …, sorted canonically. The narrow-phase
 * solver in `gpuNarrowPhase.ts` consumes exactly this layout.
 */

export interface DedupOverflow {
  /** Pairs that were dropped because the unique count exceeded maxPairs. */
  dropped: number;
  /** Pairs dropped because they were out of [0, N) or self-pairs. */
  invalid: number;
  /** Pairs dropped because they were duplicates of an already-seen pair. */
  duplicates: number;
}

export interface DedupResult {
  /** Packed [i, j, i, j, ...]; length = 2 * count. */
  pairs: Uint32Array;
  /** Number of UNIQUE canonical pairs in `pairs`. */
  count: number;
  /** Per-category drop counts. Sum should equal `inputCount - count`. */
  overflow: DedupOverflow;
  /** True iff `dropped > 0` (kept-set truncated). */
  saturated: boolean;
}

export interface DedupOptions {
  /** Total body count; pairs with i ≥ N or j ≥ N are dropped. */
  N: number;
  /** Hard cap on output pair count. Required — overflow MUST be explicit. */
  maxPairs: number;
}

/**
 * Pack two uint32 indices into a single bigint key with i in the high
 * 32 bits and j in the low 32 bits. After normalization (i < j) the key
 * uniquely identifies an unordered pair.
 *
 * Why bigint: 2N indices can each reach ~10⁷ in real workloads. The
 * combined key (i·2³² + j) overflows Number above ~2⁵³, which happens
 * once N exceeds ~94K. BigInt has no such ceiling.
 */
function packKey(i: number, j: number): bigint {
  return (BigInt(i) << 32n) | BigInt(j);
}

/**
 * Generic dedup over an iterable of (i, j) tuples. Used by the array
 * and packed-Uint32Array entry points below.
 *
 * Algorithm:
 *   1. Stream input → normalize (i,j) so i < j → validate → hash-set
 *      insert. Track per-category drop counts.
 *   2. After streaming: extract keys from the set, sort numerically
 *      (canonical order), unpack back to Uint32Array.
 *
 * Sorting in step 2 is what makes the output deterministic: even if a
 * parallel emitter wrote pairs in different orders across runs, the
 * sorted output is bit-identical.
 */
function dedupCore(
  input: Iterable<readonly [number, number]>,
  inputCount: number,
  opts: DedupOptions,
): DedupResult {
  if (opts.maxPairs < 0) throw new Error("maxPairs must be >= 0");
  if (opts.N < 0)        throw new Error("N must be >= 0");

  const seen = new Set<bigint>();
  let invalid = 0;
  let duplicates = 0;

  for (const [rawI, rawJ] of input) {
    // Normalize unordered pair: smaller index first.
    let i = rawI | 0;
    let j = rawJ | 0;
    // Drop negative / NaN / out-of-range / self-pairs.
    if (i < 0 || j < 0 || i >= opts.N || j >= opts.N || i === j) {
      invalid++;
      continue;
    }
    if (i > j) { const t = i; i = j; j = t; }
    const key = packKey(i, j);
    if (seen.has(key)) { duplicates++; continue; }
    seen.add(key);
  }

  // Canonical sort. Comparing bigints with subtraction would overflow Number,
  // so use the three-way comparator.
  const keys = Array.from(seen).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));

  const uniqueCount = keys.length;
  const keptCount   = Math.min(uniqueCount, opts.maxPairs);
  const dropped     = uniqueCount - keptCount;

  const pairs = new Uint32Array(keptCount * 2);
  const LOW32 = (1n << 32n) - 1n;
  for (let k = 0; k < keptCount; k++) {
    const key = keys[k];
    pairs[k * 2]     = Number((key >> 32n) & LOW32);
    pairs[k * 2 + 1] = Number(key & LOW32);
  }

  // Sanity: invariant inputCount = count + dropped + duplicates + invalid.
  // Useful when caller wants to alert on systemic over-emit.
  void inputCount;

  return {
    pairs,
    count: keptCount,
    overflow: { dropped, invalid, duplicates },
    saturated: dropped > 0,
  };
}

/**
 * Deduplicate an array of CandidatePair-like objects.
 * Accepts the broad-phase's natural output shape.
 */
export function dedupPairs(
  input: ReadonlyArray<{ i: number; j: number }>,
  opts: DedupOptions,
): DedupResult {
  function* iter(): Iterable<readonly [number, number]> {
    for (const p of input) yield [p.i, p.j];
  }
  return dedupCore(iter(), input.length, opts);
}

/**
 * Deduplicate a packed Uint32Array of [i,j,i,j,...] pairs.
 * Faster path: avoids the per-pair object allocation when callers (e.g.
 * GPU readback) already have a flat buffer. Length must be even.
 */
export function dedupPacked(
  input: Uint32Array,
  opts: DedupOptions,
): DedupResult {
  if (input.length % 2 !== 0) {
    throw new Error("dedupPacked: input length must be even (i,j pairs)");
  }
  function* iter(): Iterable<readonly [number, number]> {
    for (let k = 0; k < input.length; k += 2) {
      yield [input[k], input[k + 1]];
    }
  }
  return dedupCore(iter(), input.length / 2, opts);
}

// ── Streaming dedup (multi-source / multi-batch) ─────────────────────────

/**
 * For multi-emitter pipelines (per-worker buffers, multi-frame coalesce,
 * GPU + CPU broad-phase merging) it's wasteful to re-allocate the Set
 * per call. `PairDedupStream` reuses one `Set<bigint>` across `add()`s
 * and finalizes once via `flush()`.
 *
 * Saturation policy: once the unique-pair count reaches `maxPairs`,
 * subsequent `add()` calls still validate + count duplicates correctly,
 * but new unique pairs are dropped (not stored). This keeps memory
 * bounded under adversarial inputs (e.g. a runaway emitter producing
 * 10⁹ candidate pairs).
 */
export class PairDedupStream {
  private readonly seen = new Set<bigint>();
  private readonly opts: DedupOptions;
  private invalid = 0;
  private duplicates = 0;
  private dropped = 0;
  private inputs = 0;

  constructor(opts: DedupOptions) {
    if (opts.maxPairs < 0) throw new Error("maxPairs must be >= 0");
    if (opts.N < 0)        throw new Error("N must be >= 0");
    this.opts = opts;
  }

  /** Add a single pair. Returns true iff it was a NEW unique pair stored. */
  addPair(rawI: number, rawJ: number): boolean {
    this.inputs++;
    let i = rawI | 0, j = rawJ | 0;
    if (i < 0 || j < 0 || i >= this.opts.N || j >= this.opts.N || i === j) {
      this.invalid++;
      return false;
    }
    if (i > j) { const t = i; i = j; j = t; }
    const key = packKey(i, j);
    if (this.seen.has(key)) { this.duplicates++; return false; }
    if (this.seen.size >= this.opts.maxPairs) {
      // Bounded-memory saturation: still count, but don't store.
      this.dropped++;
      return false;
    }
    this.seen.add(key);
    return true;
  }

  /** Add many pairs from an array of {i,j}. */
  addArray(pairs: ReadonlyArray<{ i: number; j: number }>): void {
    for (const p of pairs) this.addPair(p.i, p.j);
  }

  /** Add many pairs from a packed Uint32Array of [i,j,...]. */
  addPacked(packed: Uint32Array): void {
    if (packed.length % 2 !== 0) {
      throw new Error("addPacked: length must be even");
    }
    for (let k = 0; k < packed.length; k += 2) {
      this.addPair(packed[k], packed[k + 1]);
    }
  }

  /** Total raw pairs offered (across all add* calls). */
  get inputCount(): number { return this.inputs; }

  /** Current unique stored pair count (≤ maxPairs). */
  get uniqueCount(): number { return this.seen.size; }

  /**
   * Finalize and emit the canonical, sorted Uint32Array. The stream is
   * NOT cleared by `flush()` (call `reset()` to clear) so callers can
   * peek mid-pipeline without losing state.
   */
  flush(): DedupResult {
    const keys = Array.from(this.seen).sort((a, b) => (a < b ? -1 : a > b ? 1 : 0));
    const pairs = new Uint32Array(keys.length * 2);
    const LOW32 = (1n << 32n) - 1n;
    for (let k = 0; k < keys.length; k++) {
      pairs[k * 2]     = Number((keys[k] >> 32n) & LOW32);
      pairs[k * 2 + 1] = Number(keys[k] & LOW32);
    }
    return {
      pairs, count: keys.length,
      overflow: {
        dropped: this.dropped,
        invalid: this.invalid,
        duplicates: this.duplicates,
      },
      saturated: this.dropped > 0,
    };
  }

  /** Drop all state — ready to consume the next frame. */
  reset(): void {
    this.seen.clear();
    this.invalid = 0;
    this.duplicates = 0;
    this.dropped = 0;
    this.inputs = 0;
  }
}
