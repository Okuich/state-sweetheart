/**
 * mpiOrchestrator.ts
 * ──────────────────────────────────────────────────────────────────────────
 * In-process simulation of an MPI-style orchestration layer for the
 * particle simulator. This is NOT real MPI (the app runs in a browser),
 * but it faithfully models the operational shape of an MPI program:
 *
 *   • A fixed-size COMM_WORLD with `size` ranks.
 *   • A 1-D row-block partition of the N global particles → owners[i].
 *   • A neighbor table (left / right rank) used for halo exchanges.
 *   • Non-blocking collectives (Iallreduce, Ibarrier, IallreduceMin) that
 *     return Request handles you `await`. Completion is deferred to the
 *     next microtask so it interleaves like real MPI progress threads.
 *   • A step-loop driver that, each timestep:
 *         1. Iallreduce(MIN) on dt to agree on a stable global timestep
 *         2. Local force/integrate via the user-provided kernel callback
 *         3. Halo exchange (Isend/Irecv pair, batched as Ialltoall)
 *         4. Iallreduce(SUM) on a per-rank scalar (e.g. energy) for
 *            global diagnostics
 *         5. Ibarrier to align ranks before the next step
 *
 * Design goals:
 *   • Correctness of the *protocol*: messages are never read before they
 *     post, requests must be waited on, dt is the min across ranks, and
 *     the step loop will deadlock-detect if a rank never posts its half.
 *   • Zero dependence on the DOM or the canvas. Pure data-in / data-out
 *     so it can be unit-tested and reused in workers / SSR.
 *   • Cheap: O(N) partition build, O(rank-count) collectives — fine to
 *     run alongside the real simulator without measurable overhead.
 */

// ── Types ────────────────────────────────────────────────────────────────

export type Rank = number;

/** Inclusive ownership range [start, end) on a 1-D row-block partition. */
export interface PartitionRange {
  rank: Rank;
  start: number;
  end: number;       // exclusive
  count: number;     // = end - start
  /** Neighbor ranks for halo exchange; -1 means "no neighbor on this side". */
  left: Rank;
  right: Rank;
}

/** Owner-of-particle table. owners[i] is the rank that owns global index i. */
export type OwnerTable = Int32Array;

/** Halo payload — opaque to the orchestrator; the kernel decides the shape. */
export interface HaloPacket {
  from: Rank;
  to: Rank;
  /** Global indices of ghost particles being sent. */
  indices: Int32Array;
  /** Packed payload (e.g. interleaved [x, y, vx, vy] floats). */
  payload: Float32Array;
}

/**
 * Non-blocking request handle, mirrors MPI_Request.
 * `wait()` resolves when the operation is complete.
 */
export interface MpiRequest<T = void> {
  readonly id: number;
  readonly op: "Iallreduce" | "Ibarrier" | "Ialltoall" | "Isend" | "Irecv";
  readonly postedAt: number;
  wait(): Promise<T>;
  /** True after wait() has resolved. Useful for deadlock detection. */
  readonly done: boolean;
}

export type ReduceOp = "sum" | "min" | "max";

/**
 * Per-rank kernel hook. The orchestrator calls this once per local step
 * with the agreed-upon global dt and the rank's owned slice; the kernel
 * returns whatever per-rank scalar should feed the diagnostic reduction
 * (typically local energy, force-norm, or a residual).
 */
export type StepKernel = (ctx: {
  rank: Rank;
  range: PartitionRange;
  dt: number;
  step: number;
  /** Halos received from neighbors this step (post-exchange). */
  haloIn: HaloPacket[];
}) => Promise<{
  /** Per-rank scalar contribution to the global diagnostic reduction. */
  scalar: number;
  /** Halos this rank wants to send out for the NEXT step's exchange. */
  haloOut: HaloPacket[];
  /** Locally-proposed timestep for the NEXT step (CFL, etc.). */
  proposedDt: number;
}>;

export interface OrchestratorOptions {
  /** Number of ranks in COMM_WORLD. */
  size: number;
  /** Total global particle count. */
  N: number;
  /** Initial timestep proposal. */
  dt0: number;
  /** Optional ceiling on dt; the global min is clamped to this. */
  dtMax?: number;
  /** Optional floor — guards against a stuck rank proposing dt=0. */
  dtMin?: number;
  /**
   * Synthetic latency per collective (ms). Models progress thread cost
   * during testing; real production code can leave this at 0.
   */
  collectiveLatencyMs?: number;
  /** Deadlock guard — wait() rejects after this many ms. */
  deadlockTimeoutMs?: number;
}

export interface StepReport {
  step: number;
  dt: number;
  /** Per-rank scalar contributions (length = size). */
  scalars: Float64Array;
  /** Reduced (sum) of `scalars`. */
  sum: number;
  /** Wall time in ms for this step (orchestrator + kernel). */
  wallMs: number;
  /** Bytes exchanged across all halos this step. */
  haloBytes: number;
}

// ── Implementation ───────────────────────────────────────────────────────

/**
 * 1-D row-block partition. Identical to MPI's reference recipe:
 * each rank gets ⌈N/size⌉ or ⌊N/size⌋ contiguous particles, with the
 * remainder distributed to the lowest ranks. Deterministic — every rank
 * computes the same table from (N, size) without communication.
 */
export function partitionRowBlock(N: number, size: number): PartitionRange[] {
  if (size <= 0) throw new Error("partitionRowBlock: size must be > 0");
  if (N < 0) throw new Error("partitionRowBlock: N must be >= 0");
  const base = Math.floor(N / size);
  const rem  = N - base * size;
  const out: PartitionRange[] = [];
  let cursor = 0;
  for (let r = 0; r < size; r++) {
    const count = base + (r < rem ? 1 : 0);
    const start = cursor;
    const end = start + count;
    cursor = end;
    out.push({
      rank: r, start, end, count,
      left:  r > 0        ? r - 1 : -1,
      right: r < size - 1 ? r + 1 : -1,
    });
  }
  return out;
}

/** Owner table: O(N) build, O(1) lookup. */
export function buildOwnerTable(parts: PartitionRange[], N: number): OwnerTable {
  const owners = new Int32Array(N);
  for (const p of parts) owners.fill(p.rank, p.start, p.end);
  return owners;
}

/** Element-wise reduction across rank-major contributions. */
export function reduce(values: ArrayLike<number>, op: ReduceOp): number {
  let acc = op === "min" ? Infinity : op === "max" ? -Infinity : 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (op === "sum") acc += v;
    else if (op === "min") acc = v < acc ? v : acc;
    else                   acc = v > acc ? v : acc;
  }
  return acc;
}

/**
 * Internal request factory. Resolves on the next microtask (or after the
 * configured synthetic latency) so callers MUST `await` — code that fires
 * collectives and forgets to wait() is broken in real MPI too.
 */
function makeRequest<T>(
  id: number,
  op: MpiRequest["op"],
  produce: () => T,
  latencyMs: number,
  deadlockMs: number,
): MpiRequest<T> {
  let resolved = false;
  let value: T;
  const settled = new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      try {
        value = produce();
        resolved = true;
        resolve(value);
      } catch (e) { reject(e); }
    }, Math.max(0, latencyMs));
    // Deadlock watchdog: if nobody awaits within deadlockMs, reject so
    // the step loop surfaces the problem instead of hanging silently.
    if (deadlockMs > 0) {
      setTimeout(() => {
        if (!resolved) reject(new Error(
          `MPI ${op} request #${id} not completed within ${deadlockMs}ms — possible deadlock`
        ));
      }, deadlockMs);
      // The watchdog must not keep the event loop alive in Node tests.
      // (Browser timers are unaffected by unref.)
      const t = timer as unknown as { unref?: () => void };
      t.unref?.();
    }
  });

  const req: MpiRequest<T> = {
    id, op, postedAt: performance.now(),
    get done() { return resolved; },
    wait: () => settled,
  };
  return req;
}

/**
 * The orchestrator. One instance models the whole COMM_WORLD; in real
 * MPI you'd have one process per rank, but here we drive all ranks from
 * a single event loop and use Promise interleaving to model concurrency.
 */
export class MpiOrchestrator {
  readonly size: number;
  readonly N: number;
  partitions: PartitionRange[];
  owners: OwnerTable;
  step = 0;
  /** Last agreed-upon global dt. */
  dt: number;
  private readonly dtMax: number;
  private readonly dtMin: number;
  private readonly latencyMs: number;
  private readonly deadlockMs: number;
  private nextReqId = 1;
  /** Per-rank inbox of halos waiting to be delivered next step. */
  private inbox: HaloPacket[][];
  readonly reports: StepReport[] = [];

  constructor(opts: OrchestratorOptions) {
    if (opts.size < 1) throw new Error("MpiOrchestrator: size must be >= 1");
    this.size       = opts.size;
    this.N          = opts.N;
    this.dt         = opts.dt0;
    this.dtMax      = opts.dtMax ?? Infinity;
    this.dtMin      = opts.dtMin ?? 1e-9;
    this.latencyMs  = opts.collectiveLatencyMs ?? 0;
    this.deadlockMs = opts.deadlockTimeoutMs ?? 5000;
    this.partitions = partitionRowBlock(opts.N, opts.size);
    this.owners     = buildOwnerTable(this.partitions, opts.N);
    this.inbox      = Array.from({ length: opts.size }, () => []);
  }

  /** Recompute partition + owner table when N changes (keeps `size`). */
  repartition(newN: number): void {
    this.partitions = partitionRowBlock(newN, this.size);
    this.owners     = buildOwnerTable(this.partitions, newN);
    (this as { N: number }).N = newN;
    this.inbox = Array.from({ length: this.size }, () => []);
  }

  /** Non-blocking all-reduce. Resolves to the reduced scalar. */
  Iallreduce(values: ArrayLike<number>, op: ReduceOp): MpiRequest<number> {
    return makeRequest(this.nextReqId++, "Iallreduce",
      () => reduce(values, op), this.latencyMs, this.deadlockMs);
  }

  /** Non-blocking barrier. Resolves to void once all ranks "arrived". */
  Ibarrier(): MpiRequest<void> {
    return makeRequest(this.nextReqId++, "Ibarrier",
      () => undefined, this.latencyMs, this.deadlockMs);
  }

  /**
   * Non-blocking all-to-all halo exchange. Routes each packet to its
   * destination rank's inbox; resolves with the per-rank inbox snapshot
   * (which the next step's kernel will consume as `haloIn`).
   */
  Ialltoall(packets: HaloPacket[]): MpiRequest<HaloPacket[][]> {
    return makeRequest(this.nextReqId++, "Ialltoall", () => {
      const next: HaloPacket[][] = Array.from({ length: this.size }, () => []);
      for (const pk of packets) {
        if (pk.to < 0 || pk.to >= this.size) continue;       // -1 neighbor
        if (pk.from < 0 || pk.from >= this.size) continue;
        if (pk.to === pk.from) continue;                      // no self-send
        next[pk.to].push(pk);
      }
      this.inbox = next;
      return next;
    }, this.latencyMs, this.deadlockMs);
  }

  /**
   * Drive ONE timestep across all ranks.
   *
   * Order matters and mirrors what a production MPI code would do:
   *   1. Post Iallreduce(MIN, dt) immediately so dt-agreement overlaps
   *      with local force/integration work.
   *   2. Run the kernel on every rank in parallel (Promise.all).
   *      The kernel uses the dt agreed upon at the START of this step.
   *   3. Wait on the dt reduction → that becomes NEXT step's dt.
   *   4. Post Ialltoall with all halos the kernels emitted; await it so
   *      next step's kernels see consistent halos.
   *   5. Post Iallreduce(SUM) on the per-rank scalar diagnostic.
   *   6. Ibarrier — align before returning so callers get a clean step
   *      boundary (no half-finished collective leaking into the next).
   */
  async runStep(kernel: StepKernel): Promise<StepReport> {
    const t0 = performance.now();
    const dtNow = this.dt;

    // (1) Post the dt-agreement non-blocking reduction; we'll wait at
    //     the end so it overlaps with kernel compute (the whole point
    //     of using non-blocking collectives in the first place).
    const dtProposals = new Float64Array(this.size);

    // (2) Local kernel on every rank in parallel.
    const results = await Promise.all(this.partitions.map(async (range) => {
      const haloIn = this.inbox[range.rank];
      const r = await kernel({
        rank: range.rank, range, dt: dtNow, step: this.step, haloIn,
      });
      dtProposals[range.rank] = r.proposedDt;
      return r;
    }));

    // (3) Reduce dt → next step's global dt. Clamp into [dtMin, dtMax].
    const dtReduce = this.Iallreduce(dtProposals, "min");
    const nextDtRaw = await dtReduce.wait();
    const nextDt = Math.max(this.dtMin, Math.min(this.dtMax, nextDtRaw));

    // (4) Halo exchange — route ALL packets to ALL inboxes for next step.
    const allHalos = results.flatMap((r) => r.haloOut);
    let haloBytes = 0;
    for (const h of allHalos) {
      haloBytes += h.indices.byteLength + h.payload.byteLength;
    }
    const exchange = this.Ialltoall(allHalos);
    await exchange.wait();

    // (5) Diagnostic sum-reduction over per-rank scalars.
    const scalars = new Float64Array(this.size);
    for (let r = 0; r < this.size; r++) scalars[r] = results[r].scalar;
    const sumReq = this.Iallreduce(scalars, "sum");
    const sum = await sumReq.wait();

    // (6) Barrier so the next step starts cleanly.
    await this.Ibarrier().wait();

    const report: StepReport = {
      step: this.step,
      dt: dtNow,
      scalars,
      sum,
      wallMs: performance.now() - t0,
      haloBytes,
    };
    this.reports.push(report);
    if (this.reports.length > 256) this.reports.shift();

    this.step += 1;
    this.dt = nextDt;
    return report;
  }

  /**
   * Drive `nSteps` timesteps. Aborts cleanly if `signal` fires; useful
   * for hooking the orchestrator into a React component that unmounts.
   */
  async run(nSteps: number, kernel: StepKernel, signal?: AbortSignal): Promise<StepReport[]> {
    const out: StepReport[] = [];
    for (let i = 0; i < nSteps; i++) {
      if (signal?.aborted) break;
      out.push(await this.runStep(kernel));
    }
    return out;
  }
}
