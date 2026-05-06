import { createFileRoute } from "@tanstack/react-router";
import { useRef, useState } from "react";
import { PhysicsCanvas, type SimParams, type ValidationReport } from "@/components/PhysicsCanvas";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/")({
  component: Index,
  head: () => ({
    meta: [
      { title: "PhysicsState — Interactive N-body Visualizer" },
      {
        name: "description",
        content:
          "Real-time particle physics sandbox: gravity, damping, attractors. Click and drag to push or pull particles.",
      },
    ],
  }),
});

function Field({
  label,
  value,
  unit,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  unit?: string;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-xs uppercase tracking-[0.18em]">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-primary tabular-nums">
          {value.toFixed(step < 1 ? 2 : 0)}
          {unit && <span className="text-muted-foreground ml-1">{unit}</span>}
        </span>
      </div>
      <Slider
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={([v]) => onChange(v)}
      />
    </div>
  );
}

function Index() {
  const [params, setParams] = useState<SimParams>({
    gravity: 60,
    damping: 0.4,
    attractor: 1.2,
    particleCount: 400,
    trail: 0.22,
    paused: false,
    springK: 80,
    restLength: 40,
    edgesPerNode: 2,
    showEdges: true,
    pairwiseStrength: 200,
    pairwiseRadius: 50,
    integrator: "semi-euler",
    dtype: "float32",
    device: "cpu",
    constraintIters: 0,
    field: "none",
    fieldStrength: 0.6,
    subSteps: 1,
    workers: 4,
    showPartitions: true,
    optimize: false,
    objectiveLR: 0.05,
    pairwiseMode: "lj",
    boundary: "walls",
    forceViz: "off",
  });
  const [resetKey, setResetKey] = useState(0);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [loss, setLoss] = useState<number | null>(null);
  const pointerRef = useRef({ x: 0, y: 0, active: false, mode: 1 as 1 | -1 });

  const update = <K extends keyof SimParams>(k: K, v: SimParams[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

  return (
    <main className="relative min-h-screen overflow-hidden">
      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-5 lg:px-10">
        <div className="flex items-center gap-3">
          <div className="h-2.5 w-2.5 rounded-full bg-primary glow-mint animate-pulse" />
          <div className="text-xs uppercase tracking-[0.32em] text-muted-foreground">
            PhysicsState
          </div>
          <div className="hidden sm:block text-xs text-muted-foreground/70">
            / N-body sandbox
          </div>
        </div>
        <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          <span className={`h-1.5 w-1.5 rounded-full ${params.device === "webgpu" ? "bg-accent" : "bg-primary"}`} />
          <span>device · {params.device}</span>
          <span className="text-muted-foreground/50">/</span>
          <span>dtype · {params.dtype}</span>
        </div>
      </header>

      {/* Hero / Title */}
      <section className="relative z-10 px-6 lg:px-10 pb-6 max-w-4xl">
        <h1 className="font-display text-4xl md:text-6xl font-bold leading-[0.95] text-glow">
          Forces.
          <br />
          <span className="text-primary">Integrated</span> in real time.
        </h1>
        <p className="mt-4 text-sm md:text-base text-muted-foreground max-w-xl">
          A live visualization of <code className="text-primary">PhysicsState</code> —
          positions, velocities, mass and force fields, integrated each frame.
          <span className="text-foreground/80">
            {" "}Click to attract, right-click to repel.
          </span>
        </p>
      </section>

      {/* Canvas */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-4 h-[58vh] rounded-xl border border-border bg-card backdrop-blur-sm overflow-hidden">
        <PhysicsCanvas key={resetKey} params={params} pointerRef={pointerRef} onValidation={setValidation} onLoss={setLoss} />
        {/* HUD — SoA memory layout */}
        <div className="pointer-events-none absolute left-4 top-4 flex flex-col gap-0.5 font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          <div className="text-accent/80 mb-1">struct PhysicsState · SoA</div>
          <div><span className="text-primary">x ,y</span>   float* [{params.particleCount}]</div>
          <div><span className="text-primary">vx,vy</span>  float* [{params.particleCount}]</div>
          <div><span className="text-primary">fx,fy</span>  float* [{params.particleCount}]</div>
          <div><span className="text-primary">m</span>      float* [{params.particleCount}]</div>
          <div><span className="text-accent">edge_i,j</span> int* [{params.particleCount * params.edgesPerNode}]</div>
          <div><span className="text-accent">rest_len</span> float* [{params.particleCount * params.edgesPerNode}]</div>
          <div className="text-muted-foreground/60 mt-1">{params.dtype === "float64" ? "f64" : "f32"} · {params.device}</div>
          <div className="mt-2 text-accent/80">launch · reset_forces</div>
          <div className="text-foreground/70">&lt;&lt;&lt;{Math.ceil(params.particleCount / 256)}, 256&gt;&gt;&gt;</div>
          <div className="text-muted-foreground/60">grid · {Math.ceil(params.particleCount / 256)} blk × 256 thr = {Math.ceil(params.particleCount / 256) * 256} threads</div>
        </div>

        {/* Validation badge */}
        <div className="pointer-events-none absolute right-4 top-4 max-w-[260px] rounded-md border border-border bg-background/70 px-3 py-2 backdrop-blur-md">
          <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.22em]">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                validation?.ok
                  ? "bg-primary glow-mint animate-pulse"
                  : validation
                  ? "bg-destructive"
                  : "bg-muted-foreground"
              }`}
            />
            <span className={validation?.ok ? "text-primary" : validation ? "text-destructive" : "text-muted-foreground"}>
              {validation ? (validation.ok ? "assert OK" : `${validation.issues.length} assert fail`) : "checking…"}
            </span>
          </div>
          {validation && !validation.ok && (
            <ul className="mt-1.5 space-y-0.5 text-[10px] font-mono text-destructive">
              {validation.issues.slice(0, 3).map((i, k) => (
                <li key={k}>· {i.field}: {i.expected} ≠ {i.got}</li>
              ))}
            </ul>
          )}
        </div>

        <div className="pointer-events-none absolute right-4 bottom-4 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          drag · attract &nbsp;·&nbsp; right-drag · repel
        </div>
      </section>

      {/* Controls */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-10 grid gap-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm md:grid-cols-2 lg:grid-cols-3">
        <Field label="Gravity"   value={params.gravity}   min={-200} max={400} step={1}    onChange={(v) => update("gravity", v)} />
        <Field label="Damping"   value={params.damping}   min={0}    max={1}   step={0.01} onChange={(v) => update("damping", v)} />
        <Field label="Attractor" value={params.attractor} min={0}    max={5}   step={0.1}  onChange={(v) => update("attractor", v)} />
        <Field label="Particles · N" value={params.particleCount} min={50} max={8000} step={50} onChange={(v) => update("particleCount", v)} />
        <Field label="Trail"     value={params.trail}     min={0}    max={0.95} step={0.01} onChange={(v) => update("trail", v)} />
        <Field label="Spring k"  value={params.springK}   min={0}    max={400} step={5}    onChange={(v) => update("springK", v)} />
        <Field label="Rest length" value={params.restLength} unit="px" min={5} max={200} step={1} onChange={(v) => update("restLength", v)} />
        <Field label="Edges / node" value={params.edgesPerNode} min={0} max={6} step={1} onChange={(v) => { update("edgesPerNode", v); setResetKey((k) => k + 1); }} />
        <Field label="Pairwise · ε"  value={params.pairwiseStrength} min={-500} max={1000} step={10} onChange={(v) => update("pairwiseStrength", v)} />
        <Field label="Pairwise radius" value={params.pairwiseRadius} unit="px" min={0} max={200} step={1} onChange={(v) => update("pairwiseRadius", v)} />
        <Field label="Constraint iters" value={params.constraintIters} min={0} max={20} step={1} onChange={(v) => update("constraintIters", v)} />
        <Field label="Sub-steps / frame" value={params.subSteps} min={1} max={8} step={1} onChange={(v) => update("subSteps", v)} />
        <Field label="Workers" value={params.workers} min={1} max={8} step={1} onChange={(v) => update("workers", v)} />
        <Field label="Field · strength" value={params.fieldStrength} min={-2} max={2} step={0.05} onChange={(v) => update("fieldStrength", v)} />
        <Field label="∂L/∂x · lr" value={params.objectiveLR} min={0} max={0.5} step={0.005} onChange={(v) => update("objectiveLR", v)} />

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">loss.backward()</div>
          <Button
            variant={params.optimize ? "default" : "outline"}
            className={`w-full uppercase tracking-[0.18em] text-[10px] ${
              params.optimize ? "bg-secondary text-secondary-foreground glow-coral" : ""
            }`}
            onClick={() => update("optimize", !params.optimize)}
            title="Differentiate L=½‖x-target‖² and descend"
          >
            {params.optimize ? `optimizing · L=${(loss ?? 0).toExponential(2)}` : "off"}
          </Button>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Show partitions</div>
          <Button
            variant={params.showPartitions ? "default" : "outline"}
            className={`w-full uppercase tracking-[0.18em] text-[10px] ${
              params.showPartitions ? "bg-accent text-accent-foreground" : ""
            }`}
            onClick={() => update("showPartitions", !params.showPartitions)}
          >
            {params.showPartitions ? "On" : "Off"}
          </Button>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Potential field</div>
          <div className="grid grid-cols-4 gap-1.5">
            {(["none", "swirl", "wells", "ripple"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.field === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.field === opt ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("field", opt)}
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Pairwise force</div>
          <div className="grid grid-cols-3 gap-1.5">
            {(["lj", "repel", "attract"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.pairwiseMode === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.pairwiseMode === opt ? "bg-primary text-primary-foreground glow-mint" : ""
                }`}
                onClick={() => update("pairwiseMode", opt)}
                title={
                  opt === "lj"
                    ? "Lennard-Jones-like: short-range repulsion + medium-range attraction"
                    : opt === "repel"
                    ? "Soft-core repulsion only"
                    : "Linear attractive well"
                }
              >
                {opt === "lj" ? "L-J" : opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Boundary</div>
          <div className="grid grid-cols-3 gap-1.5">
            {(["walls", "wrap", "periodic"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.boundary === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.boundary === opt ? "bg-secondary text-secondary-foreground glow-coral" : ""
                }`}
                onClick={() => update("boundary", opt)}
                title={
                  opt === "walls"
                    ? "Hard reflective walls (restitution 0.7)"
                    : opt === "wrap"
                    ? "Positions wrap; pairwise forces ignore the seam"
                    : "Periodic box: positions wrap AND minimum-image pairwise forces"
                }
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Force viz</div>
          <div className="grid grid-cols-3 gap-1.5">
            {(["off", "vectors", "heatmap"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.forceViz === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.forceViz === opt ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("forceViz", opt)}
                title={
                  opt === "vectors"
                    ? "Yellow arrows showing per-particle force direction & magnitude"
                    : opt === "heatmap"
                    ? "Color particles by |F| (viridis-style)"
                    : "Hide force overlay"
                }
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2 md:col-span-2 lg:col-span-1">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Integrator</div>
          <div className="grid grid-cols-3 gap-2">
            {(["euler", "semi-euler", "verlet"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.integrator === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.16em] text-[9px] px-1 ${
                  params.integrator === opt ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("integrator", opt)}
              >
                {opt === "euler" ? "Euler" : opt === "semi-euler" ? "Semi-impl." : "Verlet"}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Dtype</div>
          <div className="flex gap-2">
            {(["float32", "float64"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.dtype === opt ? "default" : "outline"}
                className={`flex-1 uppercase tracking-[0.18em] text-[10px] ${
                  params.dtype === opt ? "bg-primary text-primary-foreground glow-mint" : ""
                }`}
                onClick={() => update("dtype", opt)}
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Device</div>
          <div className="flex gap-2">
            {(["cpu", "webgpu"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.device === opt ? "default" : "outline"}
                className={`flex-1 uppercase tracking-[0.18em] text-[10px] ${
                  params.device === opt ? "bg-secondary text-secondary-foreground glow-coral" : ""
                }`}
                onClick={() => update("device", opt)}
                title={opt === "webgpu" ? "Logical device tag — falls back to CPU when navigator.gpu is absent" : undefined}
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="flex items-end gap-3 md:col-span-2 lg:col-span-1">
          <Button
            variant="default"
            className="flex-1 bg-primary text-primary-foreground hover:bg-primary/90 glow-mint uppercase tracking-[0.18em] text-xs"
            onClick={() => update("paused", !params.paused)}
          >
            {params.paused ? "Resume" : "Pause"}
          </Button>
          <Button
            variant="outline"
            className="flex-1 uppercase tracking-[0.18em] text-xs border-secondary/50 text-secondary hover:bg-secondary/10"
            onClick={() => setResetKey((k) => k + 1)}
          >
            Reset
          </Button>
        </div>
      </section>

      {/* Footer / code echo */}
      <footer className="relative z-10 mx-4 lg:mx-10 mb-8 rounded-xl border border-border bg-card/60 p-5 backdrop-blur-sm">
        <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-3">
          determinism.cpp — bitwise reproducible mode (fixed reductions · stable coloring · replay traces)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`// Floating-point math is associative in theory but NOT on real hardware:
// (a+b)+c ≠ a+(b+c) once you cross a guard bit. Determinism = controlling
// EVERY source of ordering: reduction trees, atomic interleavings, RNG
// seeds, kernel launch order, NCCL algorithm choice, even cuBLAS heuristics.
// Trade-off: ~12% slower than the fast path. Used for debugging,
// regression tests, scientific replay, and post-hoc rollback.

// ─── 1. global config — flip ONE flag, the whole stack reconfigures ──
struct DetCfg {
    bool     enabled;
    uint64_t seed;            // master seed; all RNGs derive from it
    int      reduction_tree;  // 0 = ring, 1 = recursive-doubling (fixed root)
    int      ckpt_every;      // steps between rollback snapshots
    char     trace_path[256]; // append-only event log
};

void enable_deterministic_mode(DetCfg c) {
    // a. cuBLAS / cuDNN — opt out of heuristic kernel selection
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);   // required for det
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);    // no TF32 / FP16 fast paths

    // b. NCCL — pin algorithm and protocol (no autotune)
    setenv("NCCL_ALGO",  "Tree", 1);                   // fixed reduction tree
    setenv("NCCL_PROTO", "Simple", 1);                 // no LL128 (timing-dependent)
    setenv("NCCL_NTHREADS", "256", 1);                 // fixed worker count

    // c. CUDA — disable Lazy module loading & async malloc reordering
    setenv("CUDA_MODULE_LOADING", "EAGER", 1);
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1", 1);     // single command queue

    // d. all kernels switch to deterministic variants (see below)
    g_det = c;
}

// ─── 2. deterministic reductions — fixed binary tree, no atomics ─────
//
// Atomics on float are nondeterministic because the OS scheduler decides
// who wins the race. Replace atomicAdd with a two-pass tree reduction
// using an EXPLICIT pair-up order keyed on global node ID.
__global__ void det_reduce_forces(int N, const float* contrib, int n_contrib,
                                  const int* sorted_node_ids, float* out)
{
    // Pass 1: bin contributions by sorted node ID (Hopcroft-style stable sort)
    // Pass 2: per-node, sum contributions IN INDEX ORDER — same order every run
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int begin = bin_start[n], end = bin_end[n];
    float acc = 0.f;
    for (int k = begin; k < end; k++) acc += contrib[k];   // canonical order
    out[n] = acc;
}
// CPU host-side reductions use Kahan summation in addition to fixed order
// to suppress the last-bit drift that deterministic ordering alone misses.

// ─── 3. deterministic graph coloring — global-ID tie-breaker ─────────
//
// Jones-Plassmann's randomness comes from per-edge weights w[e]. Replacing
// hash(e, round) with a function of GLOBAL EDGE ID makes the coloring
// identical regardless of which GPU owns the edge, which order edges are
// streamed, or how many ranks participate.
__device__ uint32_t det_weight(uint64_t global_edge_id, uint64_t seed) {
    // SplitMix64 — bijective mixer, no rounding, deterministic everywhere
    uint64_t z = global_edge_id + seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return (uint32_t)(z ^ (z >> 31));
}
// Now color(edge) is a pure function of (global_id, seed). Bit-identical
// across runs, partition layouts, and GPU counts.

// ─── 4. fixed scheduling order — canonical kernel sequence ───────────
//
// The persistent-thread scheduler normally pulls tasks out of a ring queue
// (load-balanced but order-dependent). In det mode we LINEARISE the queue:
// the same (task_type, range) tuples in the same order on every rank.
const TaskOrder DET_ORDER[] = {
    {RESET,      0, N},
    {GRAVITY,    0, N},
    {SPRING,     0, E},                  // edges in global_id order
    {INTEGRATE,  0, N},
    {CONSTRAINT, 0, E, /*iters*/ 8},     // PBD passes also in fixed order
    {STEP_END,   0, 0},
};
// Cost: ~12% throughput loss (load imbalance reappears). Det mode is opt-in.

// ─── 5. replayable trace — append-only event log with hashes ─────────
//
// Every "interesting" event (frame end, checkpoint, fault, repartition)
// writes a record carrying a hash of the entire simulation state. Two runs
// produce byte-identical trace files iff they're bit-equivalent.
struct TraceRecord {
    uint64_t step;
    uint64_t state_hash;     // xxhash3 of (x, v, f, edges)
    uint64_t reduction_hash; // hash of last all-reduce result
    int32_t  topology_epoch; // bumps on rebalance
    char     event[16];      // "FRAME", "CKPT", "FAULT", "REPART"
};
void trace_emit(TraceRecord r) {
    pwrite(g_trace_fd, &r, sizeof r, g_trace_offset.fetch_add(sizeof r));
}

// ─── 6. rollback — restore from any snapshot, replay forward ─────────
//
// Snapshot every CKPT_EVERY steps; trace records every step. To rewind to
// step k, load snapshot floor(k/CKPT_EVERY), replay log entries forward.
// In det mode the replay is BIT-IDENTICAL to the original — it's not a
// re-simulation, it's a deterministic re-execution.
void rollback_to(int target_step, OrchestratorCtx* o) {
    int snap = (target_step / g_det.ckpt_every) * g_det.ckpt_every;
    restore_from_checkpoint(o, snap);
    g_det.seed = o->seed_at(snap);                     // RNG state restored too
    for (int s = snap; s < target_step; s++) {
        run_local_simulation(o->state, o->comm);       // det ⇒ identical replay
        if (state_hash(o->state) != trace_lookup(s).state_hash)
            abort_with("nondeterminism leak at step " + std::to_string(s));
    }
}

// ─── 7. validation pipeline — CI runs to catch det regressions ───────
//
//   run_a = simulate(seed=42, world=4,  steps=10000)
//   run_b = simulate(seed=42, world=8,  steps=10000)   // different topology
//   run_c = simulate(seed=42, world=4,  steps=10000)   // re-run
//   assert(trace_hash(run_a) == trace_hash(run_b) == trace_hash(run_c))
//
// Any failure points to a leaked nondeterminism source. Common culprits:
//   • cuBLAS picked a different GEMM kernel (workspace size changed)
//   • new NCCL version added an LL128 fast path
//   • a developer used atomicAdd in a hot loop
//   • OS scheduler latency caused a different completion ordering

// ─── 8. cost / value summary ──────────────────────────────────────────
//   throughput cost ........ ~12% (fixed schedule + tree reductions)
//   debug value ............ git-bisect on simulation divergence works
//   science value .......... reviewers can rerun the exact 4096-GPU job
//   rollback cost .......... O(CKPT_EVERY) replay, ~few seconds typical
//   trace size ............. ~80 B/step → 28 GB for a 24h run @ 60 Hz`}
        </pre>
      </footer>
    </main>
  );
}
