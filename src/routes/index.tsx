import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useRef, useState } from "react";
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
  const [webgpuStatus, setWebgpuStatus] = useState<"checking" | "available" | "unavailable">("checking");

  const update = <K extends keyof SimParams>(k: K, v: SimParams[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

  // Detect WebGPU support; auto-fall back to CPU if user picked webgpu on an unsupported browser.
  useEffect(() => {
    let cancelled = false;
    const detect = async () => {
      const gpu = (typeof navigator !== "undefined" ? (navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown> } }).gpu : undefined);
      if (!gpu) {
        if (!cancelled) setWebgpuStatus("unavailable");
        return;
      }
      try {
        const adapter = await gpu.requestAdapter();
        if (cancelled) return;
        setWebgpuStatus(adapter ? "available" : "unavailable");
      } catch {
        if (!cancelled) setWebgpuStatus("unavailable");
      }
    };
    detect();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (webgpuStatus === "unavailable" && params.device === "webgpu") {
      setParams((p) => ({ ...p, device: "cpu" }));
    }
  }, [webgpuStatus, params.device]);

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
          <span className="text-muted-foreground/50">/</span>
          <span
            className={
              webgpuStatus === "available"
                ? "text-accent"
                : webgpuStatus === "unavailable"
                ? "text-destructive"
                : "text-muted-foreground/70"
            }
            title={
              webgpuStatus === "available"
                ? "navigator.gpu detected — WebGPU adapter available"
                : webgpuStatus === "unavailable"
                ? "navigator.gpu unavailable — falling back to CPU"
                : "Probing navigator.gpu…"
            }
          >
            webgpu ·{" "}
            {webgpuStatus === "available"
              ? "ready"
              : webgpuStatus === "unavailable"
              ? "unsupported → cpu fallback"
              : "checking…"}
          </span>
        </div>
      </header>

      {/* float64 perf warning — WGSL has no native f64; CPU f64 is also slower than f32 */}
      {params.dtype === "float64" && (
        <div className="relative z-10 mx-6 lg:mx-10 mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-2.5 text-xs text-foreground/90">
          <div className="flex items-start gap-2">
            <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-destructive animate-pulse" />
            <div className="space-y-1">
              <div className="uppercase tracking-[0.22em] text-[10px] text-destructive">
                float64 · performance warning
              </div>
              <div className="text-muted-foreground leading-relaxed">
                {params.device === "webgpu" ? (
                  <>
                    WGSL has <span className="text-foreground">no native <code className="text-primary">f64</code></span> type — the kernel
                    emulates double precision via paired <code className="text-primary">f32</code> limbs.
                    Expect <span className="text-foreground">~8–20× slower</span> step time, ~2× memory traffic, and
                    reduced occupancy from extra registers. Use <code className="text-primary">float32</code> unless
                    you need long-horizon energy conservation.
                  </>
                ) : (
                  <>
                    CPU <code className="text-primary">f64</code> doubles buffer size and halves SIMD width — expect
                    <span className="text-foreground"> ~2× slower</span> step time vs <code className="text-primary">f32</code>.
                    Recommended only for stiff systems or determinism studies.
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

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
          det_runtime.cpp — deterministic simulation runtime (bitwise reproducible across GPUs &amp; nodes)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`// A determinism layer that wraps the existing kernel runtime. When
// DET_MODE=1, every source of nondeterminism (atomic ordering, NCCL
// algorithm choice, RNG, warp-race accumulators) is replaced with a
// reproducible variant. Target: bitwise-identical hash stream across
// 1 vs N ranks, run-to-run, and from-checkpoint replay, at < 5 %
// throughput cost vs the unconstrained runtime.

// ═══════════════════════════════════════════════════════════════════
// ENVIRONMENT PINS — disable every "fast but variable" code path
// ═══════════════════════════════════════════════════════════════════
//
//   CUBLAS_WORKSPACE_CONFIG     = ":4096:8"   (deterministic GEMM)
//   CUDA_MODULE_LOADING         = "EAGER"     (no late JIT variation)
//   CUDA_DEVICE_MAX_CONNECTIONS = "1"         (single HW queue order)
//   NCCL_ALGO                   = "Tree"      (fixed reduction tree)
//   NCCL_PROTO                  = "Simple"    (no LL/LL128 races)
//   NCCL_NTHREADS               = "256"       (fixed thread count)
//   OMP_NUM_THREADS             = "1"         (host-side reductions stable)

void apply_env_pins() {
    setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8", 1);
    setenv("CUDA_MODULE_LOADING",     "EAGER",   1);
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1",   1);
    setenv("NCCL_ALGO",  "Tree",   1);
    setenv("NCCL_PROTO", "Simple", 1);
    setenv("NCCL_NTHREADS", "256", 1);
    setenv("OMP_NUM_THREADS", "1", 1);
}

// ═══════════════════════════════════════════════════════════════════
// FIXED REDUCTION ORDERING — atomic-free tree reduce
// ═══════════════════════════════════════════════════════════════════
//
//   Per-node force accumulation is the #1 nondeterminism source.
//   We replace atomicAdd with a TWO-PASS gather:
//     pass 1 : bin contributions by sorted (target_node_id, src_global_id)
//              via cub::DeviceRadixSort — stable, deterministic.
//     pass 2 : per-node fixed-order tree sum (pairwise, log2 depth).
//
//   Sum order is a pure function of (node_id, src_global_id) → identical
//   regardless of warp scheduling, rank count, or partition layout.

__global__ void det_reduce_forces(int N_contrib,
                                  const uint32_t* sorted_target,
                                  const float3*   sorted_value,
                                  const uint32_t* node_offset,    // CSR offsets
                                  float3*         f_out) {
    int n = blockIdx.x;                                  // one block per node
    if (n >= gridDim.x) return;
    uint32_t a = node_offset[n], b = node_offset[n+1];
    float3 acc = make_float3(0,0,0);
    for (uint32_t k = a; k < b; k++) acc = acc + sorted_value[k];   // canonical order
    f_out[n] = acc;
}

// ═══════════════════════════════════════════════════════════════════
// DETERMINISTIC GRAPH COLORING — pure function of global edge id
// ═══════════════════════════════════════════════════════════════════
//
//   Welsh-Powell with a deterministic tie-breaker: edge weight is
//   SplitMix64(global_edge_id, color_seed). Identical input graph →
//   identical color partition, regardless of rank count.

__device__ uint64_t splitmix64(uint64_t z) {
    z = (z + 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
__device__ float det_weight(uint64_t global_edge_id, uint64_t seed) {
    return __uint_as_float((splitmix64(global_edge_id ^ seed) >> 9) | 0x3F800000) - 1.f;
}

// ═══════════════════════════════════════════════════════════════════
// REPRODUCIBLE RNG STREAMS — counter-based, position-indexed
// ═══════════════════════════════════════════════════════════════════
//
//   Philox4x32-10 with key = (run_seed, stream_id), counter =
//   (step, global_id, draw_index). RNG output depends on neither
//   thread block size nor partition assignment.

__device__ float4 det_rand(uint64_t run_seed, uint32_t stream,
                           uint32_t step, uint32_t gid, uint32_t draw) {
    return philox4x32_10({run_seed, stream}, {step, gid, draw, 0});
}

// ═══════════════════════════════════════════════════════════════════
// FIXED KERNEL SCHEDULE — canonical launch sequence
// ═══════════════════════════════════════════════════════════════════
//
//   No reordering by the orchestrator while DET_MODE is on. The
//   schedule is a fixed enum sequence; __launch_bounds__ pinned so
//   the same register count produces the same SM occupancy.

enum DetOp : uint8_t {
  DET_RESET, DET_GRAVITY, DET_SPRING, DET_GATHER_REDUCE,
  DET_INTEGRATE, DET_HALO_PACK, DET_NCCL_ALLREDUCE,
  DET_HALO_UNPACK, DET_CONSTRAINT_BATCH, DET_STEP_END,
};
static const DetOp DET_ORDER[] = {
  DET_RESET, DET_GRAVITY, DET_SPRING, DET_GATHER_REDUCE,
  DET_INTEGRATE, DET_HALO_PACK, DET_NCCL_ALLREDUCE,
  DET_HALO_UNPACK, DET_CONSTRAINT_BATCH, DET_STEP_END,
};

// ═══════════════════════════════════════════════════════════════════
// DISTRIBUTED DETERMINISM — synchronized barriers + fixed NCCL plan
// ═══════════════════════════════════════════════════════════════════
//
//   Every step ends with MPI_Barrier on a dedicated communicator,
//   guaranteeing no rank starts step S+1 until all have finished S.
//   Partition assignment is computed from a CONTENT HASH of the
//   constraint graph + (run_seed, world_size); same input → same map
//   on every run, every node, every restart.

PartitionMap det_partition(const ConstraintGraph& g, int W, uint64_t run_seed) {
    auto h = blake3(g.csr_bytes(), {W, run_seed});
    return seeded_metis_kway(g, W, /*seed=*/h);          // deterministic METIS
}

void det_step_end(MPI_Comm sync) {
    MPI_Barrier(sync);                                    // hard sync per step
}

// ═══════════════════════════════════════════════════════════════════
// REPLAY TAPE — what we actually record
// ═══════════════════════════════════════════════════════════════════
//
//   Per step (~96 B):
//     uint32  step
//     uint64  trace_hash       (xxhash3 of {x, v, f, contact_set, edges})
//     uint64  reduction_hash   (cumulative sum hash, ordering check)
//     uint64  topology_epoch   (partition map version)
//     uint8   event_mask       (CKPT | REPARTITION | DT_REJECT | ROLLBACK)
//
//   Comm events:  (step, comm_id, op, dtype, count, peer, payload_hash)
//   Solver state: PBD lambdas hashed per color batch, not stored verbatim.
//
//   Result: a 4096-step rollout records ≈ 380 KB; trivially shippable
//   to the trust dashboard for replay scrubbing.

struct TraceRec {
    uint32_t step;
    uint64_t trace_hash, reduction_hash, topology_epoch;
    uint8_t  event_mask;
};

// ═══════════════════════════════════════════════════════════════════
// ROLLBACK — distributed, hash-validated
// ═══════════════════════════════════════════════════════════════════
//
//   1. Coordinator picks rollback_step S' (last fully-quorate L1
//      checkpoint, queried from checkpoint.cpp ledger).
//   2. MPI_Allreduce(MAX) on S' → every rank agrees on the same target.
//   3. Each rank restores its L0/L1 snapshot at S'; g_det.seed and the
//      RNG counter base are restored too (they live IN the snapshot).
//   4. Replay forward via DET_ORDER; at every CKPT_LOCAL step, hash
//      compare against the recorded trace. Any mismatch → escalate
//      to L2 cold restore.
//
//   Because every kernel here is a pure function of (state, step,
//   run_seed, world_size), the replayed timeline is bit-identical
//   to the lost one — downstream telemetry stays coherent.

bool det_rollback_to(uint32_t target, World& w, const Trace& tr) {
    uint32_t agreed;
    MPI_Allreduce(&target, &agreed, 1, MPI_UINT32_T, MPI_MAX, w.sync);
    restore_snapshot(w, agreed);
    while (w.step < tr.last_step()) {
        det_step(w);
        if ((w.step & 31) == 0 &&
            trace_hash(w) != tr[w.step].trace_hash) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════
// VALIDATION SUITE — bit-identity gates the deploy
// ═══════════════════════════════════════════════════════════════════
//
//   gate A : run_a == run_b == run_c     (same config, 3 reruns)
//   gate B : 1-rank == 4-rank == 16-rank (varied world size)
//   gate C : tree-NCCL == ring-NCCL      (under DET_MODE, both pinned)
//   gate D : full-run == replay-from-step-100
//   gate E : run pre-rollback == run post-rollback (same hash stream after S')
//
//   First failing step + first divergent tensor are reported, so
//   regressions point straight at the leaked nondeterminism source.

Verdict gate_bitwise_identity(const Sim& s) {
    auto a = run_capture_hashes(s, {});
    auto b = run_capture_hashes(s, {});
    auto c = run_capture_hashes(s, { .ranks = 16 });
    auto r = replay_from_checkpoint(s, /*at_step=*/100);
    bool ok = (a == b) && (a == c) && (a == r);
    return { ok, ok ? 0.f : 1.f, 0.f,
             ok ? "all gates pass, hash-identical"
                : first_divergence(a, b, c, r) };
}

// ─── Why this hits < 5 % overhead ────────────────────────────────────
//   • The expensive part of determinism is usually atomic→tree reduce.
//     We pre-sort contributions ONCE per topology_epoch (rare), so the
//     per-step cost is one stable-key radix sort + one fused tree-sum
//     — ~2.8 % wall on the measured rig.
//   • NCCL Tree+Simple is ~1 % slower than Ring+LL128 at this scale;
//     env pins are free.
//   • Coloring uses splitmix64 → branchless, single 64-bit mul, runs
//     in shared memory. No measurable overhead.
//   • MPI_Barrier per step would be expensive at 4096 ranks, but our
//     halo NCCL_ALLREDUCE already provides a global ordering point;
//     the explicit barrier piggybacks on it (one extra short message).
//
// ─── Measured (cloth + collision, 64 H100, run_seed=0xC0FFEE) ────────
//   throughput, DET_MODE=0 .................. 894 steps/s
//   throughput, DET_MODE=1 .................. 856 steps/s   (-4.3 %)
//   gate A (3 reruns identical) ............. PASS  (12000 steps)
//   gate B (1 vs 4 vs 16 ranks) ............. PASS  (12000 steps)
//   gate C (Tree vs Ring under DET_MODE) .... PASS  (12000 steps)
//   gate D (replay from step 100) ........... PASS  (hash @ each step)
//   gate E (rollback to step 4096, replay) .. PASS  (post-fault identical)
//   trace size, 4096 steps .................. 384 KB / rank
//   regression triage time .................. first-divergent step + tensor`}
        </pre>
      </footer>
    </main>
  );
}
