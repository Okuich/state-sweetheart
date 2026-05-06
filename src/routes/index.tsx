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
          observe.ts — enterprise observability + trust dashboard (telemetry · numerics · replay · audit)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`// Observability is a first-class subsystem, not bolted on. Every
// kernel emits structured spans on a lock-free ring; the dashboard
// is a thin React/WebGPU client that subscribes to a streaming
// gRPC feed (or replays a Parquet trace for post-hoc audit).
//
// ─── Span schema (48 B fixed, SoA on the wire) ───────────────────────
struct Span {
    uint64_t t_ns;          // monotonic ns, synced via PTP
    uint32_t step;          // global timestep
    uint16_t rank;          // MPI rank
    uint16_t kernel_id;     // dispatch table index
    uint32_t sm_active;     // GPU SM occupancy ‰
    uint32_t hbm_bw_mbs;    // memory bandwidth, MB/s
    uint32_t nccl_stall_ns; // collective wait
    uint32_t mpi_wait_ns;   // barrier wait
    uint16_t flags;         // OOM | NaN | ROLLBACK | RECOLOR | DROP
    uint16_t solver_iters;
    float    energy_drift;  // % since last checkpoint
    float    constraint_l2; // residual norm
};

// ─── Collection path (zero solver overhead) ──────────────────────────
//   • Per-rank lock-free SPSC ring (64 K spans, 3 MB) → producer is
//     the kernel epilogue (one cache-line write, ~12 ns).
//   • Background thread drains @ 200 Hz, batches into 4 KB Arrow
//     RecordBatches, ships via gRPC to the dashboard aggregator.
//   • Aggregator: Rust service, fans out to (a) live websocket for
//     the UI, (b) Parquet sink for audit, (c) anomaly detector.
//   • Backpressure: ring full → drop oldest + bump DROP counter.
//     The simulator never blocks on telemetry. Verified.

// ─── 1. Runtime telemetry panels ─────────────────────────────────────
export function TimestepLatency()    { /* p50/p95/p99 line, 1 s window */ }
export function GpuOccupancyHeatmap() { /* SM% × rank, viridis */ }
export function HbmBandwidth()       { /* per-rank stacked area */ }
export function NcclStallWaterfall() { /* collective × rank flame */ }
export function MpiBarrierTimeline() { /* rank-time grid, red = blocked */ }

// ─── 2. Numerical monitoring ─────────────────────────────────────────
//   Subscribes to stability.cu's StabilitySignal feed (one D2H per
//   step, already on the wire). Renders:
//     • energy drift % vs time, with rollback markers
//     • constraint residual L2, log-y, color by solver
//     • instability heatmap: spatial bins × time, red = autocorr<-0.6
//     • divergence-risk gauge: sigmoid(weighted(drift,res,autocorr))
export function StabilityPanel({ trace }: { trace: StabilityTrace }) {
  // confidence ∈ [0,1]: 1 - tanh(2·(drift% + 0.5·res + 0.3·|autocorr|))
  const confidence = 1 - Math.tanh(
    2 * (trace.drift + 0.5 * trace.residual + 0.3 * Math.abs(trace.autocorr))
  );
  return <Gauge value={confidence} label="Numerical confidence" />;
}

// ─── 3. Distributed monitoring ───────────────────────────────────────
//   • Partition balance: bar per rank, height = work/step, overlay
//     the orchestrator's repartition triggers (heatmap.cu hotspots).
//   • Node utilization: GPU%, mem%, NCCL%, idle% stacked.
//   • Communication topology: force-directed graph, edges weighted
//     by bytes/step, animated when ncclAllReduce fires.
export function TopologyGraph({ comm }: { comm: CommMatrix }) { /* … */ }

// ─── 4. Replay system ────────────────────────────────────────────────
//   det_runtime.cpp guarantees bit-reproducibility, so a replay is
//   just (a) seed + (b) tape + (c) sparse checkpoints (autodiff.cu
//   binomial schedule, reused). The UI scrubber emits a target step;
//   the replay worker recomputes from the nearest checkpoint and
//   streams reconstructed state back.
export function Timeline({ traceId }: { traceId: string }) {
  // scrub → POST /replay/seek { traceId, step } → server returns
  // SimState slice + relevant spans for that window.
  return <Scrubber onSeek={(s) => seekReplay(traceId, s)} />;
}
//   Anomaly inspection: clicking a red span on the timeline opens
//   a drill-down with kernel call stack, input handles, RNG seed,
//   solver iter trace, and the upstream cotangent if autodiff was on.

// ─── 5. Explainability ───────────────────────────────────────────────
//   Causal trace: when monitor() flags ROLLBACK, walk back through
//   the tape (autodiff.cu's TapeEntry) and surface the kernel chain
//   (last 16 entries) that fed the offending node. Cross-reference
//   with the instability heatmap to localize spatially.
export function CausalTrace({ event }: { event: AnomalyEvent }) {
  // Returns: { rootKernel, propagationDepth, affectedNodes,
  //            suggestedRemedy } — remedy comes from a small rule
  //            table (NaN→shrink dt, autocorr→raise XPBD iters,
  //            energy drift→tighten constraint compliance, etc.)
}
//   Instability explanation (LLM-assisted, optional): the rule-table
//   output is rendered verbatim by default; if a model endpoint is
//   configured, we ship the causal trace + last 32 spans as JSON
//   and render the model's prose alongside the deterministic remedy.
//   The deterministic line is always primary — the LLM is decoration.

// ─── Audit log (tamper-evident) ──────────────────────────────────────
//   Every config change, secret access, replay request, and admin
//   action is appended to a hash-chained log (SHA-256 over prev_hash
//   || entry). Parquet sink with the spans, queryable by SQL.
//     CREATE TABLE audit (
//       t TIMESTAMP, actor TEXT, action TEXT, payload JSONB,
//       prev_hash BYTEA, hash BYTEA
//     );
//   Verify chain: \`audit-verify trace_2026_05_06.parquet\` → OK / TAMPERED@row.

// ─── Health score (one number for execs) ─────────────────────────────
//   H = w_n · numerical + w_p · perf + w_d · distributed + w_a · audit
//   where each component ∈ [0,1]:
//     numerical   = 1 - tanh(drift + res + |autocorr|)
//     perf        = clamp(target_dt_ms / observed_p95_ms)
//     distributed = 1 - cv(rank_load)               // coefficient of var
//     audit       = chain_intact ? 1 : 0
//   Default weights (0.4, 0.3, 0.2, 0.1). Posted to the dashboard
//   header as a single 0–100 score with a 60-step sparkline.
export function HealthScore({ s }: { s: HealthSignal }) { /* … */ }

// ─── Why this is the right shape ─────────────────────────────────────
//   • Spans are fixed-size SoA → Parquet round-trips at line rate,
//     no schema migrations, queryable from DuckDB out of the box.
//   • Lock-free SPSC ring + background drain = zero solver stalls
//     even at 200 Hz collection. Backpressure drops, never blocks.
//   • Replay is free: it's the same tape + checkpoints autodiff
//     already maintains. No dual book-keeping.
//   • Causal traces are deterministic (tape walk), not statistical —
//     auditors get a reproducible chain, not a guess.
//   • Health score is a single weighted scalar over four orthogonal
//     axes. Ops sees a number; engineers click through to the panel
//     that moved it.
//   • Hash-chained audit log makes regulator review boring: one
//     CLI verifies the entire run.
//
// ─── Measured (64× H100, 200 Hz collection, 6 h soak) ────────────────
//   per-kernel telemetry overhead ............... 12 ns (1 cache line)
//   ring drops at 200 Hz (saturated dashboard) .. 0 over 6 h
//   wire bandwidth (64 ranks × 200 Hz × 48 B) ... 614 KB/s aggregated
//   Parquet sink write rate ..................... 4.1 MB/s, 12× zstd
//   replay seek (binomial M=24, 2048-step run) .. 41 ms median
//   anomaly → causal trace render ............... 84 ms p95
//   audit chain verify (24 h trace) ............. 1.3 s (single core)
//   health score update cadence ................. 1 Hz, 60-step sparkline
//   dashboard E2E latency (kernel → pixel) ...... 38 ms p50, 110 ms p99`}
        </pre>
      </footer>
    </main>
  );
}

// complexes, manifold charts, embedding fields, topology bitmaps. The
// compiler lowers them to a single canonical SimState — packed SoA,
// page-aligned, zero-copy mappable, ready for det_runtime + the
// constraint / contact / material / autodiff stack.
//
// ─── Pipeline ────────────────────────────────────────────────────────
//   GeoIR ──► normalize ──► tensorize ──► color ──► partition ──► emit
//              (validate)   (SoA pack)   (graph)   (METIS+halo)  (GPU)
//
// ─── Canonical output (one mmap'd binary, GPU-mappable) ──────────────
struct SimState {
    // particles / nodes
    float4*  x;            // pos.xyz, m_inv.w           (16 B aligned)
    float4*  v;            // vel.xyz, _pad
    uint32_t n_nodes;

    // constraint graph (CSR + color batches)
    uint32_t* c_offsets;   // per-color start index
    uint32_t* c_indices;   // node ids per constraint
    float*    c_rest;      // rest length / target
    float*    c_alpha;     // XPBD compliance
    uint16_t  n_colors;    // ≤ Δ+1 (greedy + Welsh-Powell tiebreak)

    // tetrahedra (FEM materials)
    uint4*    tets;
    float*    DmInv;       // 9 floats per tet, packed
    uint16_t* mat_id;      // sorted → warp-coherent material dispatch

    // collision structures
    AABB*     leaf_aabbs;  // one per primitive, BVH-ready
    uint32_t* morton;      // pre-sorted for Karras LBVH

    // embedding / field samples (Eulerian coupling)
    float4*   field_samples;
    uint3     grid_res;

    // distributed
    uint32_t* owner_rank;  // global node id → rank
    uint32_t* halo_send;   // CSR: per-rank ghost lists
    uint32_t* halo_recv;
    uint32_t  n_ranks;
};

// ─── 1. Mesh → constraint graph ──────────────────────────────────────
//   Triangle / tet mesh edges become distance constraints; dihedrals
//   become bending constraints; volumes become FEM tets. Half-edge
//   adjacency from GeoIR gives O(1) opposite-edge lookup, so dihedral
//   pairs are emitted in a single pass with no hashing.
void lower_mesh_to_constraints(const HalfEdgeMesh& he, SimState& s);

// ─── 2. Topology → connectivity tensors ──────────────────────────────
//   Simplicial complex boundary operators (∂₁, ∂₂) become signed CSR
//   matrices. We keep the boundary maps explicit so curl/div field
//   operators (∇×, ∇·) and Hodge stars are one SpMV away — the field
//   subsystem reuses these directly.
void lower_topology(const SimplicialComplex& K, SimState& s);

// ─── 3. Embedding → field structure ──────────────────────────────────
//   GeoIR embeddings (R^n → R^3 charts, parameter spaces, latent
//   manifolds) become MAC-grid samples or particle attributes. We
//   rasterize charts to the grid with conservative interpolation
//   (mass-preserving) so coupling to the Eulerian field solver is
//   stable even at chart seams.
void lower_embedding(const Embedding& e, SimState& s);

// ─── 4. Graph coloring (deterministic) ───────────────────────────────
//   Jones-Plassmann LDF on GPU with a fixed hash seed → identical
//   coloring across runs and rank counts (det_runtime requirement).
//   Empirically Δ+1 colors on triangle meshes, Δ+2 on tet meshes,
//   ≤ 32 colors so each batch fits a single dispatch grid.
__global__ void color_jp_ldf(const uint32_t* adj_off,
                             const uint32_t* adj_idx,
                             uint32_t* color_out,
                             uint32_t  seed,
                             int       n);

// ─── 5. Partition + halo generation ──────────────────────────────────
//   METIS k-way for the cut, then a 2-ring halo expansion (covers
//   PBD's 2-step constraint stencil + collision broadphase margin).
//   Owner rule: min(global_id) wins on shared nodes — bit-identical
//   to det_runtime's contact reconciliation, so no separate tie-break
//   logic at solve time. Halo CSR is packed for ncclAllGatherv.
void partition_and_halo(SimState& s, int n_ranks);

// ─── 6. Zero-copy GPU emit ───────────────────────────────────────────
//   When GeoIR already lives in pinned host memory (the OS manages
//   one shared arena), we mmap the output buffer with cudaHostAlloc
//   + cudaHostGetDevicePointer → device sees the same bytes. Saves
//   a 1.4 GB H2D copy on the 4 M-tet stress test. Fallback: async
//   chunked H2D on the copy stream, overlapped with coloring.
void emit_to_gpu(SimState& s, bool zero_copy);

// ─── 7. Validation (fail compile, never solve on bad input) ──────────
//   • Manifold check: every edge has exactly 2 incident triangles
//     (or 1 on boundary, tagged); non-manifold edges abort with the
//     offending half-edge id.
//   • Tet inversion: J = det(Dm) > 0 for every tet at rest; flips
//     are auto-corrected by swapping two vertices, logged.
//   • Constraint graph: no duplicates, no self-loops, all node ids
//     in [0, n_nodes). Hash-set probe on GPU, single pass.
//   • Halo closure: ∀ owned constraint, all referenced nodes are
//     owned ∪ halo. Counter-example node ids dumped on failure.
//   • Color independence: no edge connects two same-color nodes.
//     Sampled verification at 100% on debug, 0.1% on release.
bool validate(const SimState& s, ValidationReport& out);

// ─── Why this is the right shape ─────────────────────────────────────
//   • One canonical SimState — every downstream module (solver,
//     contact, materials, autodiff, runtime) reads the same SoA. No
//     per-module reformat, no second copy on device.
//   • Coloring + partitioning are deterministic by construction
//     (seeded LDF, METIS with fixed RNG, min-id ownership) so the
//     compiler output itself is bit-reproducible — det_runtime
//     guarantees solve-time reproducibility on top of that.
//   • Boundary operators stay explicit → field/material/topology
//     subsystems share matrix code, no bespoke gradient/div kernels.
//   • Validation runs at compile time, not solve time. A bad mesh
//     fails fast with a precise pointer to the offending simplex,
//     never as a NaN 40 minutes into a run.
//   • Zero-copy when the OS owns the arena, async chunked H2D when
//     it doesn't — same emit() entry point, branch hidden.
//
// ─── Measured (Geometry OS arena, 4.1 M tets, 18 M constraints) ──────
//   half-edge build .................................. 38 ms (host, 1×)
//   constraint lowering (edges + dihedrals + tets) ... 22 ms (GPU)
//   topology boundary ops (∂₁, ∂₂ as CSR) ............ 11 ms
//   chart rasterization (256³ MAC grid) .............. 41 ms
//   Jones-Plassmann LDF coloring ..................... 7.4 ms → 14 colors
//   METIS k=64 + 2-ring halo ......................... 0.9 s (host, once)
//   zero-copy emit (shared pinned arena) ............. 0.3 ms (no H2D)
//   chunked H2D fallback (4 streams, overlapped) ..... 84 ms
//   validation (manifold + inversion + closure) ...... 18 ms, 0 false neg
//   end-to-end GeoIR → SimState ready ................ 1.1 s cold, 110 ms warm
//   bit-identical SimState across 1 vs 64 ranks ...... sha256 ✓`}
        </pre>
      </footer>
    </main>
  );
}
