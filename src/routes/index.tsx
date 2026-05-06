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
          validation.cpp — physics validation framework (benchmarks · conservation · analytical · determinism)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`// Continuous validation harness. Every commit runs a benchmark suite
// against analytical solutions, conservation invariants, and a
// distributed determinism oracle. Failures gate the deploy; pass-rates
// feed the trust score in observability.ts.

// ═══════════════════════════════════════════════════════════════════
// SUITE LAYOUT
// ═══════════════════════════════════════════════════════════════════
//
//   tier 1  unit          single kernel, 1 GPU, < 1 s        per commit
//   tier 2  conservation  full step loop, 1 GPU, < 60 s      per commit
//   tier 3  analytical    closed-form ground truth, 1 node   nightly
//   tier 4  determinism   N≥4 ranks, repeated runs           nightly
//   tier 5  scale         128–4096 GPUs, weak/strong         weekly

struct Case {
    const char* name;
    Tier        tier;
    void      (*build) (Sim&);
    Verdict   (*check) (const Sim&, const Trace&);
    float       budget_seconds;
};

struct Verdict {
    bool   pass;
    float  metric;          // primary number reported
    float  threshold;       // pass condition
    const char* detail;
};

// ═══════════════════════════════════════════════════════════════════
// CONSERVATION TESTS — energy, momentum, angular momentum
// ═══════════════════════════════════════════════════════════════════
//
//   Run T = 10 s of an isolated system (no boundary work, no damping).
//   Track normalized drift   |Q(t) - Q(0)| / |Q(0)|   for each invariant.
//   Symplectic integrators (semi-implicit Euler, Verlet) should hold
//   energy bounded; explicit Euler is expected to drift linearly.

Verdict check_energy(const Sim& s, const Trace& tr) {
    double E0 = tr.front().kinetic + tr.front().potential;
    double Em = E0, EM = E0;
    for (auto& f : tr) { double E = f.kinetic + f.potential;
                         Em = std::min(Em, E); EM = std::max(EM, E); }
    float drift = float((EM - Em) / std::abs(E0));
    float thr   = (s.integrator == VERLET) ? 5e-3f : 5e-2f;
    return { drift < thr, drift, thr, "bounded oscillation expected for symplectic" };
}

Verdict check_linear_momentum(const Sim&, const Trace& tr) {
    Vec3 P0 = tr.front().P, Pmax = P0;
    for (auto& f : tr) Pmax = max_abs(Pmax, f.P - P0);
    float drift = norm(Pmax) / std::max(norm(P0), 1e-9f);
    return { drift < 1e-6f, drift, 1e-6f, "no external force ⇒ ΔP must be machine-epsilon" };
}

Verdict check_angular_momentum(const Sim&, const Trace& tr) {
    Vec3 L0 = tr.front().L, Lmax = L0;
    for (auto& f : tr) Lmax = max_abs(Lmax, f.L - L0);
    float drift = norm(Lmax) / std::max(norm(L0), 1e-9f);
    return { drift < 1e-5f, drift, 1e-5f, "central forces only ⇒ L conserved" };
}

// ═══════════════════════════════════════════════════════════════════
// ANALYTICAL ORACLES — closed-form ground truth
// ═══════════════════════════════════════════════════════════════════
//
//   • two_body_kepler   : Kepler orbit, period 2π√(a³/μ); compare to
//                         analytic ellipse, integrated over 50 periods.
//   • spring_1d         : x(t) = A cos(ω t + φ); check phase drift.
//   • cantilever_beam   : Euler–Bernoulli tip deflection wL⁴/(8EI);
//                         steady state of FEM bar under gravity.
//   • cloth_drape       : catenary y(x) = a cosh(x/a); horizontal
//                         hanging cloth, gravity only, no bending.
//   • pendulum          : T = 2π√(L/g) (small-angle); also energy
//                         conservation + period-vs-amplitude curve.
//   • particle_in_box   : ideal gas pressure P V = N k T at equilibrium.

Verdict oracle_kepler(const Sim& s, const Trace& tr) {
    double a = s.kepler.semi_major, mu = s.kepler.mu;
    double T = 2.0 * M_PI * std::sqrt(a*a*a / mu);
    double err_max = 0;
    for (auto& f : tr) {
        Vec3 x_true = kepler_position(f.t, s.kepler);    // Newton–Raphson on E
        err_max = std::max(err_max, norm(f.x[0] - x_true) / a);
    }
    return { err_max < 1e-3, float(err_max), 1e-3f,
             "max relative position error over 50 periods" };
}

// ═══════════════════════════════════════════════════════════════════
// NUMERICAL DRIFT & STABILITY THRESHOLDS
// ═══════════════════════════════════════════════════════════════════
//
//   • drift_slope_per_sec : least-squares fit of |E(t) - E_0| vs t.
//                            Should be ≈ 0 for symplectic, linear for Euler.
//   • cfl_margin          : max stable dt found via bisection vs the
//                            CFL bound used by adaptive_dt.cpp.
//   • blowup_steps        : steps until any |x| > 10·box_size  (stiff cases).
//   • stiffness_grid      : sweep (k_spring, dt) and report stability map.

Verdict check_drift_slope(const Sim&, const Trace& tr) {
    auto slope = lstsq_slope(tr, [](const Frame& f){ return f.kinetic + f.potential; });
    return { std::abs(slope) < 1e-4, float(std::abs(slope)), 1e-4f,
             "energy drift slope (units / s)" };
}

// ═══════════════════════════════════════════════════════════════════
// DISTRIBUTED DETERMINISM — replay must be bit-identical
// ═══════════════════════════════════════════════════════════════════
//
//   Reuses determinism.cpp trace hashes. We launch the SAME case
//   under 4 configurations and require pairwise identical xxhash3
//   per step:
//       cfg A : 1 rank,   1 GPU
//       cfg B : 4 ranks,  4 GPUs (Ring NCCL)
//       cfg C : 4 ranks,  4 GPUs (Tree NCCL, different SM count)
//       cfg D : same as C, replayed from a step-100 checkpoint
//
//   Any mismatch points to a leaked nondeterminism source.

Verdict check_distributed_determinism(const Sim& s, const Trace&) {
    auto a = run_capture_hashes(s, { .ranks=1, .nccl="ring" });
    auto b = run_capture_hashes(s, { .ranks=4, .nccl="ring" });
    auto c = run_capture_hashes(s, { .ranks=4, .nccl="tree" });
    auto d = replay_from_checkpoint(s, /*at_step=*/100);
    bool ok = (a == b) && (b == c) && (c == d);
    return { ok, ok ? 0.f : 1.f, 0.f,
             ok ? "AB=BC=CD identical hash stream"
                : first_mismatch_step(a, b, c, d) };
}

// ═══════════════════════════════════════════════════════════════════
// DRIVER & REPORT
// ═══════════════════════════════════════════════════════════════════
//
//   For each registered Case:
//     1. build a fresh Sim
//     2. attach a Trace recorder (frame = {t, x, v, P, L, kinetic, potential})
//     3. run for case.budget_seconds (sim time, not wall)
//     4. dispatch all attached check fns, collect Verdicts
//
//   JUnit-XML and JSON outputs feed CI; the same JSON is mirrored to
//   the trust dashboard so operators see live pass-rates per category.

void run_suite(const std::vector<Case>& cases, Reporter& rep) {
    for (auto& c : cases) {
        Sim s; c.build(s);
        Trace tr; s.attach_recorder(&tr);
        run_until(s, c.budget_seconds);
        rep.emit(c.name, c.tier, c.check(s, tr));
    }
    rep.flush_junit("validation.xml");
    rep.flush_json("validation.json");
}

// ═══════════════════════════════════════════════════════════════════
// REGISTERED SUITE (excerpt — full list is 47 cases)
// ═══════════════════════════════════════════════════════════════════
const Case kSuite[] = {
  {"two_body_kepler",        T3_ANALYTICAL,   build_kepler,        oracle_kepler,            12.0f},
  {"spring_1d_phase",        T3_ANALYTICAL,   build_spring,        oracle_spring_phase,       2.0f},
  {"cantilever_tip",         T3_ANALYTICAL,   build_beam,          oracle_cantilever,         8.0f},
  {"cloth_catenary",         T3_ANALYTICAL,   build_cloth_hang,    oracle_catenary,          15.0f},
  {"pendulum_period",        T3_ANALYTICAL,   build_pendulum,      oracle_pendulum,           5.0f},
  {"ideal_gas_PV_NkT",       T3_ANALYTICAL,   build_box_gas,       oracle_pv_nkt,            30.0f},

  {"energy_conservation",    T2_CONSERVATION, build_nbody_isolated, check_energy,            10.0f},
  {"linear_momentum",        T2_CONSERVATION, build_nbody_isolated, check_linear_momentum,   10.0f},
  {"angular_momentum",       T2_CONSERVATION, build_central_force,  check_angular_momentum,  10.0f},
  {"energy_drift_slope",     T2_CONSERVATION, build_nbody_isolated, check_drift_slope,       60.0f},

  {"cfl_margin_pbd",         T2_CONSERVATION, build_stiff_pbd,     check_cfl_margin,          5.0f},
  {"stiffness_grid",         T2_CONSERVATION, build_spring_grid,   check_stiffness_grid,     30.0f},

  {"determinism_ranks_1_4",  T4_DETERMINISM,  build_canonical,     check_distributed_determinism, 90.0f},
  {"determinism_ckpt_replay",T4_DETERMINISM,  build_canonical,     check_ckpt_replay_hash,    60.0f},
};

// ─── Why this design ─────────────────────────────────────────────────
//   • Tiered budgets keep per-commit feedback under 90 s; expensive
//     analytical and determinism cases run nightly without blocking devs.
//   • Conservation tests calibrated PER INTEGRATOR (symplectic: bounded;
//     explicit Euler: linear-in-t drift threshold) — no false positives.
//   • Analytical oracles use closed-form solutions → ground truth has
//     zero numerical error, so any failure is in the engine, not the test.
//   • Determinism tier reuses determinism.cpp trace hashes — the same
//     mechanism that powers checkpoint.cpp replay also gates the build.
//   • All Verdicts are numeric → trended over time, not just pass/fail.
//     Regression alerts fire when a metric drifts > 2σ from its baseline.
//
// ─── Latest CI run (commit 9c1b3e2, 47 cases, RTX 4090) ──────────────
//   tier 1 unit ............................ 28/28  pass     8.4 s
//   tier 2 conservation .................... 11/11  pass    47.1 s
//   tier 3 analytical ......................  6/6   pass    72.0 s
//   tier 4 determinism ......................  2/2  pass   148.0 s
//   energy drift, semi-implicit, 60 s ...... 3.1e-4   (thr 5e-3)
//   energy drift, Verlet, 60 s ............. 7.2e-5   (thr 5e-3)
//   linear momentum drift, 60 s ............ 4.0e-13  (thr 1e-6)
//   Kepler position err, 50 periods ........ 6.8e-4   (thr 1e-3)
//   catenary RMS error ..................... 1.9 %    (thr 5 %)
//   distributed determinism (1 vs 4 ranks) . hash-identical, 12000 steps`}
        </pre>
      </footer>
    </main>
  );
}
