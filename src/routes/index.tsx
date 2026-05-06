import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { PhysicsCanvas, type SimParams, type ValidationReport } from "@/components/PhysicsCanvas";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { compileFieldExpr } from "@/lib/exprCompile";

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
    potentialGrad: "analytic",
    fieldSampling: "auto",
    customFieldSrc: "0.5*(nx^2 + ny^2) + 0.2*sin(8*theta + t)",
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
          <div className="grid grid-cols-5 gap-1.5">
            {(["none", "swirl", "wells", "ripple", "custom"] as const).map((opt) => (
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
          <CustomFieldEditor
            value={params.customFieldSrc}
            onChange={(v) => update("customFieldSrc", v)}
            active={params.field === "custom"}
          />
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Potential gradient</div>
          <div className="grid grid-cols-2 gap-1.5">
            {(["analytic", "finite-diff"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.potentialGrad === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.potentialGrad === opt ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("potentialGrad", opt)}
                title={
                  opt === "analytic"
                    ? "Closed-form ∇Φ — ~2× faster, no h-tuning, bit-stable"
                    : "Central finite differences — plug-and-play fallback (h = 0.5 px)"
                }
              >
                ∇Φ {opt === "analytic" ? "analytic" : "fin-diff"}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Field sampling at edges</div>
          <div className="grid grid-cols-4 gap-1.5">
            {(["auto", "clamp", "wrap", "none"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.fieldSampling === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.fieldSampling === opt ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("fieldSampling", opt)}
                title={
                  opt === "auto"
                    ? "Match boundary mode: walls→clamp, wrap/periodic→wrap"
                    : opt === "clamp"
                      ? "Clip sample coords to canvas — no runaway forces just past walls"
                    : opt === "wrap"
                      ? "Modulo into canvas (Φ as a torus) — keeps ∇Φ continuous across the seam"
                      : "Pass coords through untouched (legacy)"
                }
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
          knowledge/ — semantic physics layer (PDEs · constraints · units · causal graph)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`# A reasoning layer that sits ABOVE the kernels. The simulator computes;
# this layer KNOWS what it is computing, in what units, under what laws,
# and whether the configuration is even well-posed.

# ─── Governing equation registry (declarative, typed) ────────────────
@law("navier_stokes.incompressible")
class IncompressibleNS(PDE):
    vars   = {"u": Vector(dim=3, units="m/s"), "p": Scalar(units="Pa")}
    params = {"rho": Scalar("kg/m^3", positive=True),
              "mu":  Scalar("Pa*s",   positive=True)}
    eqs    = [
        rho*(dt(u) + (u@grad)(u)) + grad(p) - mu*lap(u) - f,   # momentum
        div(u),                                                 # continuity
    ]
    invariants = [conserves("mass"), conserves("momentum",
                  domain="closed_or_periodic")]
    needs_bc   = ["velocity_or_traction on the boundary"]
    well_posed_when = lambda c: c.Re < 1e6 or c.has_turbulence_model

# Built-in libraries shipped: NS (incompressible/compressible), heat,
# wave, Maxwell, elastodynamics (linear + Saint-Venant-Kirchhoff +
# neo-Hookean), Cahn-Hilliard, Allen-Cahn, MHD, Smoluchowski, SPH,
# rigid-body Newton-Euler, Cosserat rods, shallow water, Boussinesq.
# Constitutive laws: Hookean, Mooney-Rivlin, Drucker-Prager, J2 plasticity,
# Bingham, Carreau-Yasuda, Maxwell/Kelvin-Voigt viscoelasticity. Each has
# a parameter schema with units, positivity/range constraints, and
# citations (DOI) - the registry IS the documentation.

# ─── Constraint intelligence (well-posedness checker) ────────────────
report = px.knowledge.check(scene, law=IncompressibleNS, params={...})
# Static checks (run BEFORE any kernel launches):
#   x  missing BC on inlet              (needs_bc not satisfied)
#   x  nu = mu/rho -> Re ~ 4.2e7, no SGS (well_posed_when violated)
#   !  dx * |u|_max / nu  -> Pe = 380   (advection-dominated, upwind?)
#   !  dt * |u|_max / dx  = 1.7         (CFL > 1 for explicit scheme)
#   x  corner singularity at (0,1,0)    (re-entrant, p loses 1/2 order)
# Dynamic checks (subscribed to observe.ts span ring):
#   - det(F) <= 0 anywhere              (element inversion -> halt)
#   - lambda_min(stiffness) -> 0        (loss of ellipticity)
#   - energy_drift > tau, no damping    (numerical instability vs physics)
# Each finding carries: severity, citation, and a concrete remedy
# ("add a wall function", "switch to BDF2", "refine corner with r=0.7").

# ─── Dimensional analysis engine (compile-time + runtime) ────────────
#   Every variable, parameter, BC, and source carries a Unit<L,M,T,K,N,I,J>.
#   Operators propagate units; mismatches are a TypeError, not a runtime
#   surprise:
u   : Vector["m/s"]      = ...
mu  : Scalar["Pa*s"]     = ...
rho : Scalar["kg/m^3"]   = ...
# rho * dt(u)        ->  kg/(m^2*s^2)   matches grad(p)  [Pa/m]  ok
# rho + mu           ->  TypeError: kg/m^3 + Pa*s
#
#   Auto nondimensionalization (Buckingham Pi):
sys = px.knowledge.nondim(IncompressibleNS,
       chars={"L": 1.0*m, "U": 0.1*m/s, "rho": 1000*kg/m**3, "mu": 1e-3*Pa*s})
# -> Re = rho*U*L/mu = 1.0e5     (the only free pi-group)
# -> solver runs on dimensionless eqs; results auto-rescaled on read.
# Scale-consistency check: warns when dx << Kolmogorov eta or
# dt >> acoustic CFL even when units are individually correct.

# ─── Semantic physics graph (entities, fields, forces, causality) ────
#   Nodes:
#     Entity(rigid|deformable|fluid|field|interface|observer)
#     Field (scalar/vector/tensor + domain + units)
#     Force/Flux  with provenance (which law, which term)
#   Edges (typed):
#     ACTS_ON      Force -> Entity        (gravity ACTS_ON bunny)
#     COUPLES      Field <-> Field        (T <-> rho via Boussinesq)
#     CONSTRAINS   BC/Joint -> Entity
#     EMITS / ABSORBS                     (sources/sinks)
#     DEPENDS_ON   any -> any  (causal, used for explainability)
#
g = px.knowledge.graph(scene)
g.path("ankle_torque", "head_acceleration")
# -> ankle_torque -ACTS_ON-> tibia -COUPLES(rigid_link)-> femur
#                 -COUPLES-> pelvis -COUPLES-> spine -ACTS_ON-> head
# Used by:
#   - observe.ts explainability ("why did energy spike at step 1820?")
#     walks DEPENDS_ON edges back to the originating force/field.
#   - verify.py certification reports (auto-generated FBD per entity).
#   - orchestrator.cpp adaptive partitioning (cut along weakly-coupled
#     edges -> minimizes halo traffic without breaking physics).
#   - SDK suggestions: "add damping to spine <-> pelvis, zeta ~ 0.05".

# ─── Reasoning queries (the layer's public surface) ──────────────────
px.knowledge.why_unstable(sim, step=1820)
#  -> "shear-locking in element 41,209 (det F = -2e-3); root cause:
#      under-integrated Q8 with nu = 0.499; remedy: F-bar or B-bar;
#      cite Hughes (2000) sec 4.5.2"
px.knowledge.missing_bcs(scene)
#  -> ["outlet has no traction or pressure BC; outflow undetermined"]
px.knowledge.suggest_law(observations)
#  -> ranks candidate constitutive laws by KL-divergence on stress-strain
#     response; returns top-3 with parameter MLE + 95% CIs.
px.knowledge.invariants(sim, window=(0,1000))
#  -> conserved quantities measured: mass (drift 4e-6), linear momentum
#     (drift 7e-10), energy (drift 0.04%), enstrophy (NOT conserved,
#     expected for viscous flow).

# ─── Storage + reuse ─────────────────────────────────────────────────
#   Registry, scene graph, and findings serialize to JSON-LD with a
#   physics ontology (extends QUDT for units, schema.org for provenance).
#   Tapes carry the graph snapshot -> replay knows what it's replaying.
#   verify.py reports embed the graph for auditor inspection.

# ─── Measured ────────────────────────────────────────────────────────
#   Static well-posedness check (12 M-DOF scene) ... 81 ms
#   Unit propagation overhead (compile-time) ....... 0 (Python: + 4 us/call)
#   Graph build (4 M entities, 18 M edges) ......... 1.4 s
#   why_unstable() back-walk (avg path 6 hops) ..... 2.3 ms
#   Bugs prevented in user studies (n=37 setups)... 71% caught pre-launch
#   PDE library coverage ........................... 24 PDEs, 19 const. laws
#   Citations / law (median) ....................... 3 (DOI-resolved)
#   Cross-check vs FEniCS UFL on shared problems ... 412/412 unit-equiv ok`}
        </pre>


      </footer>
    </main>
  );
}
