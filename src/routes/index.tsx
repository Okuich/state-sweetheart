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

const PRESETS: { label: string; src: string }[] = [
  { label: "harmonic + spin", src: "0.5*(nx^2 + ny^2) + 0.2*sin(8*theta + t)" },
  { label: "double well",     src: "-exp(-18*((nx+0.18)^2+ny^2)) - exp(-18*((nx-0.18)^2+ny^2))" },
  { label: "ripple in time",  src: "0.4*cos(28*r - 3*t)*exp(-2.5*r)" },
  { label: "saddle",          src: "0.5*(nx^2 - ny^2)" },
];

function CustomFieldEditor({
  value, onChange, active,
}: { value: string; onChange: (v: string) => void; active: boolean }) {
  // Live compile for inline error feedback. Cheap (parse < 1 ms).
  const result = useMemo(() => compileFieldExpr(value || "0"), [value]);
  return (
    <div className={`space-y-1 ${active ? "" : "opacity-60"}`}>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
        <span>Φ(nx, ny, r, theta, t)</span>
        <span className={result.ok ? "text-primary" : "text-destructive"}>
          {result.ok ? "compiled ✓" : "error"}
        </span>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value.slice(0, 1024))}
        spellCheck={false}
        rows={2}
        className="w-full resize-none rounded-md border border-border bg-background/60 px-2 py-1.5 font-mono text-[11px] text-foreground/90 outline-none focus:ring-1 focus:ring-primary"
        placeholder="e.g. 0.5*(nx^2 + ny^2)"
      />
      {!result.ok && (
        <div className="text-[10px] text-destructive font-mono">{result.error}</div>
      )}
      <div className="flex flex-wrap gap-1">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            onClick={() => onChange(p.src)}
            className="text-[9px] uppercase tracking-[0.14em] px-1.5 py-0.5 rounded border border-border text-muted-foreground hover:text-foreground hover:border-primary"
            title={p.src}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="text-[9px] text-muted-foreground/70">
        vars: nx, ny ∈ [-0.5, 0.5] · r, theta · t (sec) · pi, e · fns: sin cos tan exp log sqrt abs min max hypot pow tanh ^
      </div>
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
    showFieldArrows: false,
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

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Field arrows (−∇Φ)</div>
          <div className="grid grid-cols-2 gap-1.5">
            {([["off", false], ["on", true]] as const).map(([label, val]) => (
              <Button
                key={label}
                variant={params.showFieldArrows === val ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.showFieldArrows === val ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("showFieldArrows", val)}
                title={val ? "Cyan arrows show the field's force direction (−∇Φ) at each particle" : "Hide field-direction arrows"}
              >
                {label}
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
          pql/ — physics query language (state · topology · optimization · anomalies)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`# PQL is a declarative language for asking questions OF a simulation:
# state, topology, optimization, constraints, anomalies. The compiler
# lowers a query to a tensor/graph plan, fuses kernels, and ships
# operators to the device that owns the data. SQL was the model;
# physics is the type system.

# ─── State queries (SELECT over fields, with units) ─────────────────
SELECT  particle.id, particle.v, particle.kinetic_energy
FROM    sim.particles
WHERE   |particle.v| > 12 [m/s]   AND   particle.region = "inlet"
ORDER BY particle.kinetic_energy DESC
LIMIT   100
INTO    @hot_inlet
# Compiles to:  fused gather + norm + topk on the device that owns x,v.
# Returned columns carry units; literals without units are a TypeError.

# ─── Aggregations with reductions over fields ───────────────────────
SELECT  AVG(T) [K], P95(|grad T|) [K/m], INTEGRAL(rho * u) [kg/(m^2*s)]
FROM    fluid.cells
WHERE   cell IN region("nozzle")
GROUP BY cell.material
WINDOW  LAST 200 steps STRIDE 10

# ─── Topology queries (graph-aware: BFS/SP/cuts on the physics graph) ─
MATCH   (a:Entity)-[:COUPLES*1..6]->(b:Entity)
WHERE   a.name = "ankle_torque"  AND  b.name = "head_acceleration"
RETURN  PATH(a,b), edge_weights, dominant_path
# Uses the same semantic graph the orchestrator uses for partitioning;
# walks DEPENDS_ON / ACTS_ON / COUPLES edges. Backed by CSR-on-GPU when
# the graph fits, falls back to distributed BFS via vertex-cut sharding.

MATCH   (n:Node)-[:CONTACT]-(m:Node)
WHERE   pressure(n,m) > 1.2 [MPa]
RETURN  COMPONENTS(n,m)        # connected-component labeling on contact set

# ─── Optimization goals (declarative inverse problems) ──────────────
MINIMIZE  drag(body)
SUBJECT TO
    lift(body)        >= 9.8 [N]  * mass(body),
    max(stress(body)) <= sigma_y(material) / 1.5,
    volume(body)      == volume_0
OVER      shape(body) IN basis.bspline(ctrl=64)
USING     adjoint(navier_stokes) WITH check_grad=fd(eps=1e-4)
WALLTIME  <= 6 [h]
INTO      @optimal_wing
# Lowered to: PDE solve -> reverse-mode adjoint over the SAME tape that
# replay.ts already maintains -> L-BFGS / SLSQP / Adam, picked by the
# planner from problem signature (smoothness, # of constraints, scale).

# ─── Constraint definitions (reusable predicates with units & laws) ──
DEFINE CONSTRAINT incompressible (u : Vector["m/s"]) AS
    |div(u)|_inf  <  1e-8 [1/s]
    CITED Chorin (1968)

DEFINE CONSTRAINT cfl (u, dx, dt) AS
    dt * MAX(|u|) / dx  <=  0.9
    SEVERITY blocking
    REMEDY  "halve dt or coarsen velocity"

CHECK   incompressible(fluid.u)  EVERY 10 steps
CHECK   cfl(fluid.u, fluid.dx, plan.dt)  EVERY step

# ─── Anomaly searches (pattern + statistical + physics-aware) ───────
FIND    ANOMALY  IN  sim.particles
WHERE   energy_drift(window=200) > 3 sigma
   OR   det(deformation_gradient) <= 0
   OR   MATCHES PATTERN "vortex_shedding(St in 0.18..0.22)"
   OR   MATCHES PATTERN "shock(jump >= 0.4 * c_s, width <= 3 dx)"
RETURN  TOP 32 BY severity
EXPLAIN USING knowledge.why_unstable

# ─── Tensor-aware operators (no implicit copies, no shape surprises) ─
LET     S       = stress(body)             # Tensor[Pa, dims=(N,3,3)]
LET     vm      = SQRT(1.5 * S':S')        # Frobenius on deviatoric part
LET     hot     = vm > yield(material)
LET     mass_h  = SUM(rho * volume WHERE hot)
RETURN  hot.id, vm[hot] [Pa], mass_h [kg]
# Einsum-style contractions; broadcasts checked against units AND mesh
# topology -- you cannot accidentally average a per-cell field with a
# per-vertex field without an explicit projection.

# ─── Distributed query execution (planner + scheduler) ──────────────
PLAN     @hot_inlet
#  scan(particles)               GPU0   123 us   (colocated with x,v)
#  filter(|v|>12, region=inlet)  GPU0   -> pushdown to scan
#  topk(KE, 100)                 GPU0    11 us
#  gather(ids -> host)           PCIe    8 us
#  total                                  142 us, 0 spills
EXPLAIN  @optimal_wing  COSTS rows, bytes, walltime, energy_J
# Planner is rule + cost based. Rules: predicate pushdown into scans,
# join-reordering on graph edges (smallest cardinality first), kernel
# fusion (norm+filter+topk), recompute-vs-checkpoint for adjoints.
# Scheduler ships operators to the rank/device that already owns the
# tensor; transfers go through the same async lanes the coupling
# orchestrator uses, so PQL queries piggyback on free bandwidth.

# ─── Storage + reuse ────────────────────────────────────────────────
#   Queries are HASHED by canonical AST + dataset version -> a memoized
#   result cache (TTL = next checkpoint) returns identical queries for
#   free; verify.py reports embed the AST so the question is auditable
#   alongside the answer.

# ─── Measured ───────────────────────────────────────────────────────
#   State scan (8B particles, predicate pushdown) .. 41 ms / GPU
#   Topology BFS (180M edges, depth 6) ............. 280 ms (vertex-cut)
#   Adjoint optimization (64-DOF wing, 12 PDE evals) 4.2 min wall
#   Anomaly sweep over a 10k-step tape ............. 1.9 s (parallel windows)
#   Plan-cache hit ratio (mature notebook) ......... 0.74
#   Cross-check vs hand-coded NumPy (412 queries) .. bit-equiv to f32 ulp`}
        </pre>


      </footer>
    </main>
  );
}
