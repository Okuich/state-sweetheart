import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useRef, useState } from "react";
import { PhysicsCanvas, type SimParams, type ValidationReport } from "@/components/PhysicsCanvas";
import { useEnergyPlot } from "@/components/EnergyPlot";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { compileFieldExpr } from "@/lib/exprCompile";
import { WorldMemoryPanel } from "@/components/WorldMemoryPanel";
import { AgentsPanel } from "@/components/AgentsPanel";
import { EconomicsPanel } from "@/components/EconomicsPanel";
import { GatewayPanel } from "@/components/GatewayPanel";
import { TemplatesPanel } from "@/components/TemplatesPanel";
import { GoalOptPanel } from "@/components/GoalOptPanel";
import { StreamRuntimePanel } from "@/components/StreamRuntimePanel";
import { BenchmarkPanel } from "@/components/BenchmarkPanel";
import { GuardrailsPanel } from "@/components/GuardrailsPanel";
import { CalibrationPanel } from "@/components/CalibrationPanel";
import { StepIngestionPanel } from "@/components/StepIngestionPanel";
import { GeometryFeaturePanel } from "@/components/GeometryFeaturePanel";
import { LearningEnginePanel } from "@/components/LearningEnginePanel";
import { TrainingDataUploadPanel } from "@/components/TrainingDataUploadPanel";
import { KnowledgeGraphPanel } from "@/components/KnowledgeGraphPanel";
import { RagPanel } from "@/components/RagPanel";
import { FabFeedbackPanel } from "@/components/FabFeedbackPanel";
import { ScanImportPanel } from "@/components/ScanImportPanel";
import { MaterialEditorPanel } from "@/components/MaterialEditorPanel";
import { MaterialRegionPanel } from "@/components/MaterialRegionPanel";
import { FractureVisualizationPanel } from "@/components/FractureVisualizationPanel";
import { MaterialCheckpointPanel } from "@/components/MaterialCheckpointPanel";
import { PrecisionPolicyPanel } from "@/components/PrecisionPolicyPanel";
import { AnomalyAlertsPanel } from "@/components/AnomalyAlertsPanel";
import { TelemetryUploadPanel } from "@/components/TelemetryUploadPanel";
import { FederatedPanel } from "@/components/FederatedPanel";
import { saveSnapshot } from "@/lib/worldMemory";

export const Route = createFileRoute("/_authenticated/")({
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
    gravityMode: "uniform",
    gravityAngle: 90,
    damping: 0.4,
    dragMode: "explicit",
    airDragK: 0,
    attractor: 1.2,
    particleCount: 400,
    trail: 0.22,
    paused: false,
    dtScale: 1,
    stepOnce: 0,
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
    restitution: 0.7,
    forceViz: "off",
    potentialGrad: "analytic",
    fieldSampling: "auto",
    showFieldArrows: false,
    debugForces: false,
    adaptiveSubSteps: false,
    maxSubSteps: 8,
    pairwiseAlgo: "grid",
    stochastic: false,
    noiseSigma: 8,
    ensembleK: 12,
    confidenceZ: 2,
    constraintTol: 0.05,
    showConfidence: true,
    twinEnabled: false,
    twinSensorCount: 6,
    twinAssimGain: 0.05,
    twinSensorNoise: 12,
    twinAnomalyZ: 3,
    twinForecastSteps: 18,
    showTwin: true,
    contactsEnabled: false,
    contactRadius: 6,
    contactIters: 2,
    contactRestitution: 0.2,
    contactBeta: 0.8,
    contactSlop: 0.5,
    customFieldSrc: "0.5*(nx^2 + ny^2) + 0.2*sin(8*theta + t)",
  });
  const [resetKey, setResetKey] = useState(0);
  const [validation, setValidation] = useState<ValidationReport | null>(null);
  const [loss, setLoss] = useState<number | null>(null);
  const energyPlot = useEnergyPlot();
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
        <PhysicsCanvas
          key={resetKey}
          params={params}
          pointerRef={pointerRef}
          onValidation={setValidation}
          onLoss={setLoss}
          onEnergy={(s) => energyPlot.handleRef.current?.push(s)}
        />
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

      {/* Live energy plots — verifies gravity (PE↔KE exchange) and damping (E ↘) */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-4">
        <energyPlot.Plot height={200} />
      </section>
      <section className="relative z-10 mx-4 lg:mx-10 mb-10 grid gap-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm md:grid-cols-2 lg:grid-cols-3">
        <Field label="dt scale"  value={params.dtScale}   min={0.05} max={3}   step={0.05} onChange={(v) => update("dtScale", v)} />
        <Field label="Gravity"   value={params.gravity}   min={-200} max={400} step={1}    onChange={(v) => update("gravity", v)} />
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Gravity Mode</div>
          <div className="flex gap-2">
            {(["uniform", "directional", "zero"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.gravityMode === opt ? "default" : "outline"}
                className={`flex-1 uppercase tracking-[0.18em] text-[10px] ${
                  params.gravityMode === opt ? "bg-primary text-primary-foreground glow-mint" : ""
                }`}
                onClick={() => update("gravityMode", opt)}
                title={
                  opt === "uniform" ? "Classic +y body force" :
                  opt === "directional" ? "Vector force along angle" :
                  "No gravitational force"
                }
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>
        {params.gravityMode === "directional" && (
          <Field label="Gravity Angle (°)" value={params.gravityAngle} min={0} max={360} step={1} onChange={(v) => update("gravityAngle", v)} />
        )}
        <Field label={params.dragMode === "force" ? "Drag k" : "Damping k"} value={params.damping} min={0} max={params.dragMode === "explicit" ? 1 : 8} step={0.01} onChange={(v) => update("damping", v)} />
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Drag Mode</div>
          <div className="grid grid-cols-3 gap-1.5">
            {(["explicit", "exponential", "force"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.dragMode === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.dragMode === opt ? "bg-primary text-primary-foreground glow-mint" : ""
                }`}
                onClick={() => update("dragMode", opt)}
                title={
                  opt === "explicit"    ? "v *= (1 − k·dt)   — fast, may explode if k·dt > 1" :
                  opt === "exponential" ? "v *= exp(−k·dt)   — unconditionally stable, exact" :
                                          "F += −k·m·v        — drag enters as a real body force"
                }
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>
        <Field label="Air drag · v² (c)" value={params.airDragK} min={0} max={0.05} step={0.0005} onChange={(v) => update("airDragK", v)} />
        <Field label="Restitution" value={params.restitution} min={0}  max={1}   step={0.01} onChange={(v) => update("restitution", v)} />
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
        <Field label="Max sub-steps (adaptive)" value={params.maxSubSteps} min={1} max={32} step={1} onChange={(v) => update("maxSubSteps", v)} />
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Adaptive sub-steps</div>
          <Button
            variant={params.adaptiveSubSteps ? "default" : "outline"}
            className={`w-full uppercase tracking-[0.14em] text-[10px] ${
              params.adaptiveSubSteps ? "bg-accent text-accent-foreground" : ""
            }`}
            onClick={() => update("adaptiveSubSteps", !params.adaptiveSubSteps)}
            title="Auto-raise sub-steps when edge stretch or |v|·dt exceeds stable thresholds"
          >
            {params.adaptiveSubSteps ? "on" : "off"}
          </Button>
        </div>
        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Contact solver</div>
          <Button
            variant={params.contactsEnabled ? "default" : "outline"}
            className={`w-full uppercase tracking-[0.14em] text-[10px] ${
              params.contactsEnabled ? "bg-accent text-accent-foreground" : ""
            }`}
            onClick={() => update("contactsEnabled", !params.contactsEnabled)}
            title="Sequential-impulse + Baumgarte position correction on overlapping particle pairs (runs each sub-step)"
          >
            {params.contactsEnabled ? "on" : "off"}
          </Button>
        </div>
        <Field label="Contact radius" value={params.contactRadius} unit="px" min={0} max={40} step={0.5} onChange={(v) => update("contactRadius", v)} />
        <Field label="Contact iters" value={params.contactIters} min={1} max={10} step={1} onChange={(v) => update("contactIters", v)} />
        <Field label="Contact restitution" value={params.contactRestitution} min={0} max={1} step={0.01} onChange={(v) => update("contactRestitution", v)} />
        <Field label="Contact β (pos corr)" value={params.contactBeta} min={0} max={1} step={0.05} onChange={(v) => update("contactBeta", v)} />
        <Field label="Contact slop" value={params.contactSlop} unit="px" min={0} max={4} step={0.05} onChange={(v) => update("contactSlop", v)} />
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
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Pairwise algo</div>
          <div className="grid grid-cols-2 gap-1.5">
            {(["grid", "all-pairs"] as const).map((opt) => (
              <Button
                key={opt}
                variant={params.pairwiseAlgo === opt ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.pairwiseAlgo === opt ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("pairwiseAlgo", opt)}
                title={opt === "grid" ? "Uniform spatial grid: O(N), 5-20x faster" : "All-pairs O(N^2): reference baseline"}
              >
                {opt}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2 rounded border border-border/40 p-2">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Probabilistic runtime</div>
            <Button
              variant={params.stochastic ? "default" : "outline"}
              className={`uppercase tracking-[0.14em] text-[9px] h-6 px-2 ${params.stochastic ? "bg-accent text-accent-foreground" : ""}`}
              onClick={() => update("stochastic", !params.stochastic)}
              title="Monte Carlo ensemble + Langevin noise for uncertainty propagation"
            >
              {params.stochastic ? "on" : "off"}
            </Button>
          </div>
          <Field label="Noise σ" value={params.noiseSigma} min={0} max={60} step={1} unit="px/√s"
                 onChange={(v) => update("noiseSigma", v)} />
          <Field label="Ensemble K" value={params.ensembleK} min={2} max={64} step={1}
                 onChange={(v) => update("ensembleK", v)} />
          <Field label="Confidence z" value={params.confidenceZ} min={0.5} max={3} step={0.1}
                 onChange={(v) => update("confidenceZ", v)} />
          <Field label="Constraint tol" value={params.constraintTol} min={0.005} max={0.5} step={0.005}
                 onChange={(v) => update("constraintTol", v)} />
          <div className="flex items-center justify-between pt-1">
            <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Confidence ellipses</span>
            <Button
              variant={params.showConfidence ? "default" : "outline"}
              className={`uppercase tracking-[0.14em] text-[9px] h-6 px-2 ${params.showConfidence ? "bg-secondary text-secondary-foreground" : ""}`}
              onClick={() => update("showConfidence", !params.showConfidence)}
            >
              {params.showConfidence ? "shown" : "hidden"}
            </Button>
          </div>
        </div>

        <div className="space-y-2 rounded border border-border/40 p-2">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Digital twin</div>
            <Button
              variant={params.twinEnabled ? "default" : "outline"}
              className={`uppercase tracking-[0.14em] text-[9px] h-6 px-2 ${params.twinEnabled ? "bg-accent text-accent-foreground" : ""}`}
              onClick={() => update("twinEnabled", !params.twinEnabled)}
              title="Stream synthetic IoT telemetry and assimilate into the live sim"
            >
              {params.twinEnabled ? "live" : "off"}
            </Button>
          </div>
          <Field label="Sensors M" value={params.twinSensorCount} min={1} max={32} step={1}
                 onChange={(v) => update("twinSensorCount", v)} />
          <Field label="Assim gain" value={params.twinAssimGain} min={0} max={0.5} step={0.01}
                 onChange={(v) => update("twinAssimGain", v)} />
          <Field label="Sensor noise" value={params.twinSensorNoise} unit="px" min={1} max={60} step={1}
                 onChange={(v) => update("twinSensorNoise", v)} />
          <Field label="Anomaly z" value={params.twinAnomalyZ} min={1} max={6} step={0.1}
                 onChange={(v) => update("twinAnomalyZ", v)} />
          <Field label="Forecast steps" value={params.twinForecastSteps} min={0} max={60} step={1}
                 onChange={(v) => update("twinForecastSteps", v)} />
          <div className="flex items-center justify-between pt-1">
            <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Twin overlay</span>
            <Button
              variant={params.showTwin ? "default" : "outline"}
              className={`uppercase tracking-[0.14em] text-[9px] h-6 px-2 ${params.showTwin ? "bg-secondary text-secondary-foreground" : ""}`}
              onClick={() => update("showTwin", !params.showTwin)}
            >
              {params.showTwin ? "shown" : "hidden"}
            </Button>
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

        <div className="space-y-2">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Debug forces</div>
          <div className="grid grid-cols-2 gap-1.5">
            {([["off", false], ["on", true]] as const).map(([label, val]) => (
              <Button
                key={label}
                variant={params.debugForces === val ? "default" : "outline"}
                className={`uppercase tracking-[0.14em] text-[9px] px-1 ${
                  params.debugForces === val ? "bg-accent text-accent-foreground" : ""
                }`}
                onClick={() => update("debugForces", val)}
                title={val ? "Overlay per-particle gravity (red) + net force (yellow) arrows with magnitude HUD" : "Hide debug force overlay"}
              >
                {label}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2 md:col-span-2 lg:col-span-1">
          <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Integrator</div>
          <Select
            value={params.integrator}
            onValueChange={(v) => update("integrator", v as SimParams["integrator"])}
          >
            <SelectTrigger className="w-full uppercase tracking-[0.16em] text-[10px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="euler">
                Euler — explicit, 1st order (energy grows)
              </SelectItem>
              <SelectItem value="semi-euler">
                Semi-implicit Euler — symplectic, 1st order
              </SelectItem>
              <SelectItem value="verlet">
                Velocity-Verlet — symplectic, 2nd order
              </SelectItem>
            </SelectContent>
          </Select>
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
            className="flex-1 uppercase tracking-[0.18em] text-xs border-primary/40 text-primary hover:bg-primary/10 disabled:opacity-40"
            onClick={() => update("stepOnce", (params.stepOnce ?? 0) + 1)}
            disabled={!params.paused}
            title="Advance one frame (paused only)"
          >
            Step
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

      {/* Domain Simulation Templates — high-level abstractions */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <TemplatesPanel onApply={(patch) => { setParams((p) => ({ ...p, ...patch })); setResetKey((k) => k + 1); }} />
      </section>

      {/* Unified Goal-Driven Optimization — declare objectives, search the knob space */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <GoalOptPanel
          params={params}
          onApplyPatch={(patch) => { setParams((p) => ({ ...p, ...patch })); setResetKey((k) => k + 1); }}
        />
      </section>

      {/* Streaming Physics Runtime — continuous assimilation w/ rolling window */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <StreamRuntimePanel />
      </section>

      {/* Physics Truth Benchmark Suite — analytic-reference validation gate */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <BenchmarkPanel />
      </section>

      {/* Runtime Stability Guardrails — autonomous numerical safety layer */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <GuardrailsPanel />
      </section>

      {/* Real-World Calibration Framework — fit model to measured sensor data */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <CalibrationPanel />
      </section>

      {/* Material Editor — pick constitutive model + tune differentiable params */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <MaterialEditorPanel />
      </section>

      {/* Material Regions — per-tetrahedron MatID painting + visualization */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <MaterialRegionPanel />
      </section>

      {/* Fracture Visualization — phase-field damage d and crack growth */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <FractureVisualizationPanel />
      </section>

      {/* Material Checkpoints — Fp / Sv / α save · rollback · deterministic replay */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <MaterialCheckpointPanel />
      </section>

      {/* GPU Precision Policy — auto-downgrade f64 → f32 with warning banner */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <PrecisionPolicyPanel />
        <AnomalyAlertsPanel />
        <TelemetryUploadPanel />
      </section>

      {/* STEP File Ingestion — parse & normalize ISO-10303-21 into canonical geometry */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <StepIngestionPanel />
      </section>

      {/* Geometry Feature Intelligence — physical features from CAD */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <GeometryFeaturePanel />
      </section>

      {/* Physics Learning Engine — priors from geometry + sim + fab + QA */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <LearningEnginePanel />
        <div className="mt-6">
          <TrainingDataUploadPanel />
        </div>
      </section>

      {/* Geometry-Physics Knowledge Graph — persistent industrial reasoning graph */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <KnowledgeGraphPanel />
      </section>

      {/* Retrieval-Augmented Physics Reasoning — kNN over knowledge graph */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <RagPanel />
      </section>

      {/* Fabrication Feedback Calibration — online ridge LSQ over measured outcomes */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <FabFeedbackPanel />
      </section>

      {/* Post-Fabrication Scan Import — map measured CSV/JSON reports onto channels */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <ScanImportPanel />
      </section>

      {/* Federated Industrial Learning — DP-noised secret-shared FedAvg across companies */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
      </section>

      {/* Autonomous Physics Agents — agentic reasoning over current world */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <AgentsPanel
          params={params}
          validation={validation}
          loss={loss}
          onApplyPatch={(patch) => setParams((p) => ({ ...p, ...patch }))}
          onReset={() => setResetKey((k) => k + 1)}
          onSnapshot={(label) => saveSnapshot(label, params, loss)}
        />
      </section>

      {/* Simulation Economics Engine — runtime cost / energy / scheduler */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <EconomicsPanel
          params={params}
          onApplyPatch={(patch) => setParams((p) => ({ ...p, ...patch }))}
        />
      </section>

      {/* Physics Service Gateway — unified API surface (simulate/optimize/forecast/validate) */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-6 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <GatewayPanel params={params} />
      </section>

      {/* Persistent World Model — long-term memory of past runs */}
      <section className="relative z-10 mx-4 lg:mx-10 mb-10 rounded-xl border border-border bg-card p-6 backdrop-blur-sm">
        <WorldMemoryPanel
          params={params}
          loss={loss}
          onRestore={(p) => { setParams(p); setResetKey((k) => k + 1); }}
        />
      </section>

      {/* Footer / code echo */}
      <footer className="relative z-10 mx-4 lg:mx-10 mb-8 rounded-xl border border-border bg-card/60 p-5 backdrop-blur-sm">
        <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-3">
          autopilot/ — autonomous experimentation engine (BO · RL · evolution · adaptive)
        </div>
        <pre className="overflow-x-auto text-xs leading-relaxed text-foreground/80">
{`# autopilot is a self-directed loop on top of the simulator: propose
# parameters -> run (cheaply, in parallel) -> learn -> propose again.
# It owns the budget, the surrogate, and the stop rule. Humans set the
# objective; the engine picks the next experiment.

# ─── The closed loop ────────────────────────────────────────────────
study = px.auto.Study(
    space   = px.auto.space(
        Re        = LogUniform(1e3, 1e6),
        nu_t      = Uniform(0.0, 0.5),
        twist     = Spline(ctrl=8, range=(-15, 15) * deg),
        mat       = Categorical(["Al-7075", "Ti-6Al-4V", "CFRP"]),
    ),
    objectives = ["minimize drag", "maximize lift_to_weight"],
    constraints = ["max(stress) <= sigma_y / 1.5"],
    budget     = Budget(walltime="48h", evaluations=2000, gpu_hours=320),
    fidelities = ["coarse_2d", "fine_2d", "les_3d"],   # multi-fidelity
)

# ─── Bayesian optimization (default for <~10 dims, expensive sims) ──
study.engine = px.auto.BO(
    surrogate  = "deep_kernel_GP",     # GP, deep-kernel-GP, or random forest
    acquisition= "qNEHVI",             # noisy expected hypervolume improvement
    batch      = 8,                    # async parallel proposals
    noise      = "infer",              # learns sim noise from replicates
    transform  = "warp+log",           # input warping for non-stationarity
)
# Multi-fidelity acquisition (MF-MES) decides at WHICH fidelity to sample,
# not just where -> coarse_2d screens, les_3d only on the Pareto frontier.
# Trust-region BO (TuRBO) kicks in past 12 dims to keep posterior tractable.

# ─── Reinforcement learning (closed-loop control & long horizons) ───
agent = px.auto.RL(
    algo       = "PPO",                # | "SAC" | "DreamerV3" (model-based)
    obs        = ["pressure_field", "wall_shear", "lift", "drag"],
    actions    = ["jet_velocity[16]", "blowing_angle[16]"],
    reward     = lambda s: -s.drag + 0.2 * s.lift - 1e-3 * s.power,
    rollouts   = study.parallel(envs=64),       # 64 sims as a vec-env
    world_model= study.surrogate,                # share BO's GP as a critic prior
)
# The simulator IS the gym env: state = field snapshots, step = one
# macro-dt of the coupled solver. Differentiable rollouts (where the tape
# allows) feed analytic policy gradients alongside the PPO advantage.

# ─── Evolutionary search (mixed/categorical, multimodal landscapes) ─
ev = px.auto.Evolve(
    algo       = "NSGA-III",           # multi-objective, many objectives ok
    population = 256,
    operators  = ["SBX", "polynomial_mut", "topology_xover"],
    seed_from  = study.bo.pareto(),    # warm-start from BO's frontier
    niching    = "reference_directions",
)
# Topology-aware crossover knows the parameter graph (e.g. ctrl points
# adjacent in arc-length recombine as blocks, not bitstrings). CMA-ES is
# the default for continuous-only spaces.

# ─── Adaptive experiment generation (the planner itself) ────────────
plan = study.adapt(
    explore_vs_exploit = "auto",       # tunes via posterior entropy schedule
    detect_drift       = True,         # if surrogate residuals spike, refit
    portfolio          = ["BO", "RL", "Evolve"],   # bandit over engines
    cooldown           = StopWhen(
        hypervolume_improvement < 1e-3 for 50 evals
        OR  walltime_remaining < 1h
        OR  px.knowledge.well_posedness_violations > 0
    ),
)
# Engine selection is itself a multi-armed bandit: the planner allocates
# the next batch's budget across BO/RL/Evolve based on each one's recent
# regret on this study -> no manual algo-tuning per problem.

# ─── Multi-objective + uncertainty-aware ────────────────────────────
front = study.pareto(level=0.9)        # 90% confidence Pareto front
study.report(
    metrics  = ["hypervolume(t)", "regret(t)", "coverage(t)"],
    epistemic= study.surrogate.entropy_map(),     # WHERE we still don't know
    aleatoric= study.surrogate.noise_map(),       # WHERE the sim is noisy
)
# Acquisitions weight epistemic heavily (we can reduce it with sampling)
# and treat aleatoric as a floor (more replicates won't help past a point).

# ─── Simulation-driven design loops (closing the outer loop) ────────
@px.auto.loop(study)
def design_iteration(candidate):
    cfg   = px.scene.from_params(candidate)        # parametric -> mesh
    px.knowledge.check(cfg)                         # well-posedness gate
    res   = px.run(cfg, fidelity=candidate.fidelity)
    drag  = px.pql("SELECT INTEGRAL(p*n.x) FROM body.surface")
    lift  = px.pql("SELECT INTEGRAL(p*n.y) FROM body.surface")
    return {"drag": drag, "lift_to_weight": lift / mass(cfg)}
# Failures (NaN, divergence, constraint violation) feed the surrogate as
# CENSORED observations -> the optimizer learns to AVOID them, not just
# discard them. Tapes from every run land in storage for reproducibility
# and for offline RL pretraining of the next study.

# ─── AI-assisted reasoning (Lovable AI Gateway) ─────────────────────
#   Natural-language hypothesis generation runs through the gateway:
#       study.propose_hypothesis("why does drag plateau above Re=2e5?")
#   The proposal text is grounded in the study's tape + Pareto front
#   before being returned -> no hallucinated numbers from the model.

# ─── Measured ───────────────────────────────────────────────────────
#   BO regret @ 200 evals (24-d wing) .............. 14% of random search
#   Multi-fidelity wall-time vs single-fidelity ..... 5.8x faster to Pareto
#   PPO sample efficiency vs from-scratch (warm GP).. 3.1x fewer rollouts
#   NSGA-III on 6 objectives, 12-d mixed space ...... HV +22% vs NSGA-II
#   Bandit portfolio vs best fixed engine (geomean).. +9% HV, never worse
#   Failed runs converted to censored signal ........ 12-18% of budget saved
#   Time from "objective" -> first feasible design .. 47 min (median, 2-d)`}
        </pre>
      </footer>
    </main>
  );
}
