// Physics OS — coupled multi-physics co-simulation panel.
// Shows the unified PIKAN running thermal+flow+structural in ONE pass,
// vs the staggered baseline.
import { useEffect, useMemo, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  defaultConfig, initShockScenario, buildCouplingKAN, staggeredStep,
  coupledStep, refineCouplingKAN, benchmark, coupledL2, couplingResidual,
  type CoupledState, type PhysicsOSConfig, type KAN,
} from "@/lib/physicsos";

type ChannelKey = "T" | "p" | "u" | "v" | "s";
const CHANNELS: { key: ChannelKey; label: string; unit: string }[] = [
  { key: "T", label: "Temperature", unit: "K (norm.)" },
  { key: "p", label: "Pressure", unit: "Pa (norm.)" },
  { key: "u", label: "Velocity u", unit: "m/s (norm.)" },
  { key: "v", label: "Velocity v", unit: "m/s (norm.)" },
  { key: "s", label: "Stress σ", unit: "Pa (norm.)" },
];

function colorForT(t: number, kind: ChannelKey): string {
  const x = Math.max(0, Math.min(1, t));
  if (kind === "T") {
    const r = Math.round(255 * x);
    const g = Math.round(64 + 64 * x);
    const b = Math.round(220 * (1 - x));
    return `rgb(${r},${g},${b})`;
  }
  if (kind === "s") {
    const r = Math.round(40 + 215 * x);
    const g = Math.round(20 + 20 * x);
    const b = Math.round(60 - 40 * x);
    return `rgb(${r},${g},${b})`;
  }
  // Flow channels: blue→cyan→white
  const r = Math.round(20 + 200 * x);
  const g = Math.round(80 + 175 * x);
  const b = Math.round(160 + 95 * x);
  return `rgb(${r},${g},${b})`;
}

function FieldCanvas({ field, N, kind }: { field: Float32Array; N: number; kind: ChannelKey }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const ctx = c.getContext("2d"); if (!ctx) return;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < field.length; i++) { if (field[i] < lo) lo = field[i]; if (field[i] > hi) hi = field[i]; }
    const span = hi - lo || 1;
    const px = Math.floor(c.width / N);
    for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
      const t = (field[j * N + i] - lo) / span;
      ctx.fillStyle = colorForT(t, kind);
      ctx.fillRect(i * px, (N - 1 - j) * px, px, px);
    }
  }, [field, N, kind]);
  return <canvas ref={ref} width={128} height={128} className="rounded border border-border/40 bg-background/40" />;
}

function Metric({ label, value, hint, tone = "default" }: {
  label: string; value: string; hint?: string; tone?: "default" | "good" | "warn";
}) {
  const toneCls = tone === "good" ? "text-emerald-400" : tone === "warn" ? "text-amber-400" : "text-foreground";
  return (
    <div className="rounded-md border border-border/40 bg-background/30 p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`mt-0.5 text-xl font-semibold tabular-nums ${toneCls}`}>{value}</div>
      {hint ? <div className="mt-0.5 text-[10px] text-muted-foreground">{hint}</div> : null}
    </div>
  );
}

export function PhysicsOSPanel() {
  const [cfg] = useState<PhysicsOSConfig>(() => defaultConfig());
  const [net, setNet] = useState<KAN>(() => buildCouplingKAN(cfg));
  const [stateA, setStateA] = useState<CoupledState>(() => initShockScenario(cfg.gridN));
  const [stateB, setStateB] = useState<CoupledState>(() => initShockScenario(cfg.gridN));
  const [running, setRunning] = useState(false);
  const [refineOn, setRefineOn] = useState(true);
  const [stepCount, setStepCount] = useState(0);
  const [refineSteps, setRefineSteps] = useState(0);
  const [lastRefineLoss, setLastRefineLoss] = useState(0);
  const [bench, setBench] = useState(() => benchmark(cfg, net, stateA, 4));

  // animation loop
  useEffect(() => {
    if (!running) return;
    let raf = 0; let alive = true;
    const tick = () => {
      if (!alive) return;
      setStateA(prev => staggeredStep(prev, cfg));
      setStateB(prev => coupledStep(prev, net));
      setStepCount(s => s + 1);
      if (refineOn) {
        const loss = refineCouplingKAN(stateA, net, cfg, 24);
        setLastRefineLoss(loss);
        setRefineSteps(r => r + 1);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => { alive = false; cancelAnimationFrame(raf); };
  }, [running, refineOn, cfg, net, stateA]);

  const l2 = useMemo(() => coupledL2(stateB, stateA), [stateA, stateB]);
  const resid = useMemo(() => couplingResidual(stateA, stateB, cfg), [stateA, stateB, cfg]);

  const reset = () => {
    const fresh = initShockScenario(cfg.gridN);
    setStateA(fresh); setStateB(fresh); setStepCount(0); setRefineSteps(0);
  };
  const rebench = () => setBench(benchmark(cfg, net, stateA, 6));
  const rebuild = () => { const n = buildCouplingKAN(cfg); setNet(n); setBench(benchmark(cfg, n, stateA, 6)); };

  const speedupTone = bench.speedup >= 1.5 ? "good" : bench.speedup >= 1 ? "default" : "warn";
  const errTone = bench.coupledL2 <= 0.1 ? "good" : bench.coupledL2 <= 0.25 ? "default" : "warn";
  const improveTone = bench.predImprovementPct >= 20 ? "good" : "default";

  return (
    <Card className="border-border/50 bg-card/50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <span className="text-base">Physics OS — Neural Operator Co-Simulation</span>
          <Badge variant="outline" className="border-violet-500/40 text-violet-300">PIKAN</Badge>
          <Badge variant="outline" className="border-cyan-500/40 text-cyan-300">Coupled Multi-Physics</Badge>
        </CardTitle>
        <p className="text-xs text-muted-foreground">
          A single Physics-Informed Kolmogorov-Arnold Network maps the unified state
          [T, p, u, v, σ] → next state in one pass. Hypersonic shockwave → skin
          deformation → cabin heat, no operator splitting.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" variant={running ? "secondary" : "default"} onClick={() => setRunning(r => !r)}>
            {running ? "Pause" : "Run"}
          </Button>
          <Button size="sm" variant="outline" onClick={reset}>Reset shock</Button>
          <Button size="sm" variant="outline" onClick={rebench}>Re-benchmark</Button>
          <Button size="sm" variant="outline" onClick={rebuild}>Re-init PIKAN</Button>
          <div className="ml-auto flex items-center gap-2">
            <Switch id="phyos-refine" checked={refineOn} onCheckedChange={setRefineOn} />
            <Label htmlFor="phyos-refine" className="text-xs">Online PI refinement</Label>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="Speedup vs staggered" value={`${bench.speedup.toFixed(2)}×`}
                  hint={`${bench.fullSolveMs.toFixed(2)}ms → ${bench.coupledMs.toFixed(2)}ms`}
                  tone={speedupTone} />
          <Metric label="Compute cost reduction" value={`${bench.costReductionPct.toFixed(1)}%`}
                  tone={speedupTone} />
          <Metric label="L2 vs staggered truth" value={bench.coupledL2.toFixed(3)}
                  hint="lower is better" tone={errTone} />
          <Metric label="Pred. improvement vs decoupled" value={`${bench.predImprovementPct.toFixed(1)}%`}
                  hint="coupling captured by PIKAN" tone={improveTone} />
        </div>

        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {(["Staggered baseline", "Coupled PIKAN"] as const).map((title, side) => {
            const st = side === 0 ? stateA : stateB;
            return (
              <div key={title} className="rounded-lg border border-border/40 bg-background/20 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <div className="text-sm font-medium">{title}</div>
                  <Badge variant="outline" className={side === 0
                    ? "border-sky-500/40 text-sky-300"
                    : "border-violet-500/40 text-violet-300"}>
                    {side === 0 ? "T → flow → struct" : "one unified pass"}
                  </Badge>
                </div>
                <div className="grid grid-cols-5 gap-2">
                  {CHANNELS.map(c => (
                    <div key={c.key} className="flex flex-col items-center gap-1">
                      <FieldCanvas field={st[c.key]} N={st.N} kind={c.key} />
                      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{c.label}</div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Metric label="Step" value={`${stepCount}`} />
          <Metric label="Live L2 (B vs A)" value={l2.toFixed(4)} tone={l2 < 0.15 ? "good" : "default"} />
          <Metric label="Coupling residual" value={resid.toExponential(2)}
                  hint="thermo-mech + fluid joint" />
          <Metric label="PI refine steps" value={`${refineSteps}`}
                  hint={refineSteps ? `loss ${lastRefineLoss.toFixed(3)}` : "off"} />
        </div>

        <div className="rounded-md border border-border/40 bg-background/20 p-3 text-xs text-muted-foreground">
          <div className="mb-1 font-medium text-foreground">How it works</div>
          <ul className="list-disc space-y-1 pl-4">
            <li>Each grid cell packs [T, p, u, v, σ] + 4 Laplacian neighbors into a 9-D vector.</li>
            <li>The PIKAN’s edge splines (G={cfg.splineGrid} control points) map 9→5 with no
                channel-specific weights — the same network handles thermal, fluid, and stress.</li>
            <li>Online physics-informed refinement fits one step of staggered ground truth into
                the last-layer spline coefficients (gradient descent on PDE residual).</li>
            <li>Speedup &amp; accuracy are measured against the standard staggered T→flow→struct
                co-simulator on the same shock scenario.</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  );
}

export default PhysicsOSPanel;
