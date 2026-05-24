import { useEffect, useMemo, useRef, useState } from "react";
import {
  defaultConfig,
  buildOperator,
  initState,
  surrogateStep,
  fullSolveStep,
  refineOperator,
  benchmark,
  emptyManifold,
  manifoldUpdate,
  manifoldEncode,
  effectiveDim,
  encodeSpectral,
  turbulent,
  gaussianBlob,
  l2Error,
} from "@/lib/piai";
import type { PIConfig, SurrogateMetrics } from "@/lib/piai";
import { Slider } from "@/components/ui/slider";
import { Button } from "@/components/ui/button";

type Init = "turbulent" | "blob";

function FieldView({ field, N, label }: { field: Float32Array; N: number; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext("2d"); if (!ctx) return;
    const img = ctx.createImageData(N, N);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < field.length; i++) { if (field[i] < lo) lo = field[i]; if (field[i] > hi) hi = field[i]; }
    const span = (hi - lo) || 1;
    for (let i = 0; i < field.length; i++) {
      const t = (field[i] - lo) / span;
      // teal → magenta gradient
      const r = Math.floor(40 + 200 * Math.max(0, t - 0.4));
      const g = Math.floor(220 * (1 - Math.abs(t - 0.5) * 2));
      const b = Math.floor(180 + 60 * (1 - t));
      img.data[i * 4 + 0] = r;
      img.data[i * 4 + 1] = g;
      img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [field, N]);
  return (
    <div className="space-y-1">
      <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <canvas
        ref={canvasRef}
        width={N}
        height={N}
        className="w-full aspect-square rounded-md border border-border bg-background image-rendering-pixelated"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  );
}

function Metric({ label, value, unit, good }: { label: string; value: string; unit?: string; good?: boolean }) {
  return (
    <div className="rounded-md border border-border bg-background/40 p-2.5">
      <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-lg tabular-nums ${good ? "text-primary" : "text-foreground"}`}>
        {value}
        {unit && <span className="ml-1 text-xs text-muted-foreground">{unit}</span>}
      </div>
    </div>
  );
}

export function PhysicsInformedAIPanel() {
  const [cfg, setCfg] = useState<PIConfig>(defaultConfig());
  const [initKind, setInitKind] = useState<Init>("turbulent");
  const [refineCount, setRefineCount] = useState(0);
  const [tick, setTick] = useState(0);

  const op = useMemo(() => buildOperator(cfg), [cfg]);
  const stateRef = useRef(initState(cfg));
  const truthRef = useRef<Float32Array>(stateRef.current.field);
  const manifoldRef = useRef(emptyManifold(cfg.operatorModes * cfg.operatorModes, cfg.latentDim));
  const [metrics, setMetrics] = useState<SurrogateMetrics>(stateRef.current.metrics);

  // reset when config / init changes
  useEffect(() => {
    const init = initKind === "turbulent" ? turbulent(cfg.gridN) : gaussianBlob(cfg.gridN);
    stateRef.current = { ...initState(cfg), field: init };
    truthRef.current = init;
    manifoldRef.current = emptyManifold(cfg.operatorModes * cfg.operatorModes, Math.min(cfg.latentDim, cfg.operatorModes * cfg.operatorModes));
    setTick((t) => t + 1);
  }, [cfg, initKind]);

  // adaptive refinement passes (online physics-informed learning)
  useEffect(() => {
    if (refineCount === 0) return;
    for (let r = 0; r < refineCount; r++) refineOperator(op, stateRef.current.field, cfg);
    setTick((t) => t + 1);
  }, [refineCount, op, cfg]);

  // animation loop: step both surrogate and truth
  useEffect(() => {
    let raf = 0; let last = performance.now(); let acc = 0;
    const loop = () => {
      const now = performance.now();
      acc += now - last; last = now;
      if (acc >= 80) {
        acc = 0;
        const s = stateRef.current;
        s.field = surrogateStep(s.field, op);
        truthRef.current = fullSolveStep(truthRef.current, cfg.gridN, cfg.pdeViscosity, cfg.pdeDt);
        s.step += 1;
        // update manifold with current spectral encoding
        const coeffs = encodeSpectral(s.field, op);
        manifoldUpdate(manifoldRef.current, coeffs);
        // every 30 steps, run benchmark
        if (s.step % 30 === 0) {
          const m = benchmark(cfg, op, s.field, { runs: 4 });
          m.l2Error = l2Error(s.field, truthRef.current);
          m.manifoldDim = effectiveDim(manifoldRef.current);
          setMetrics(m);
        }
        setTick((t) => t + 1);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [op, cfg]);

  const gateAccel = metrics.speedup >= 2.0;          // ≥50% acceleration → speedup ≥2×
  const gateCost = metrics.costReductionPct >= 30;
  const gatePred = metrics.predImprovementPct >= 20;
  const allPass = gateAccel && gateCost && gatePred;

  return (
    <div className="rounded-xl border border-border bg-card p-6 backdrop-blur-sm space-y-6">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="text-[10px] uppercase tracking-[0.32em] text-muted-foreground">Physics-Informed AI</div>
          <h2 className="font-display text-2xl mt-1">Latent metric space surrogate</h2>
          <p className="text-xs text-muted-foreground mt-1 max-w-xl">
            Neural operator (spectral) + manifold learning (online PCA) + GNN polish.
            Encode field → step in latent → decode. Adaptive refinement nudges the operator
            toward the true PDE step.
          </p>
        </div>
        <div className={`px-3 py-1.5 rounded-md text-[10px] uppercase tracking-[0.22em] ${allPass ? "bg-primary/15 text-primary" : "bg-muted/30 text-muted-foreground"}`}>
          {allPass ? "all targets met" : "warming up"}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr,1fr,1.4fr] gap-4">
        <FieldView field={stateRef.current.field} N={cfg.gridN} label={`surrogate · step ${stateRef.current.step}`} />
        <FieldView field={truthRef.current} N={cfg.gridN} label="ground truth (full PDE)" />
        <div className="grid grid-cols-2 gap-2">
          <Metric label="speedup" value={metrics.speedup.toFixed(2)} unit="×" good={gateAccel} />
          <Metric label="cost reduction" value={metrics.costReductionPct.toFixed(1)} unit="%" good={gateCost} />
          <Metric label="pred. improvement" value={metrics.predImprovementPct.toFixed(1)} unit="%" good={gatePred} />
          <Metric label="L2 vs truth" value={metrics.l2Error.toExponential(2)} />
          <Metric label="PDE residual" value={metrics.pdeResidual.toExponential(2)} />
          <Metric label="effective manifold dim" value={metrics.manifoldDim.toFixed(2)} />
          <Metric label="full solve" value={metrics.fullSolveMs.toFixed(2)} unit="ms" />
          <Metric label="surrogate" value={metrics.surrogateMs.toFixed(2)} unit="ms" />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">grid N</span>
            <span className="text-primary tabular-nums">{cfg.gridN}</span>
          </div>
          <Slider value={[cfg.gridN]} min={16} max={64} step={4}
            onValueChange={([v]) => setCfg((c) => ({ ...c, gridN: v }))} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">spectral modes</span>
            <span className="text-primary tabular-nums">{cfg.operatorModes}</span>
          </div>
          <Slider value={[cfg.operatorModes]} min={3} max={12} step={1}
            onValueChange={([v]) => setCfg((c) => ({ ...c, operatorModes: v }))} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">latent dim</span>
            <span className="text-primary tabular-nums">{cfg.latentDim}</span>
          </div>
          <Slider value={[cfg.latentDim]} min={2} max={32} step={1}
            onValueChange={([v]) => setCfg((c) => ({ ...c, latentDim: v }))} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">viscosity ν</span>
            <span className="text-primary tabular-nums">{cfg.pdeViscosity.toFixed(3)}</span>
          </div>
          <Slider value={[cfg.pdeViscosity * 1000]} min={1} max={80} step={1}
            onValueChange={([v]) => setCfg((c) => ({ ...c, pdeViscosity: v / 1000 }))} />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.18em]">
        <span className="text-muted-foreground mr-2">init:</span>
        <Button variant={initKind === "turbulent" ? "default" : "outline"} size="sm" onClick={() => setInitKind("turbulent")}>turbulent</Button>
        <Button variant={initKind === "blob" ? "default" : "outline"} size="sm" onClick={() => setInitKind("blob")}>blob</Button>
        <span className="mx-2 text-muted-foreground/40">·</span>
        <Button size="sm" variant="outline" onClick={() => setRefineCount((n) => n + 5)}>
          refine operator ×5
        </Button>
        <Button size="sm" variant="outline" onClick={() => {
          const m = benchmark(cfg, op, stateRef.current.field, { runs: 8 });
          m.l2Error = l2Error(stateRef.current.field, truthRef.current);
          m.manifoldDim = effectiveDim(manifoldRef.current);
          setMetrics(m);
        }}>
          re-benchmark
        </Button>
        <span className="ml-auto text-muted-foreground">refinement passes: <span className="text-primary tabular-nums">{refineCount}</span></span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[10px] uppercase tracking-[0.18em]">
        <div className={`rounded-md border px-3 py-2 ${gateAccel ? "border-primary/40 bg-primary/5 text-primary" : "border-border text-muted-foreground"}`}>
          <div>≥50% acceleration</div>
          <div className="text-foreground/70 mt-0.5 font-mono normal-case tracking-normal">{metrics.speedup.toFixed(2)}× speedup</div>
        </div>
        <div className={`rounded-md border px-3 py-2 ${gateCost ? "border-primary/40 bg-primary/5 text-primary" : "border-border text-muted-foreground"}`}>
          <div>≥30% cost reduction</div>
          <div className="text-foreground/70 mt-0.5 font-mono normal-case tracking-normal">{metrics.costReductionPct.toFixed(1)}%</div>
        </div>
        <div className={`rounded-md border px-3 py-2 ${gatePred ? "border-primary/40 bg-primary/5 text-primary" : "border-border text-muted-foreground"}`}>
          <div>≥20% prediction improvement</div>
          <div className="text-foreground/70 mt-0.5 font-mono normal-case tracking-normal">{metrics.predImprovementPct.toFixed(1)}% vs naive</div>
        </div>
      </div>

      <div className="text-[10px] text-muted-foreground/60" data-tick={tick}>
        latent encode: {(() => {
          const z = manifoldEncode(manifoldRef.current, encodeSpectral(stateRef.current.field, op));
          return Array.from(z).slice(0, 6).map((v) => v.toFixed(2)).join(" · ");
        })()} …
      </div>
    </div>
  );
}
