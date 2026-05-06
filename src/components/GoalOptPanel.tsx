import { useMemo, useState } from "react";
import type { SimParams } from "@/components/PhysicsCanvas";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  APP_PRESETS, DEFAULT_KNOBS, optimize,
  type AppKey, type EngineKey, type ObjectiveSpec, type ConstraintSpec, type OptResult,
} from "@/lib/goalOpt";

interface Props {
  params: SimParams;
  onApplyPatch: (patch: Partial<SimParams>) => void;
}

const ENGINES: { key: EngineKey; label: string; blurb: string }[] = [
  { key: "gradient",  label: "Differentiable", blurb: "Finite-diff projected descent" },
  { key: "bayesian",  label: "Bayesian",       blurb: "GP-lite UCB acquisition" },
  { key: "evolution", label: "Evolution",      blurb: "(μ+λ) ES · multi-modal" },
];

const APPS: AppKey[] = ["fabrication", "operational", "maintenance", "custom"];

export function GoalOptPanel({ params, onApplyPatch }: Props) {
  const [app, setApp] = useState<AppKey>("operational");
  const [engine, setEngine] = useState<EngineKey>("bayesian");
  const [iterations, setIterations] = useState(40);
  const [objectives, setObjectives] = useState<ObjectiveSpec[]>(APP_PRESETS.operational.objectives);
  const [constraints, setConstraints] = useState<ConstraintSpec[]>(APP_PRESETS.operational.constraints);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<OptResult | null>(null);

  const choosePreset = (k: AppKey) => {
    setApp(k);
    setObjectives(APP_PRESETS[k].objectives.map((o) => ({ ...o })));
    setConstraints(APP_PRESETS[k].constraints.map((c) => ({ ...c })));
    setResult(null);
  };

  const run = () => {
    setRunning(true);
    // run sync (small budget); yield to UI
    requestAnimationFrame(() => {
      const r = optimize(params, {
        engine, iterations,
        knobs: DEFAULT_KNOBS,
        objectives, constraints,
        seed: 7,
      });
      setResult(r);
      setRunning(false);
    });
  };

  const apply = () => {
    if (!result) return;
    const patch: Partial<SimParams> = {};
    for (const k of DEFAULT_KNOBS) {
      (patch as Record<string, number>)[k.key] = result.best[k.key] as number;
    }
    onApplyPatch(patch);
  };

  const sparkline = useMemo(() => {
    if (!result) return null;
    const h = result.history;
    if (h.length === 0) return null;
    const max = Math.max(...h.map((p) => p.scalar));
    const min = Math.min(...h.map((p) => p.best));
    const W = 320, H = 56;
    const path = h.map((p, i) => {
      const x = (i / Math.max(1, h.length - 1)) * W;
      const y = H - ((p.best - min) / Math.max(1e-9, max - min)) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const cur = h.map((p, i) => {
      const x = (i / Math.max(1, h.length - 1)) * W;
      const y = H - ((p.scalar - min) / Math.max(1e-9, max - min)) * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    return { path, cur, W, H };
  }, [result]);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            autopilot · goal-driven
          </div>
          <h2 className="font-display text-2xl text-foreground">
            Specify <span className="text-primary">objectives</span>, not parameters.
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          engine · {engine} · {result ? `${result.evals} evals` : "idle"}
        </div>
      </div>

      {/* Application preset */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {APPS.map((k) => (
          <button
            key={k}
            onClick={() => choosePreset(k)}
            className={`text-left rounded-md border px-3 py-2 transition ${
              app === k
                ? "border-primary bg-primary/10 text-foreground"
                : "border-border bg-background/40 text-muted-foreground hover:text-foreground"
            }`}
          >
            <div className="text-[10px] uppercase tracking-[0.2em]">{APP_PRESETS[k].label}</div>
            <div className="text-[10px] text-muted-foreground/80 mt-0.5 leading-snug">{APP_PRESETS[k].blurb}</div>
          </button>
        ))}
      </div>

      {/* Engine */}
      <div className="grid grid-cols-3 gap-2">
        {ENGINES.map((e) => (
          <button
            key={e.key}
            onClick={() => setEngine(e.key)}
            className={`rounded-md border px-3 py-2 text-left transition ${
              engine === e.key
                ? "border-accent bg-accent/10 text-foreground"
                : "border-border bg-background/40 text-muted-foreground hover:text-foreground"
            }`}
          >
            <div className="text-[10px] uppercase tracking-[0.2em]">{e.label}</div>
            <div className="text-[10px] text-muted-foreground/80 mt-0.5">{e.blurb}</div>
          </button>
        ))}
      </div>

      {/* Objective weights */}
      <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-1">
          Objectives · weights
        </div>
        {objectives.map((o, idx) => (
          <div key={o.key} className="grid grid-cols-[90px_1fr_50px] items-center gap-3">
            <div className="text-[11px] font-mono text-foreground/90">{o.key}</div>
            <Slider
              value={[o.weight]}
              min={0} max={1} step={0.05}
              onValueChange={([v]) => {
                const n = [...objectives];
                n[idx] = { ...o, weight: v };
                setObjectives(n);
              }}
            />
            <div className="text-[10px] tabular-nums text-primary text-right">{o.weight.toFixed(2)}</div>
          </div>
        ))}
      </div>

      {/* Constraints */}
      {constraints.length > 0 && (
        <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-1">
            Constraints · g(x) ≤ 0
          </div>
          {constraints.map((c, idx) => (
            <div key={c.key} className="grid grid-cols-[140px_1fr_50px] items-center gap-3">
              <div className="text-[11px] font-mono text-foreground/90">{c.key}</div>
              <Slider
                value={[c.bound]}
                min={0} max={1} step={0.05}
                onValueChange={([v]) => {
                  const n = [...constraints];
                  n[idx] = { ...c, bound: v };
                  setConstraints(n);
                }}
              />
              <div className="text-[10px] tabular-nums text-accent text-right">{c.bound.toFixed(2)}</div>
            </div>
          ))}
        </div>
      )}

      {/* Iterations + run */}
      <div className="grid grid-cols-[1fr_auto_auto] items-center gap-3">
        <div>
          <div className="flex items-baseline justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            <span>Iterations</span>
            <span className="text-primary tabular-nums">{iterations}</span>
          </div>
          <Slider value={[iterations]} min={10} max={120} step={5} onValueChange={([v]) => setIterations(v)} />
        </div>
        <Button onClick={run} disabled={running} className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "solving…" : "run"}
        </Button>
        <Button onClick={apply} disabled={!result} variant="outline"
          className="uppercase tracking-[0.18em] text-[10px]">apply</Button>
      </div>

      {/* Result */}
      {result && sparkline && (
        <div className="rounded-md border border-border bg-background/30 p-3 space-y-3">
          <div className="flex items-baseline justify-between">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              loss trace
            </div>
            <div className="text-[10px] uppercase tracking-[0.18em]">
              <span className={result.bestEval.feasible ? "text-primary" : "text-destructive"}>
                {result.bestEval.feasible ? "feasible" : "infeasible"}
              </span>
              <span className="text-muted-foreground"> · scalar </span>
              <span className="text-foreground tabular-nums">{result.bestEval.scalar.toFixed(4)}</span>
            </div>
          </div>
          <svg width={sparkline.W} height={sparkline.H} className="w-full">
            <path d={sparkline.cur} fill="none" stroke="currentColor"
              className="text-muted-foreground/50" strokeWidth={1} />
            <path d={sparkline.path} fill="none" stroke="currentColor"
              className="text-primary" strokeWidth={1.5} />
          </svg>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
            {Object.entries(result.bestEval.objectives).map(([k, v]) => (
              <div key={k} className="rounded border border-border/60 px-2 py-1">
                <div className="uppercase tracking-[0.16em] text-muted-foreground">{k}</div>
                <div className="font-mono text-foreground/90 tabular-nums">{v.toFixed(3)}</div>
              </div>
            ))}
          </div>
          {result.bestEval.constraints.length > 0 && (
            <div className="space-y-1">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">constraint slack</div>
              {result.bestEval.constraints.map((c) => (
                <div key={c.key} className="flex justify-between text-[10px] font-mono">
                  <span className="text-foreground/80">{c.key}</span>
                  <span className={c.slack <= 0 ? "text-primary" : "text-destructive"}>
                    {c.slack.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}
          <div className="rounded border border-border/60 bg-background/40 p-2">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-1">recommended knobs</div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-x-3 gap-y-0.5 text-[10px] font-mono">
              {DEFAULT_KNOBS.map((k) => (
                <div key={k.key} className="flex justify-between">
                  <span className="text-muted-foreground">{k.key}</span>
                  <span className="text-foreground/90 tabular-nums">
                    {(result.best[k.key] as number).toFixed(k.integer ? 0 : 2)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
