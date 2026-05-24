import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  optimize,
  defaultConstraints,
  type Candidate,
  type RunResult,
} from "@/lib/autoeng";

function CADPreview({ c }: { c: Candidate | null }) {
  if (!c) return <div className="aspect-[4/3] rounded-md border border-border bg-background/40" />;
  const v = c.vars;
  const VBW = 240, VBH = 160;
  const scale = Math.min((VBW - 40) / v.length, (VBH - 40) / Math.max(v.width, 20));
  const w = v.length * scale, h = v.width * scale;
  const x0 = (VBW - w) / 2, y0 = (VBH - h) / 2;
  const webH = h * (1 - v.webRatio * 0.6);
  return (
    <svg viewBox={`0 0 ${VBW} ${VBH}`} className="w-full aspect-[3/2] rounded-md border border-border bg-background/40">
      <defs>
        <linearGradient id="ae-mat" x1="0" x2="1">
          <stop offset="0" stopColor="hsl(var(--primary) / 0.45)" />
          <stop offset="1" stopColor="hsl(var(--primary) / 0.15)" />
        </linearGradient>
      </defs>
      {/* outer plate */}
      <rect x={x0} y={y0} width={w} height={h} rx={v.filletR * scale} fill="url(#ae-mat)" stroke="hsl(var(--primary))" strokeWidth={0.8} />
      {/* web cutout */}
      {v.webRatio > 0.05 && (
        <rect x={x0 + w * 0.18} y={y0 + (h - webH) / 2} width={w * 0.64} height={webH * 0.5} rx={4} fill="hsl(var(--background))" stroke="hsl(var(--muted-foreground) / 0.4)" strokeWidth={0.4} strokeDasharray="2 2" />
      )}
      {/* mount holes */}
      <circle cx={x0 + w * 0.12} cy={y0 + h / 2} r={v.holeR * scale} fill="hsl(var(--background))" stroke="hsl(var(--accent))" strokeWidth={0.6} />
      <circle cx={x0 + w * 0.88} cy={y0 + h / 2} r={v.holeR * scale} fill="hsl(var(--background))" stroke="hsl(var(--accent))" strokeWidth={0.6} />
      {/* load arrow */}
      <g stroke="hsl(var(--destructive))" strokeWidth={1} fill="none">
        <line x1={x0 + w / 2} y1={y0 - 14} x2={x0 + w / 2} y2={y0 - 2} />
        <polyline points={`${x0 + w/2 - 4},${y0 - 6} ${x0 + w/2},${y0 - 2} ${x0 + w/2 + 4},${y0 - 6}`} />
      </g>
      <text x={4} y={12} fontSize="8" fill="hsl(var(--muted-foreground))">
        {v.material} · {v.length.toFixed(0)}×{v.width.toFixed(0)}×{v.thickness.toFixed(1)}mm · web {(v.webRatio * 100).toFixed(0)}%
      </text>
    </svg>
  );
}

function ParetoChart({ pareto, pop }: { pareto: Candidate[]; pop: Candidate[] }) {
  const W = 320, H = 180, pad = 20;
  const data = pop.filter((c) => c.feasible);
  if (data.length === 0) return null;
  const xs = data.map((c) => c.obj.mass);
  const ys = data.map((c) => c.obj.cost);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const X = (x: number) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - 2 * pad);
  const Y = (y: number) => H - pad - ((y - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full aspect-[16/9] rounded-md border border-border bg-background/40">
      <text x={pad} y={12} fontSize="8" fill="hsl(var(--muted-foreground))">mass → / cost ↑ · feasible</text>
      {data.map((c) => (
        <circle key={c.id} cx={X(c.obj.mass)} cy={Y(c.obj.cost)} r={1.8}
          fill={c.rank === 0 ? "hsl(var(--primary))" : "hsl(var(--muted-foreground) / 0.4)"} />
      ))}
      {pareto.filter(c => c.feasible).sort((a, b) => a.obj.mass - b.obj.mass).map((c, i, arr) => i === 0 ? null : (
        <line key={c.id} x1={X(arr[i-1].obj.mass)} y1={Y(arr[i-1].obj.cost)}
              x2={X(c.obj.mass)} y2={Y(c.obj.cost)} stroke="hsl(var(--primary))" strokeWidth={0.6} strokeDasharray="2 2" />
      ))}
    </svg>
  );
}

function HistoryChart({ history }: { history: RunResult["history"] }) {
  if (history.length < 2) return null;
  const W = 320, H = 90, pad = 14;
  const ys = history.map((h) => isFinite(h.bestScore) ? h.bestScore : 0);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const X = (i: number) => pad + (i / (history.length - 1)) * (W - 2 * pad);
  const Y = (y: number) => H - pad - ((y - ymin) / (ymax - ymin || 1)) * (H - 2 * pad);
  const d = history.map((h, i) => `${i === 0 ? "M" : "L"}${X(i)},${Y(ys[i])}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full aspect-[16/4.5] rounded-md border border-border bg-background/40">
      <text x={pad} y={10} fontSize="8" fill="hsl(var(--muted-foreground))">best composite score / generation</text>
      <path d={d} fill="none" stroke="hsl(var(--primary))" strokeWidth={1.2} />
    </svg>
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

export function AutoEngineeringPanel() {
  const [generations, setGenerations] = useState(20);
  const [popSize, setPopSize] = useState(40);
  const [load, setLoad] = useState(800);
  const [surrogateFilter, setSurrogateFilter] = useState(true);
  const [seed, setSeed] = useState(7);
  const [result, setResult] = useState<RunResult | null>(null);
  const [selected, setSelected] = useState<Candidate | null>(null);

  const run = () => {
    const r = optimize({ generations, popSize, seed, constraints: defaultConstraints(), load, surrogateFilter });
    setResult(r);
    setSelected(r.paretoFront[0] ?? r.population[0] ?? null);
  };

  const m = result?.metrics;
  const gateA = (m?.workflowAccelPct ?? 0) >= 40;
  const gateB = (m?.fabOptGainPct ?? 0) >= 25;
  const gateC = (m?.prototypeReductionPct ?? 0) >= 30;
  const allPass = gateA && gateB && gateC;

  const sortedPareto = useMemo(() => {
    if (!result) return [];
    return [...result.paretoFront].filter((c) => c.feasible).sort((a, b) => a.score - b.score).slice(0, 8);
  }, [result]);

  return (
    <div className="rounded-xl border border-border bg-card p-6 backdrop-blur-sm space-y-6">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="text-[10px] uppercase tracking-[0.32em] text-muted-foreground">Autonomous Engineering</div>
          <h2 className="font-display text-2xl mt-1">Generative, physics-constrained design search</h2>
          <p className="text-xs text-muted-foreground mt-1 max-w-xl">
            NSGA-II multi-objective optimizer over a parametric bracket. Closed-form physics
            (Euler–Bernoulli + Kt) drives fitness; an online surrogate skips obvious losers to
            cut prototype iteration count.
          </p>
        </div>
        <div className={`px-3 py-1.5 rounded-md text-[10px] uppercase tracking-[0.22em] ${allPass ? "bg-primary/15 text-primary" : "bg-muted/30 text-muted-foreground"}`}>
          {result ? (allPass ? "all targets met" : "running") : "idle"}
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">generations</span>
            <span className="text-primary tabular-nums">{generations}</span>
          </div>
          <Slider value={[generations]} min={5} max={60} step={1} onValueChange={([v]) => setGenerations(v)} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">population</span>
            <span className="text-primary tabular-nums">{popSize}</span>
          </div>
          <Slider value={[popSize]} min={16} max={120} step={4} onValueChange={([v]) => setPopSize(v)} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">tip load</span>
            <span className="text-primary tabular-nums">{load} N</span>
          </div>
          <Slider value={[load]} min={100} max={3000} step={50} onValueChange={([v]) => setLoad(v)} />
        </div>
        <div className="space-y-1.5">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.22em]">
            <span className="text-muted-foreground">seed</span>
            <span className="text-primary tabular-nums">{seed}</span>
          </div>
          <Slider value={[seed]} min={1} max={999} step={1} onValueChange={([v]) => setSeed(v)} />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 text-[10px] uppercase tracking-[0.18em]">
        <Button size="sm" onClick={run}>run search</Button>
        <Button size="sm" variant={surrogateFilter ? "default" : "outline"} onClick={() => setSurrogateFilter((v) => !v)}>
          surrogate filter {surrogateFilter ? "on" : "off"}
        </Button>
        {result && (
          <span className="ml-auto text-muted-foreground">
            {result.metrics.evaluations} evaluations · {result.surrogate.skipped} skipped · {result.metrics.elapsedMs.toFixed(0)}ms
          </span>
        )}
      </div>

      {result && m && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
            <Metric label="workflow accel" value={m.workflowAccelPct.toFixed(1)} unit="%" good={gateA} />
            <Metric label="fab opt gain" value={m.fabOptGainPct.toFixed(1)} unit="%" good={gateB} />
            <Metric label="prototype reduction" value={m.prototypeReductionPct.toFixed(1)} unit="%" good={gateC} />
            <Metric label="pareto size" value={m.paretoSize.toString()} />
            <Metric label="feasible" value={(m.feasibleRatio * 100).toFixed(0)} unit="%" />
            <Metric label="best score" value={isFinite(m.bestKnownScore) ? m.bestKnownScore.toFixed(3) : "—"} />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-[1.2fr,1fr] gap-4">
            <div className="space-y-3">
              <ParetoChart pareto={result.paretoFront} pop={result.population} />
              <HistoryChart history={result.history} />
            </div>
            <div className="space-y-3">
              <CADPreview c={selected} />
              {selected && (
                <div className="rounded-md border border-border bg-background/40 p-3 text-[11px] font-mono space-y-0.5">
                  <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">selected candidate · {selected.id}</div>
                  <div>mass: <span className="text-primary">{selected.obj.mass.toFixed(3)} kg</span></div>
                  <div>σ_max: <span className="text-primary">{selected.obj.stressMax.toFixed(1)} MPa</span></div>
                  <div>δ: <span className="text-primary">{selected.obj.deflection.toFixed(3)} mm</span></div>
                  <div>cost: <span className="text-primary">${selected.obj.cost.toFixed(2)}</span></div>
                  <div>energy: <span className="text-primary">{selected.obj.energyMJ.toFixed(1)} MJ</span></div>
                  <div>fab: <span className="text-primary">{selected.obj.fabMinutes.toFixed(1)} min</span></div>
                  {!selected.feasible && (
                    <div className="text-destructive">infeasible: {selected.violations.join(", ")}</div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div>
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">Pareto front · top 8</div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-2">
              {sortedPareto.map((c) => (
                <button
                  key={c.id}
                  onClick={() => setSelected(c)}
                  className={`text-left rounded-md border p-2 transition hover:border-primary/60 ${selected?.id === c.id ? "border-primary bg-primary/5" : "border-border bg-background/30"}`}
                >
                  <div className="flex justify-between text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
                    <span>{c.vars.material}</span>
                    <span>r{c.rank}</span>
                  </div>
                  <div className="mt-1 font-mono text-[10px] space-y-0.5">
                    <div>m {c.obj.mass.toFixed(3)}kg · ${c.obj.cost.toFixed(1)}</div>
                    <div>σ {c.obj.stressMax.toFixed(0)}MPa · δ {c.obj.deflection.toFixed(2)}mm</div>
                    <div className="text-muted-foreground">{c.obj.fabMinutes.toFixed(1)}min · {c.obj.energyMJ.toFixed(0)}MJ</div>
                  </div>
                </button>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-2 text-[10px] uppercase tracking-[0.18em]">
            <div className={`rounded-md border px-3 py-2 ${gateA ? "border-primary/40 bg-primary/5 text-primary" : "border-border text-muted-foreground"}`}>
              <div>≥40% workflow accel</div>
              <div className="text-foreground/70 mt-0.5 font-mono normal-case tracking-normal">{m.workflowAccelPct.toFixed(1)}%</div>
            </div>
            <div className={`rounded-md border px-3 py-2 ${gateB ? "border-primary/40 bg-primary/5 text-primary" : "border-border text-muted-foreground"}`}>
              <div>≥25% fab optimization</div>
              <div className="text-foreground/70 mt-0.5 font-mono normal-case tracking-normal">{m.fabOptGainPct.toFixed(1)}%</div>
            </div>
            <div className={`rounded-md border px-3 py-2 ${gateC ? "border-primary/40 bg-primary/5 text-primary" : "border-border text-muted-foreground"}`}>
              <div>≥30% prototype reduction</div>
              <div className="text-foreground/70 mt-0.5 font-mono normal-case tracking-normal">{m.prototypeReductionPct.toFixed(1)}%</div>
            </div>
          </div>
        </>
      )}

      {!result && (
        <div className="text-center text-xs text-muted-foreground py-6">
          Press <span className="text-primary">run search</span> to generate physics-constrained designs.
        </div>
      )}
    </div>
  );
}
