/**
 * Simulation Optimization panel.
 *
 * Synthesizes a stream of FEM / CFD / thermal / structural / fluid
 * sim requests, routes each through SimulationOptimizationEngine
 * (cache → ROM → full solve), and renders live stats:
 *
 *   - compute saved (ms + %)
 *   - cache / ROM / full breakdown
 *   - convergence speedup vs baseline iters
 *   - stability score (rolling success rate)
 *   - mesh quality histogram + nearest-neighbor preview
 *
 * The synthetic ticker simulates locality: many requests fall on a
 * handful of "scenarios" so the cache + ROM stages actually fire.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  SimulationOptimizationEngine,
  evalMeshQuality,
  type RouteDecision,
  type SimDescriptor,
  type SimOptStats,
  type SolverKind,
} from "@/lib/simopt";

const SCENARIOS: Array<{
  id: string;
  domain: SimDescriptor["domain"];
  regime: string;
  ndof: number;
  spd: boolean;
  logKappa: number;
  nonlinearity: number;
}> = [
  { id: "wing-loadcase-A", domain: "structural", regime: "linear-elastic-steel",  ndof:  120_000, spd: true,  logKappa: 5.2, nonlinearity: 0.05 },
  { id: "manifold-thermal", domain: "thermal",  regime: "steady-conduction-Al",   ndof:   40_000, spd: true,  logKappa: 4.0, nonlinearity: 0.02 },
  { id: "channel-flow-Re200", domain: "cfd",    regime: "navier-stokes-Re200",    ndof:   80_000, spd: false, logKappa: 6.4, nonlinearity: 0.55 },
  { id: "gear-contact",      domain: "fem",     regime: "frictional-contact",     ndof:  220_000, spd: false, logKappa: 7.1, nonlinearity: 0.75 },
  { id: "tank-sloshing",     domain: "fluid",   regime: "free-surface",           ndof:  150_000, spd: false, logKappa: 6.9, nonlinearity: 0.62 },
  { id: "micro-pcb-thermal", domain: "thermal", regime: "transient-Cu",           ndof:    9_500, spd: true,  logKappa: 3.6, nonlinearity: 0.08 },
];

const solverColor: Record<SolverKind, string> = {
  cg:       "text-sky-300 border-sky-500/40 bg-sky-500/10",
  bicgstab: "text-violet-300 border-violet-500/40 bg-violet-500/10",
  gmres:    "text-amber-300 border-amber-500/40 bg-amber-500/10",
  amg:      "text-emerald-300 border-emerald-500/40 bg-emerald-500/10",
  direct:   "text-foreground border-border bg-background/40",
  rom:      "text-pink-300 border-pink-500/40 bg-pink-500/10",
};

const stageColor: Record<RouteDecision["stage"], string> = {
  cache: "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  rom:   "border-pink-500/40 bg-pink-500/10 text-pink-300",
  full:  "border-amber-500/40 bg-amber-500/10 text-amber-300",
};

function jitter(x: number, frac: number) {
  return x * (1 + (Math.random() - 0.5) * 2 * frac);
}

function makeDescriptor(scenarioIdx: number, step: number): SimDescriptor {
  const s = SCENARIOS[scenarioIdx];
  // Small per-request perturbation simulates load-step / param sweep variations.
  const sizeJit = jitter(s.ndof, 0.04);
  const aspect = 1.4 + Math.random() * 1.3;
  const skew = Math.min(0.95, 0.18 + Math.random() * 0.3);
  return {
    id: `${s.id}#${step}`,
    domain: s.domain, regime: s.regime,
    ndof: Math.round(sizeJit),
    density: 0.0008 + Math.random() * 0.0004,
    logKappa: jitter(s.logKappa, 0.06),
    spd: s.spd,
    nonlinearity: Math.min(1, Math.max(0, s.nonlinearity + (Math.random() - 0.5) * 0.1)),
    mesh: {
      nElems: Math.round(sizeJit * 5),
      nNodes: Math.round(sizeJit * 0.9),
      aspectMax: aspect * (1.6 + Math.random()),
      aspectMean: aspect,
      minAngle: 0.35 + Math.random() * 0.4,
      maxAngle: 1.7 + Math.random() * 0.4,
      jacobianMin: 0.05 + Math.random() * 0.6,
      skewness: skew,
    },
    bcFingerprint: [scenarioIdx, step % 3, s.spd ? 1 : 0, 0, 0, 0],
    step,
  };
}

/** Toy solver runner — synthesizes iters/elapsed based on stage + size. */
function runSolveSim(d: SimDescriptor, dec: RouteDecision) {
  const baseIters = Math.round(40 + Math.log2(d.ndof + 1) * (3 + d.logKappa));
  let iters: number, elapsedMs: number, converged = true;
  if (dec.stage === "cache") {
    iters = 0; elapsedMs = 0.4 + Math.random() * 0.6;
  } else if (dec.stage === "rom") {
    iters = Math.round(baseIters * 0.18);
    elapsedMs = iters * 0.05 + 1.2;
  } else {
    // Routed full solve: better solver picks → fewer iters.
    const factor = dec.solver === "amg" ? 0.45
                 : dec.solver === "cg" ? 0.7
                 : dec.solver === "direct" ? 0.6
                 : dec.solver === "bicgstab" ? 0.95
                 : dec.solver === "gmres" ? 0.85 : 1;
    iters = Math.round(baseIters * factor);
    elapsedMs = iters * 0.18 + d.ndof * 1.5e-4;
    converged = Math.random() > 0.04; // occasional non-convergence
  }
  return {
    residual: converged ? 1e-7 : 1e-2,
    iters, elapsedMs, solver: dec.solver, converged,
    signature: new Float64Array([d.ndof, d.logKappa, iters, elapsedMs]),
  };
}

export function SimOptPanel() {
  const engineRef = useRef<SimulationOptimizationEngine | null>(null);
  if (!engineRef.current) {
    engineRef.current = new SimulationOptimizationEngine({
      latentDim: 6, cacheRadius: 0.55, romRadius: 1.4,
    });
  }
  const engine = engineRef.current;

  const [running, setRunning] = useState(true);
  const [stats, setStats] = useState<SimOptStats>(() => engine.snapshot());
  const [recent, setRecent] = useState<Array<{
    desc: SimDescriptor; dec: RouteDecision; iters: number; ms: number; ok: boolean;
  }>>([]);
  const stepRef = useRef(0);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      // Burst of 4 requests / tick — bias toward repeats so caching matters.
      for (let k = 0; k < 4; k++) {
        const idx = Math.random() < 0.65
          ? Math.floor(Math.random() * 3) // hot path: first three scenarios
          : Math.floor(Math.random() * SCENARIOS.length);
        const step = stepRef.current++;
        const desc = makeDescriptor(idx, step);
        const { decision, result } = engine.request(desc, runSolveSim);
        setRecent((prev) => {
          const next = [{ desc, dec: decision, iters: result.iters, ms: result.elapsedMs, ok: result.converged }, ...prev];
          if (next.length > 8) next.length = 8;
          return next;
        });
      }
      setStats(engine.snapshot());
    }, 350);
    return () => clearInterval(id);
  }, [engine, running]);

  const reset = () => {
    engine.reset();
    setStats(engine.snapshot());
    setRecent([]);
    stepRef.current = 0;
  };

  const savedPct = stats.estComputeFullMs > 0
    ? (stats.computeSavedMs / stats.estComputeFullMs) * 100 : 0;

  const cacheEntries = engine.cacheEntries();
  const cacheHitsByEntry = useMemo(
    () => [...cacheEntries].sort((a, b) => b.hits - a.hits).slice(0, 4),
    [cacheEntries, stats.totalRequests],
  );

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          SimOpt · simulation optimization engine
          <Badge variant="outline" className="text-[10px]">{cacheEntries.length} cached</Badge>
        </h3>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setRunning((r) => !r)}>
            {running ? "pause" : "resume"}
          </Button>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={reset}>
            reset
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Tile label="compute saved" value={`${savedPct.toFixed(1)}%`}
          sub={`${(stats.computeSavedMs / 1000).toFixed(2)}s avoided`}
          tone={savedPct >= 30 ? "good" : "neutral"} />
        <Tile label="conv. speedup" value={`${stats.convergenceSpeedup.toFixed(2)}×`}
          sub="vs full baseline" tone={stats.convergenceSpeedup >= 1.2 ? "good" : "neutral"} />
        <Tile label="stability" value={`${(stats.stabilityScore * 100).toFixed(0)}%`}
          sub="rolling convergence" tone={stats.stabilityScore >= 0.95 ? "good" : "warn"} />
        <Tile label="requests" value={`${stats.totalRequests}`}
          sub={`${stats.cacheHits} cache · ${stats.romHits} rom · ${stats.fullSolves} full`} />
      </div>

      <div>
        <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          recent routing decisions
        </div>
        <div className="space-y-1">
          {recent.map((r) => {
            const mq = evalMeshQuality(r.desc.mesh);
            return (
              <div key={r.desc.id}
                className="grid grid-cols-[1fr_auto_auto_auto_auto] items-center gap-2 rounded border border-border/60 bg-background/40 px-2 py-1.5 text-[11px]">
                <div className="min-w-0">
                  <div className="truncate font-medium">{r.desc.id}</div>
                  <div className="truncate text-[10px] text-muted-foreground">{r.dec.reason}</div>
                </div>
                <span className={`rounded border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.14em] ${stageColor[r.dec.stage]}`}>
                  {r.dec.stage}
                </span>
                <span className={`rounded border px-1.5 py-0.5 text-[9px] uppercase tracking-[0.14em] ${solverColor[r.dec.solver]}`}>
                  {r.dec.solver}
                </span>
                <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
                  {r.iters} it · {r.ms.toFixed(1)}ms
                </span>
                <span className="font-mono text-[10px] tabular-nums text-muted-foreground"
                  title={`mesh quality ${mq.score.toFixed(2)} · instability ${mq.instability.toFixed(2)}`}>
                  mq {mq.score.toFixed(2)}
                </span>
              </div>
            );
          })}
          {recent.length === 0 && (
            <div className="rounded border border-border/60 bg-background/40 p-3 text-[11px] italic text-muted-foreground">
              warming up…
            </div>
          )}
        </div>
      </div>

      <div>
        <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          hottest cached simulations
        </div>
        <div className="grid grid-cols-1 gap-1 sm:grid-cols-2">
          {cacheHitsByEntry.map((c) => (
            <div key={c.desc.id}
              className="flex items-center justify-between rounded border border-border/60 bg-background/40 px-2 py-1 text-[11px]">
              <span className="min-w-0 truncate">
                <span className="font-medium">{c.desc.regime}</span>{" "}
                <span className="text-[10px] text-muted-foreground">{c.desc.domain} · {c.desc.ndof.toLocaleString()} dof</span>
              </span>
              <span className="font-mono text-[10px] tabular-nums text-emerald-300">{c.hits} hits</span>
            </div>
          ))}
          {cacheHitsByEntry.length === 0 && (
            <div className="rounded border border-border/60 bg-background/40 p-3 text-[11px] italic text-muted-foreground">
              cache empty
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Tile({ label, value, sub, tone = "neutral" }: {
  label: string; value: string; sub?: string;
  tone?: "good" | "warn" | "neutral";
}) {
  const cls = tone === "good"
    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
    : tone === "warn"
      ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
      : "border-border bg-background/40";
  return (
    <div className={`rounded border px-3 py-2 ${cls}`}>
      <div className="text-[9px] uppercase tracking-[0.16em] opacity-80">{label}</div>
      <div className="font-mono text-lg tabular-nums">{value}</div>
      {sub && <div className="text-[10px] opacity-70">{sub}</div>}
    </div>
  );
}
