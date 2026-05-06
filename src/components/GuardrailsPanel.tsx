import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  createGuard, ingest, syntheticStep, releaseIsolation,
  DEFAULT_THRESHOLDS, type GuardState, type GuardEvent, type GuardThresholds,
} from "@/lib/guardrails";

type Fault = "off" | "drift" | "nan" | "blowup" | "partition";
const FAULTS: { key: Fault; label: string }[] = [
  { key: "off",       label: "nominal" },
  { key: "drift",     label: "energy drift" },
  { key: "nan",       label: "NaN spike" },
  { key: "blowup",    label: "global blowup" },
  { key: "partition", label: "rogue partition" },
];

export function GuardrailsPanel() {
  const [thresh, setThresh] = useState<GuardThresholds>(DEFAULT_THRESHOLDS);
  const [fault, setFault] = useState<Fault>("off");
  const [running, setRunning] = useState(false);
  const stateRef = useRef<GuardState>(createGuard(thresh));
  const tRef = useRef(0);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    stateRef.current.thresholds = thresh;
  }, [thresh]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      tRef.current += stateRef.current.dt;
      const snap = syntheticStep(stateRef.current, tRef.current, fault);
      ingest(stateRef.current, snap);
      setTick((t) => t + 1);
    }, 33);
    return () => window.clearInterval(id);
  }, [running, fault]);

  const reset = () => {
    stateRef.current = createGuard(thresh);
    tRef.current = 0;
    setTick((t) => t + 1);
  };

  const g = stateRef.current;
  const last = g.history[g.history.length - 1];
  const driftPct = last
    ? Math.abs(last.energy - g.baselineEnergy) / Math.max(g.baselineEnergy, 1e-9) * 100
    : 0;

  const energyPath = useMemo(() => {
    if (g.history.length < 2) return null;
    const W = 480, H = 70;
    const tMin = g.history[0].t, tMax = g.history[g.history.length - 1].t;
    const span = Math.max(1e-3, tMax - tMin);
    const vals = g.history.map((s) => s.energy).filter(Number.isFinite);
    const lo = Math.min(...vals, g.baselineEnergy);
    const hi = Math.max(...vals, g.baselineEnergy);
    const ys = Math.max(1e-3, hi - lo);
    const path = g.history.map((s, i) => {
      const x = ((s.t - tMin) / span) * W;
      const v = Number.isFinite(s.energy) ? s.energy : lo;
      const y = H - ((v - lo) / ys) * (H - 6) - 3;
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const baseY = H - ((g.baselineEnergy - lo) / ys) * (H - 6) - 3;
    return { W, H, path, baseY };
  }, [tick, g.baselineEnergy, g.history]);

  const recent = [...g.events].reverse().slice(0, 8);
  const sevColor = (s: GuardEvent["severity"]) =>
    s === "critical" ? "text-destructive" : s === "warn" ? "text-accent" : "text-primary";

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            stability · guardrails
          </div>
          <h2 className="font-display text-2xl text-foreground">
            Autonomous <span className="text-primary">numerical safety</span> layer.
          </h2>
        </div>
        <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.18em]">
          <span className={g.enabled ? "text-primary" : "text-muted-foreground"}>
            <span className={`inline-block mr-1.5 h-1.5 w-1.5 rounded-full ${g.enabled ? "bg-primary animate-pulse" : "bg-muted-foreground"}`} />
            {g.enabled ? "armed" : "disarmed"}
          </span>
          <span className="text-muted-foreground">dt · </span>
          <span className="text-foreground tabular-nums">{(g.dt * 1000).toFixed(2)}ms</span>
          <span className="text-muted-foreground">drift · </span>
          <span className={driftPct > thresh.energyDriftPct ? "text-destructive" : "text-foreground"}>
            {driftPct.toFixed(2)}%
          </span>
        </div>
      </div>

      {/* Fault injection */}
      <div className="flex flex-wrap gap-2">
        {FAULTS.map((f) => (
          <button
            key={f.key}
            onClick={() => setFault(f.key)}
            className={`text-[10px] uppercase tracking-[0.18em] px-2.5 py-1 rounded border transition ${
              fault === f.key
                ? "border-secondary bg-secondary/15 text-foreground"
                : "border-border bg-background/30 text-muted-foreground hover:text-foreground"
            }`}
          >
            inject · {f.label}
          </button>
        ))}
      </div>

      {/* Energy plot */}
      <div className="rounded-md border border-border bg-background/30 p-3">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
          ΣE(t) · baseline reference
        </div>
        {energyPath ? (
          <svg viewBox={`0 0 ${energyPath.W} ${energyPath.H}`} width="100%" height={energyPath.H} className="block">
            <line x1={0} y1={energyPath.baseY} x2={energyPath.W} y2={energyPath.baseY}
              stroke="currentColor" className="text-muted-foreground/40" strokeDasharray="3 3" />
            <path d={energyPath.path} fill="none" stroke="currentColor"
              className={driftPct > thresh.energyDriftPct ? "text-destructive" : "text-primary"}
              strokeWidth={1.5} />
          </svg>
        ) : (
          <div className="h-[70px] flex items-center justify-center text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            press play to start monitoring
          </div>
        )}
      </div>

      {/* Partitions */}
      <div className="rounded-md border border-border bg-background/30 p-3">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
          partitions
        </div>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
          {(last?.partitions ?? Array(6).fill(0)).map((e, i) => {
            const isolated = g.isolated.has(i);
            const bad = !Number.isFinite(e);
            return (
              <button
                key={i}
                onClick={() => isolated && releaseIsolation(g, i)}
                disabled={!isolated}
                className={`rounded border px-2 py-1.5 text-left transition ${
                  bad ? "border-destructive/60 bg-destructive/10"
                  : isolated ? "border-secondary/60 bg-secondary/10"
                  : "border-border bg-background/40"
                } ${isolated ? "hover:bg-secondary/20 cursor-pointer" : "cursor-default"}`}
                title={isolated ? "click to re-join" : ""}
              >
                <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">p{i}</div>
                <div className={`font-mono text-[11px] tabular-nums ${
                  bad ? "text-destructive" : isolated ? "text-secondary" : "text-foreground/90"
                }`}>
                  {bad ? "NaN" : e.toFixed(3)}
                </div>
                <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground/70">
                  {bad ? "fault" : isolated ? "isolated" : "active"}
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Thresholds */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Knob label="Energy drift cap" unit="%" value={thresh.energyDriftPct}
          min={1} max={50} step={1}
          onChange={(v) => setThresh((t) => ({ ...t, energyDriftPct: v }))} />
        <Knob label="Constraint tol" value={thresh.constraintTol}
          min={0.001} max={0.2} step={0.001}
          onChange={(v) => setThresh((t) => ({ ...t, constraintTol: v }))} />
        <Knob label="|v| cap" value={thresh.velocityCap}
          min={100} max={3000} step={50}
          onChange={(v) => setThresh((t) => ({ ...t, velocityCap: v }))} />
        <Knob label="Partition variance" value={thresh.partitionVar}
          min={0.05} max={1} step={0.05}
          onChange={(v) => setThresh((t) => ({ ...t, partitionVar: v }))} />
      </div>

      {/* Counters */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
        <Stat label="rollbacks" value={String(g.rollbacks)} />
        <Stat label="dt shrinks" value={String(g.shrinks)} />
        <Stat label="isolated" value={String(g.isolated.size)} />
        <Stat label="events" value={String(g.events.length)} />
      </div>

      {/* Event log */}
      <div className="rounded-md border border-border bg-background/30 p-3">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
          guard log
        </div>
        {recent.length === 0 ? (
          <div className="text-[10px] text-muted-foreground/70 font-mono">no events</div>
        ) : (
          <ul className="space-y-1 font-mono text-[10px] max-h-[180px] overflow-auto">
            {recent.map((e, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className={`shrink-0 ${sevColor(e.severity)}`}>
                  [{e.severity.padEnd(8)}]
                </span>
                <span className="text-muted-foreground">t={e.t.toFixed(2)}</span>
                <span className="text-foreground/90">{e.guard}{e.partition !== undefined ? `·p${e.partition}` : ""}</span>
                {e.mitigation && (
                  <span className="text-accent/90">→ {e.mitigation.kind}: {e.mitigation.detail}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="flex gap-2">
        <Button onClick={() => setRunning((r) => !r)} className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "pause" : "play"}
        </Button>
        <Button variant="outline" onClick={reset}
          className="uppercase tracking-[0.18em] text-[10px]">reset</Button>
        <Button variant="outline"
          onClick={() => { g.enabled = !g.enabled; setTick((t) => t + 1); }}
          className="uppercase tracking-[0.18em] text-[10px]">
          {g.enabled ? "disarm" : "arm"}
        </Button>
      </div>
    </div>
  );
}

function Knob({ label, value, unit, min, max, step, onChange }: {
  label: string; value: number; unit?: string;
  min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-[10px] uppercase tracking-[0.18em]">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-primary tabular-nums">
          {value.toFixed(step < 1 ? 3 : 0)}
          {unit && <span className="text-muted-foreground ml-1">{unit}</span>}
        </span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => onChange(v)} />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border/60 px-2 py-1">
      <div className="uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className="font-mono text-foreground/90 tabular-nums">{value}</div>
    </div>
  );
}
