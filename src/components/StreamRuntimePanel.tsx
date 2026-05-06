import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  createStream, tickStream, forecast,
  DEFAULT_STREAM_CFG, type StreamConfig, type StreamState,
} from "@/lib/streamRuntime";

export function StreamRuntimePanel() {
  const [cfg, setCfg] = useState<StreamConfig>(DEFAULT_STREAM_CFG);
  const [running, setRunning] = useState(false);
  const stateRef = useRef<StreamState>(createStream(cfg));
  const [tick, setTick] = useState(0);

  // recreate stream when cfg changes meaningfully
  useEffect(() => {
    stateRef.current = createStream(cfg);
    setTick((t) => t + 1);
  }, [cfg]);

  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => {
      tickStream(stateRef.current, performance.now());
      setTick((t) => t + 1);
    }, cfg.tickMs);
    return () => window.clearInterval(id);
  }, [running, cfg.tickMs]);

  const update = <K extends keyof StreamConfig>(k: K, v: StreamConfig[K]) =>
    setCfg((c) => ({ ...c, [k]: v }));

  const view = useMemo(() => {
    const s = stateRef.current;
    const samples = s.samples;
    const fc = forecast(s, cfg.forecastMs);
    if (samples.length < 2) return null;
    const W = 640, H = 160;
    const tMin = samples[0].t;
    const tMax = (fc[fc.length - 1]?.t ?? samples[samples.length - 1].t);
    const span = Math.max(1, tMax - tMin);
    const all = samples.flatMap((p) => [p.sensor, p.truth, p.fused])
      .concat(fc.map((p) => p.v));
    const lo = Math.min(...all), hi = Math.max(...all);
    const yspan = Math.max(1e-3, hi - lo);
    const x = (t: number) => ((t - tMin) / span) * W;
    const y = (v: number) => H - ((v - lo) / yspan) * (H - 8) - 4;
    const path = (vals: { t: number; v: number }[]) =>
      vals.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" ");
    return {
      W, H,
      truth:   path(samples.map((p) => ({ t: p.t, v: p.truth }))),
      sensor:  path(samples.map((p) => ({ t: p.t, v: p.sensor }))),
      fused:   path(samples.map((p) => ({ t: p.t, v: p.fused }))),
      forecast: path(fc),
      cutX: x(samples[samples.length - 1].t),
    };
  }, [tick, cfg.forecastMs]);

  const s = stateRef.current;
  const lastResidual = s.samples.length ? s.samples[s.samples.length - 1].residual : 0;

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            stream · runtime
          </div>
          <h2 className="font-display text-2xl text-foreground">
            Continuous <span className="text-primary">assimilation</span>, rolling window.
          </h2>
        </div>
        <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.18em]">
          <span className={running ? "text-primary" : "text-muted-foreground"}>
            <span className={`inline-block mr-1.5 h-1.5 w-1.5 rounded-full ${running ? "bg-primary animate-pulse" : "bg-muted-foreground"}`} />
            {running ? "live" : "paused"}
          </span>
          <span className="text-muted-foreground">lag · </span>
          <span className="text-foreground tabular-nums">{s.ingestLagMs.toFixed(1)} ms</span>
          <span className="text-muted-foreground">mae · </span>
          <span className="text-foreground tabular-nums">{s.mae.toFixed(3)}</span>
        </div>
      </div>

      {/* Plot */}
      <div className="rounded-md border border-border bg-background/30 p-3 overflow-hidden">
        {view ? (
          <svg viewBox={`0 0 ${view.W} ${view.H}`} width="100%" height={view.H} className="block">
            <line x1={view.cutX} y1={0} x2={view.cutX} y2={view.H}
              stroke="currentColor" className="text-border" strokeDasharray="3 3" />
            <path d={view.truth}    fill="none" stroke="currentColor"
              className="text-muted-foreground/40" strokeWidth={1} />
            <path d={view.sensor}   fill="none" stroke="currentColor"
              className="text-accent/60" strokeWidth={1} />
            <path d={view.fused}    fill="none" stroke="currentColor"
              className="text-primary" strokeWidth={1.5} />
            <path d={view.forecast} fill="none" stroke="currentColor"
              className="text-secondary" strokeWidth={1.5} strokeDasharray="2 3" />
          </svg>
        ) : (
          <div className="h-[160px] flex items-center justify-center text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            press play to start ingesting
          </div>
        )}
        <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[9px] uppercase tracking-[0.2em] text-muted-foreground">
          <span><span className="inline-block w-3 h-px align-middle bg-muted-foreground/40 mr-1" /> truth</span>
          <span><span className="inline-block w-3 h-px align-middle bg-accent/60 mr-1" /> sensor</span>
          <span><span className="inline-block w-3 h-px align-middle bg-primary mr-1" /> fused</span>
          <span><span className="inline-block w-3 h-px align-middle bg-secondary mr-1" /> forecast</span>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Knob label="Tick" unit="ms"  value={cfg.tickMs}    min={20}  max={250}  step={10}   onChange={(v) => update("tickMs", v)} />
        <Knob label="Window" unit="ms" value={cfg.windowMs} min={1000} max={20000} step={500} onChange={(v) => update("windowMs", v)} />
        <Knob label="Forecast" unit="ms" value={cfg.forecastMs} min={50} max={2000} step={50} onChange={(v) => update("forecastMs", v)} />
        <Knob label="Sensor σ" value={cfg.sensorNoise} min={0} max={0.5} step={0.01} onChange={(v) => update("sensorNoise", v)} />
        <Knob label="Assim. gain" value={cfg.assimGain} min={0} max={1} step={0.05} onChange={(v) => update("assimGain", v)} />
        <Knob label="Signal" unit="Hz" value={cfg.signalHz} min={0.1} max={3} step={0.05} onChange={(v) => update("signalHz", v)} />
      </div>

      {/* Bus */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
        <Stat label="samples" value={String(s.samples.length)} />
        <Stat label="residual" value={lastResidual.toFixed(3)} />
        <Stat label="rmse" value={s.rmse.toFixed(3)} />
        <Stat label="ingest hz" value={(1000 / cfg.tickMs).toFixed(1)} />
      </div>

      <div className="flex gap-2">
        <Button onClick={() => setRunning((r) => !r)} className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "pause" : "play"}
        </Button>
        <Button variant="outline" onClick={() => { stateRef.current = createStream(cfg); setTick((t) => t + 1); }}
          className="uppercase tracking-[0.18em] text-[10px]">
          reset
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
          {value.toFixed(step < 1 ? 2 : 0)}
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
