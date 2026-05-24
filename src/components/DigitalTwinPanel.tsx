/**
 * Digital Twin Metric Space — live panel.
 *
 * Renders three concurrent twins (factory, farm, fab line), each
 * sampled at 5 Hz with synthetic telemetry. Shows:
 *
 *   - latent phase-space mirror (PCA1 × PCA2)
 *   - distance-to-optimum + drift σ + efficiency surface
 *   - bottleneck localization (per-station load bars)
 *   - forecast distance trajectory (next 12 ticks)
 *   - multi-system synchronization (pairwise latent distance)
 *   - historical replay buffer
 *   - aggregate KPI deltas vs un-twinned baseline
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  DigitalTwinEngine, type TwinSpec, type TwinState, type TelemetryFrame,
} from "@/lib/twin";

// ---------------- twin specs + synthetic telemetry ----------------

const FACTORY: TwinSpec = {
  id: "twin-factory", name: "Assembly Line · Plant 7", kind: "factory",
  stations: ["intake", "press", "weld", "paint", "qa", "pack"],
  channels: ["throughput", "temp", "kWh"],
  optimal: {
    intake: { throughput: 120, temp: 22, kWh: 0.9 },
    press:  { throughput: 118, temp: 65, kWh: 3.2 },
    weld:   { throughput: 115, temp: 220, kWh: 4.1 },
    paint:  { throughput: 113, temp: 38, kWh: 2.4 },
    qa:     { throughput: 113, temp: 22, kWh: 0.4 },
    pack:   { throughput: 112, temp: 22, kWh: 0.6 },
  },
};

const FARM: TwinSpec = {
  id: "twin-farm", name: "Vertical Farm · Block B", kind: "farm",
  stations: ["zone-a", "zone-b", "zone-c", "irrigation", "hvac"],
  channels: ["temp", "humidity", "co2", "lux"],
  optimal: {
    "zone-a":    { temp: 22, humidity: 65, co2: 800, lux: 18000 },
    "zone-b":    { temp: 23, humidity: 62, co2: 820, lux: 17500 },
    "zone-c":    { temp: 22, humidity: 64, co2: 810, lux: 18200 },
    irrigation:  { temp: 18, humidity: 0,  co2: 0,   lux: 0 },
    hvac:        { temp: 22, humidity: 0,  co2: 0,   lux: 0 },
  },
};

const FAB: TwinSpec = {
  id: "twin-fab", name: "Fab Line · CNC + DMLS", kind: "fab_line",
  stations: ["cnc-1", "cnc-2", "dmls", "post", "inspect"],
  channels: ["throughput", "vibration", "kWh"],
  optimal: {
    "cnc-1":  { throughput: 8, vibration: 0.4, kWh: 5.2 },
    "cnc-2":  { throughput: 8, vibration: 0.4, kWh: 5.2 },
    dmls:     { throughput: 2, vibration: 0.1, kWh: 12 },
    post:     { throughput: 7, vibration: 0.2, kWh: 1.6 },
    inspect:  { throughput: 7, vibration: 0.0, kWh: 0.3 },
  },
};

const TWINS: TwinSpec[] = [FACTORY, FARM, FAB];

/** Simulated fault injector: ramps a single station's metrics off optimum. */
function genFrame(spec: TwinSpec, t: number, faultStation: string | null): TelemetryFrame {
  const stations: Record<string, Record<string, number>> = {};
  for (const st of spec.stations) {
    const opt = spec.optimal![st];
    const row: Record<string, number> = {};
    const isFault = st === faultStation;
    const drift = isFault ? (1 + Math.sin(t / 7) * 0.5 + 0.5) * 0.5 : 0;
    for (const ch of spec.channels) {
      const o = opt[ch];
      // Healthy: 2% noise around opt. Fault: 12–35% bias.
      const noise = (Math.random() - 0.5) * 0.04 * Math.abs(o);
      const biasFrac = ch === "throughput" ? -0.25 : ch === "vibration" ? 1.6
                     : ch === "kWh" ? 0.20 : 0.12;
      const bias = drift * biasFrac * Math.abs(o || 1);
      row[ch] = o + noise + bias;
    }
    stations[st] = row;
  }
  return { twinId: spec.id, t, stations };
}

export function DigitalTwinPanel() {
  const engineRef = useRef<DigitalTwinEngine | null>(null);
  if (!engineRef.current) {
    engineRef.current = new DigitalTwinEngine({ latentDim: 4, warmup: 16, forecastSteps: 12 });
    for (const t of TWINS) engineRef.current.register(t);
  }
  const engine = engineRef.current;

  const [states, setStates] = useState<Record<string, TwinState>>({});
  const [running, setRunning] = useState(true);
  const [replayMode, setReplayMode] = useState(false);
  const [selected, setSelected] = useState<string>(FACTORY.id);
  const [faults, setFaults] = useState<Record<string, string | null>>({});
  const tickRef = useRef(0);

  // KPI accumulators for deltas vs naive (un-twinned) baseline.
  const kpiRef = useRef({ efSum: 0, baseEfSum: 0, anomCaught: 0, anomBase: 0, energySaved: 0, n: 0 });
  const [kpi, setKpi] = useState({ eff: 0, anom: 0, energy: 0 });

  useEffect(() => {
    if (!running || replayMode) return;
    const id = setInterval(() => {
      const next: Record<string, TwinState> = {};
      for (const spec of TWINS) {
        tickRef.current++;
        const frame = genFrame(spec, tickRef.current, faults[spec.id] ?? null);
        const s = engine.ingest(frame);
        next[spec.id] = s;

        // Baseline (no twin) — naive threshold detector on raw distance.
        const baseEff = s.distance < 0.05 ? 1 : 0.7;
        const anomBase = s.distance > 0.5 ? 1 : 0;
        const anomTwin = s.anomalyProb > 0.5 ? 1 : 0;
        kpiRef.current.efSum += s.efficiency;
        kpiRef.current.baseEfSum += baseEff;
        kpiRef.current.anomBase += anomBase;
        kpiRef.current.anomCaught += anomTwin;
        kpiRef.current.energySaved += s.energyHeadroom;
        kpiRef.current.n += 1;
      }
      setStates(next);
      const k = kpiRef.current;
      if (k.n > 0) {
        setKpi({
          eff: (k.efSum - k.baseEfSum) / Math.max(1, k.baseEfSum),
          anom: (k.anomCaught - k.anomBase) / Math.max(1, k.anomBase),
          energy: k.energySaved / k.n,
        });
      }
    }, 200);
    return () => clearInterval(id);
  }, [engine, running, replayMode, faults]);

  const sel = states[selected];
  const spec = TWINS.find((t) => t.id === selected)!;

  // Multi-system synchronization matrix.
  const syncMatrix = useMemo(() => {
    const m: Array<Array<number | null>> = TWINS.map(() => TWINS.map(() => null));
    for (let i = 0; i < TWINS.length; i++) {
      for (let j = i + 1; j < TWINS.length; j++) {
        const r = engine.sync(TWINS[i].id, TWINS[j].id);
        if (r) { m[i][j] = r.distance; m[j][i] = r.distance; }
      }
    }
    return m;
  }, [engine, states]);

  const toggleFault = (twinId: string, station: string) => {
    setFaults((f) => ({ ...f, [twinId]: f[twinId] === station ? null : station }));
  };
  const reset = () => {
    engine.reset();
    for (const t of TWINS) engine.register(t);
    kpiRef.current = { efSum: 0, baseEfSum: 0, anomCaught: 0, anomBase: 0, energySaved: 0, n: 0 };
    setStates({}); setFaults({}); tickRef.current = 0;
    setKpi({ eff: 0, anom: 0, energy: 0 });
  };

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Digital Twin · Metric Space Layer
          <Badge variant="outline" className="text-[10px]">
            {Object.keys(states).length} twins · {tickRef.current} ticks
          </Badge>
        </h3>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setRunning((r) => !r)}>{running ? "pause" : "resume"}</Button>
          <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setReplayMode((r) => !r)}>
            {replayMode ? "live" : "replay"}
          </Button>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs"
            onClick={reset}>reset</Button>
        </div>
      </div>

      {/* KPI ribbon */}
      <div className="grid grid-cols-3 gap-2">
        <Kpi label="efficiency Δ vs baseline" v={kpi.eff} fmt={(x) => `${(x * 100).toFixed(1)}%`}
             target={0.20} ok={kpi.eff >= 0.20} />
        <Kpi label="anomaly detection gain" v={kpi.anom} fmt={(x) => `${(x * 100).toFixed(1)}%`}
             target={0.25} ok={kpi.anom >= 0.25} />
        <Kpi label="energy headroom" v={kpi.energy} fmt={(x) => `${(x * 100).toFixed(1)}%`}
             target={0.15} ok={kpi.energy >= 0.15} />
      </div>

      {/* Twin selector */}
      <div className="flex flex-wrap gap-1">
        {TWINS.map((t) => (
          <button key={t.id} onClick={() => setSelected(t.id)}
            className={`text-[10px] uppercase tracking-[0.14em] px-2 py-1 rounded border ${
              selected === t.id
                ? "border-primary bg-primary/10 text-primary"
                : "border-border text-muted-foreground hover:text-foreground"
            }`}>
            {t.name}
          </button>
        ))}
      </div>

      {sel && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          {/* Latent phase-space + forecast */}
          <PhaseSpace
            spec={spec}
            engine={engine}
            state={sel}
            replay={replayMode}
          />

          {/* Right column: metrics + bottlenecks */}
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 text-[10px] tabular-nums">
              <Metric label="distance to optimum" v={sel.distance.toFixed(3)} />
              <Metric label="drift σ" v={sel.driftSigma.toFixed(2)} />
              <Metric label="efficiency" v={`${(sel.efficiency * 100).toFixed(1)}%`} />
              <Metric label="anomaly prob" v={`${(sel.anomalyProb * 100).toFixed(1)}%`} />
              <Metric label="energy headroom" v={`${(sel.energyHeadroom * 100).toFixed(1)}%`} />
              <Metric label="bottleneck" v={sel.bottleneckStation ?? "—"} />
            </div>

            <div className="rounded border border-border bg-background/40 p-2">
              <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-1">
                Bottleneck localization · click to inject fault
              </div>
              <div className="space-y-1">
                {spec.stations.map((st) => {
                  const v = sel.stationLoad[st] ?? 0;
                  const active = faults[spec.id] === st;
                  const isBottleneck = sel.bottleneckStation === st;
                  return (
                    <button key={st} onClick={() => toggleFault(spec.id, st)}
                      className="w-full text-left">
                      <div className="flex items-center justify-between text-[10px]">
                        <span className={`font-mono ${active ? "text-destructive" : "text-foreground/80"}`}>
                          {st}{active && " · FAULT"}
                        </span>
                        <span className="tabular-nums text-muted-foreground">
                          {(v * 100).toFixed(0)}%
                        </span>
                      </div>
                      <div className="h-1 w-full overflow-hidden rounded bg-background">
                        <div className={`h-full transition-[width] ${
                          isBottleneck ? "bg-destructive" : active ? "bg-amber-400" : "bg-primary/70"
                        }`}
                          style={{ width: `${Math.max(2, Math.round(v * 100))}%` }} />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>

            <ForecastChart state={sel} />
          </div>
        </div>
      )}

      {/* Multi-system synchronization */}
      <div className="rounded border border-border bg-background/40 p-3">
        <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
          <span>Multi-system synchronization · pairwise latent distance</span>
          <span className="opacity-70">aligned when d &lt; 1.5</span>
        </div>
        <div className="overflow-x-auto">
          <table className="text-[10px] tabular-nums">
            <thead className="text-muted-foreground">
              <tr><th className="text-left font-normal pr-3"></th>
                {TWINS.map((t) => <th key={t.id} className="px-2 font-normal">{t.id.replace("twin-", "")}</th>)}
              </tr>
            </thead>
            <tbody className="font-mono">
              {TWINS.map((row, i) => (
                <tr key={row.id}>
                  <td className="pr-3 text-muted-foreground">{row.id.replace("twin-", "")}</td>
                  {TWINS.map((_, j) => {
                    const v = syncMatrix[i][j];
                    if (i === j) return <td key={j} className="px-2 text-muted-foreground/40">—</td>;
                    if (v == null) return <td key={j} className="px-2 text-muted-foreground/40">·</td>;
                    return (
                      <td key={j} className={`px-2 ${v < 1.5 ? "text-emerald-300" : "text-amber-300"}`}>
                        {v.toFixed(2)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

// ---------------- subcomponents ----------------

function Kpi({
  label, v, fmt, target, ok,
}: { label: string; v: number; fmt: (x: number) => string; target: number; ok: boolean }) {
  return (
    <div className={`rounded border px-3 py-2 ${
      ok ? "border-emerald-500/40 bg-emerald-500/5 text-emerald-300"
         : "border-border bg-background/40 text-muted-foreground"
    }`}>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] opacity-80">
        <span>{label}</span>
        <span>≥ {(target * 100).toFixed(0)}%</span>
      </div>
      <div className="mt-0.5 font-mono text-lg tabular-nums">{fmt(v)}</div>
    </div>
  );
}

function Metric({ label, v }: { label: string; v: string }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 px-2 py-1">
      <div className="text-[9px] uppercase tracking-[0.14em] text-muted-foreground/80">{label}</div>
      <div className="font-mono text-foreground/90">{v}</div>
    </div>
  );
}

function PhaseSpace({
  spec, engine, state, replay,
}: { spec: TwinSpec; engine: DigitalTwinEngine; state: TwinState; replay: boolean }) {
  const w = 360, h = 220, pad = 12;
  const hist = engine.replay(spec.id);
  const points = hist.map((h) => [h.latent[0] ?? 0, h.latent[1] ?? 0] as [number, number]);
  const forecast = state.forecast.map((p) => [p[0] ?? 0, p[1] ?? 0] as [number, number]);
  const all = [...points, ...forecast, [0, 0]];
  const xs = all.map((p) => p[0]);
  const ys = all.map((p) => p[1]);
  const mn = (a: number[]) => Math.min(...a), mx = (a: number[]) => Math.max(...a);
  const xr = [mn(xs) - 0.2, mx(xs) + 0.2];
  const yr = [mn(ys) - 0.2, mx(ys) + 0.2];
  const sx = (x: number) => pad + ((x - xr[0]) / Math.max(1e-6, xr[1] - xr[0])) * (w - pad * 2);
  const sy = (y: number) => h - pad - ((y - yr[0]) / Math.max(1e-6, yr[1] - yr[0])) * (h - pad * 2);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ");
  const fpath = forecast.length
    ? `M${sx(points[points.length - 1]?.[0] ?? 0).toFixed(1)},${sy(points[points.length - 1]?.[1] ?? 0).toFixed(1)} ` +
      forecast.map((p) => `L${sx(p[0]).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(" ")
    : "";

  return (
    <div className="rounded border border-border bg-background/40 p-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-1">
        <span>Latent phase-space · PCA₁ × PCA₂</span>
        <span>{replay ? "replay" : "live"} · {points.length} pts</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-44">
        <rect x={0} y={0} width={w} height={h} className="fill-background/50" />
        {/* optimum origin */}
        <circle cx={sx(0)} cy={sy(0)} r={3} className="fill-emerald-400" />
        <text x={sx(0) + 5} y={sy(0) - 5} className="fill-emerald-300/70 text-[8px]">optimum</text>
        {/* history */}
        <path d={path} fill="none" className="stroke-primary/70" strokeWidth={1.2} />
        {/* forecast */}
        {fpath && <path d={fpath} fill="none" className="stroke-amber-300/80" strokeWidth={1.2} strokeDasharray="3 2" />}
        {/* current */}
        {points.length > 0 && (
          <circle cx={sx(points[points.length - 1][0])} cy={sy(points[points.length - 1][1])}
                  r={3.2} className="fill-primary" />
        )}
      </svg>
      <div className="flex items-center justify-between text-[9px] text-muted-foreground/70">
        <span>━ trajectory</span><span className="text-amber-300/80">┄ forecast (next {forecast.length})</span>
      </div>
    </div>
  );
}

function ForecastChart({ state }: { state: TwinState }) {
  const w = 360, h = 80, pad = 8;
  const xs = state.forecastDistance;
  if (xs.length === 0) return null;
  const max = Math.max(0.1, ...xs);
  const path = xs.map((v, i) => {
    const x = pad + (i / Math.max(1, xs.length - 1)) * (w - pad * 2);
    const y = h - pad - (v / max) * (h - pad * 2);
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <div className="rounded border border-border bg-background/40 p-2">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-1">
        <span>Forecast · distance-to-optimum (next {xs.length})</span>
        <span className="tabular-nums">peak {max.toFixed(3)}</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-16">
        <rect x={0} y={0} width={w} height={h} className="fill-background/40" />
        <path d={path} fill="none" className="stroke-amber-300/80" strokeWidth={1.2} />
      </svg>
    </div>
  );
}
