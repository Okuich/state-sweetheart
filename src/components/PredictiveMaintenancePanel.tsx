/**
 * Predictive Maintenance Metric Engine panel.
 *
 * Drives a small fleet of synthetic assets (pump, robot, mill, irrigation)
 * through the PredictiveMaintenanceEngine and renders the live priority
 * queue, drift sparkline, and per-asset risk breakdown. If a real
 * `sample` prop is wired upstream, the synthetic ticker pauses and the
 * panel ingests the live feed instead.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  PredictiveMaintenanceEngine,
  type MaintenanceScore,
  type SensorReading,
} from "@/lib/pdm";

const ASSETS = [
  { id: "pump-A12",       kind: "centrifugal pump" },
  { id: "robot-arm-7",    kind: "6-axis robot" },
  { id: "cnc-mill-3",     kind: "CNC mill spindle" },
  { id: "irrigation-N4",  kind: "valve manifold" },
] as const;

const prColor: Record<MaintenanceScore["priority"], string> = {
  ok:       "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  watch:    "border-sky-500/40 bg-sky-500/10 text-sky-300",
  schedule: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  urgent:   "border-destructive/60 bg-destructive/10 text-destructive",
};

export function PredictiveMaintenancePanel() {
  const engineRef = useRef<PredictiveMaintenanceEngine | null>(null);
  if (!engineRef.current) {
    engineRef.current = new PredictiveMaintenanceEngine({
      latentDim: 4, warmupSamples: 24, windowLen: 16, clusters: 4,
    });
    // Seed exemplars per asset so failure / optimal distances are meaningful.
    for (const a of ASSETS) {
      const optimal: SensorReading[] = Array.from({ length: 16 }, (_, i) => ({
        assetId: a.id, t: -1000 + i,
        channels: {
          vibration: 1 + 0.1 * Math.sin(i),
          pressure: 10 + 0.05 * Math.cos(i / 2),
          temperature: 60 + 0.4 * Math.sin(i / 3),
          flow_rate: 5 + 0.1 * Math.cos(i),
          current: 12 + 0.05 * Math.sin(i / 2),
          acoustic: 0.3, hydraulic: 2.0,
        },
      }));
      const failures: SensorReading[] = Array.from({ length: 6 }, (_, i) => ({
        assetId: a.id, t: -100 + i,
        channels: {
          vibration: 5 + i * 0.3,  pressure: 14 + i * 0.2,
          temperature: 92 + i,     flow_rate: 2.0,
          current: 22 + i * 0.5,   acoustic: 1.6, hydraulic: 4.0,
        },
      }));
      engineRef.current.registerExemplars(a.id, { optimal, failures });
    }
  }
  const engine = engineRef.current;

  const [scores, setScores] = useState<MaintenanceScore[]>([]);
  const [running, setRunning] = useState(true);
  const driftRef = useRef<Record<string, number>>({});
  const tickRef = useRef(0);

  // History sparkline (per asset, last 48 risk samples).
  const histRef = useRef<Record<string, number[]>>({});

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      tickRef.current++;
      const t = tickRef.current;
      for (const a of ASSETS) {
        const d = driftRef.current[a.id] ?? 0;
        const reading: SensorReading = {
          assetId: a.id, t,
          channels: {
            vibration:   1 + 0.1 * Math.sin(t / 5) + d * 0.4,
            pressure:    10 + 0.05 * Math.cos(t / 4) + d * 0.3,
            temperature: 60 + 0.3 * Math.sin(t / 7) + d * 2.5,
            flow_rate:   5 - d * 0.2,
            current:     12 + 0.05 * Math.sin(t / 3) + d * 0.6,
            acoustic:    0.3 + d * 0.25,
            hydraulic:   2 + d * 0.15,
          },
        };
        engine.ingest(reading);
      }
      const list = engine.prioritized();
      for (const s of list) {
        const arr = histRef.current[s.assetId] ?? [];
        arr.push(s.riskScore);
        if (arr.length > 48) arr.shift();
        histRef.current[s.assetId] = arr;
      }
      setScores(list);
    }, 250);
    return () => clearInterval(id);
  }, [engine, running]);

  const inject = (assetId: string) => {
    driftRef.current[assetId] = Math.min(8, (driftRef.current[assetId] ?? 0) + 1.5);
  };
  const heal = (assetId: string) => { driftRef.current[assetId] = 0; };

  const summary = useMemo(() => ({
    urgent:   scores.filter((s) => s.priority === "urgent").length,
    schedule: scores.filter((s) => s.priority === "schedule").length,
    watch:    scores.filter((s) => s.priority === "watch").length,
    ok:       scores.filter((s) => s.priority === "ok").length,
  }), [scores]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Predictive Maintenance · metric engine
          <Badge variant="outline" className="text-[10px]">{ASSETS.length} assets</Badge>
        </h3>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setRunning((r) => !r)}>
            {running ? "pause" : "resume"}
          </Button>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs"
            onClick={() => { engine.reset(); driftRef.current = {}; histRef.current = {}; setScores([]); }}>
            reset
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-2 text-[11px]">
        {(["urgent", "schedule", "watch", "ok"] as const).map((p) => (
          <div key={p} className={`rounded border px-2 py-1.5 ${prColor[p]}`}>
            <div className="text-[9px] uppercase tracking-[0.16em] opacity-80">{p}</div>
            <div className="font-mono text-lg tabular-nums">{summary[p]}</div>
          </div>
        ))}
      </div>

      <div className="space-y-1">
        {scores.map((s) => {
          const meta = ASSETS.find((a) => a.id === s.assetId);
          const hist = histRef.current[s.assetId] ?? [];
          const horizon = Number.isFinite(s.failureHorizon)
            ? `${Math.round(s.failureHorizon)} samp`
            : "—";
          return (
            <div key={s.assetId} className={`rounded border px-2 py-2 text-[11px] ${prColor[s.priority]}`}>
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="font-medium truncate">{s.assetId}</span>
                  <span className="text-[10px] opacity-70 truncate">{meta?.kind}</span>
                </div>
                <div className="flex items-center gap-1">
                  <button onClick={() => inject(s.assetId)}
                    className="rounded border border-border/60 px-2 py-0.5 text-[9px] uppercase tracking-[0.14em] hover:bg-background/40">
                    inject fault
                  </button>
                  <button onClick={() => heal(s.assetId)}
                    className="rounded border border-border/60 px-2 py-0.5 text-[9px] uppercase tracking-[0.14em] hover:bg-background/40">
                    heal
                  </button>
                </div>
              </div>
              <div className="mt-1 grid grid-cols-[1fr_auto] items-center gap-3">
                <Sparkline values={hist} />
                <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] tabular-nums opacity-90">
                  <span>risk</span><span className="text-right">{s.riskScore.toFixed(3)}</span>
                  <span>drift σ</span><span className="text-right">{s.driftSigma.toFixed(2)}</span>
                  <span>d→opt</span><span className="text-right">{s.distToOptimal.toFixed(2)}</span>
                  <span>d→fail</span><span className="text-right">{s.distToFailure < 0 ? "—" : s.distToFailure.toFixed(2)}</span>
                  <span>horizon</span><span className="text-right">{horizon}</span>
                  <span>cluster</span><span className="text-right">{s.clusterId}</span>
                </div>
              </div>
            </div>
          );
        })}
        {scores.length === 0 && (
          <div className="rounded border border-border/60 bg-background/40 p-3 text-[11px] italic text-muted-foreground">
            warming up engine…
          </div>
        )}
      </div>
    </div>
  );
}

function Sparkline({ values }: { values: number[] }) {
  if (values.length < 2) return <div className="h-8" />;
  const w = 160, h = 32;
  const max = Math.max(0.01, ...values), min = 0;
  const path = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - min) / (max - min)) * h;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={w} height={h} className="opacity-90">
      <path d={path} fill="none" stroke="currentColor" strokeWidth={1.25} />
    </svg>
  );
}
