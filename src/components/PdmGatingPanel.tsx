/**
 * Predictive Maintenance — Gating System panel.
 *
 *   Gate A · false-positive rate < 8%
 *   Gate B · balanced accuracy   > 80%
 *   Gate C · p95 inference       < 100 ms
 *
 * Passing all three gates unlocks the Digital Twin Layer (live latent-
 * space mirror of the synthetic asset under test).
 *
 * The panel drives its own small evaluation harness so the gates can be
 * measured deterministically without depending on the live operator
 * fleet in PredictiveMaintenancePanel.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  PredictiveMaintenanceEngine,
  type MaintenanceScore,
  type SensorReading,
} from "@/lib/pdm";
import {
  GateEvaluator, DEFAULT_GATES, type GateReport, type GroundTruth,
} from "@/lib/pdm/gating";

const ASSET_ID = "gating-rig";

function seedExemplars(engine: PredictiveMaintenanceEngine) {
  const optimal: SensorReading[] = Array.from({ length: 24 }, (_, i) => ({
    assetId: ASSET_ID, t: -2000 + i,
    channels: {
      vibration: 1.0 + 0.08 * Math.sin(i),
      pressure: 10 + 0.05 * Math.cos(i / 2),
      temperature: 60 + 0.4 * Math.sin(i / 3),
      flow_rate: 5 + 0.1 * Math.cos(i),
      current: 12 + 0.05 * Math.sin(i / 2),
      acoustic: 0.3, hydraulic: 2.0,
    },
  }));
  const failures: SensorReading[] = Array.from({ length: 10 }, (_, i) => ({
    assetId: ASSET_ID, t: -1000 + i,
    channels: {
      vibration: 5.2 + i * 0.3, pressure: 14 + i * 0.2,
      temperature: 95 + i,      flow_rate: 1.8,
      current: 22 + i * 0.5,    acoustic: 1.8, hydraulic: 4.2,
    },
  }));
  engine.registerExemplars(ASSET_ID, { optimal, failures });
}

/** Synthetic reading + ground-truth for the rig. */
function genReading(t: number, truth: GroundTruth): SensorReading {
  // Healthy: small zero-mean noise around the optimal manifold.
  // Fault: drift along the failure direction proportional to a ramp.
  const d = truth === "fault" ? 1 + (Math.sin(t / 9) * 0.5 + 0.5) * 2.5 : 0;
  const n = () => (Math.random() - 0.5) * 0.1;
  return {
    assetId: ASSET_ID, t,
    channels: {
      vibration:   1.0 + 0.1 * Math.sin(t / 5) + d * 0.45 + n(),
      pressure:    10  + 0.05 * Math.cos(t / 4) + d * 0.30 + n(),
      temperature: 60  + 0.30 * Math.sin(t / 7) + d * 2.60 + n(),
      flow_rate:   5   - d * 0.22 + n(),
      current:     12  + 0.05 * Math.sin(t / 3) + d * 0.65 + n(),
      acoustic:    0.3 + d * 0.28 + n() * 0.2,
      hydraulic:   2   + d * 0.18 + n() * 0.2,
    },
  };
}

export function PdmGatingPanel() {
  const engineRef = useRef<PredictiveMaintenanceEngine | null>(null);
  const evalRef = useRef<GateEvaluator | null>(null);
  if (!engineRef.current) {
    engineRef.current = new PredictiveMaintenanceEngine({
      latentDim: 4, warmupSamples: 24, windowLen: 16, clusters: 3,
    });
    seedExemplars(engineRef.current);
    evalRef.current = new GateEvaluator(DEFAULT_GATES);
  }
  const engine = engineRef.current;
  const evaluator = evalRef.current!;

  const [running, setRunning] = useState(true);
  const [report, setReport] = useState<GateReport>(() => evaluator.report());
  const [latentPath, setLatentPath] = useState<Array<[number, number]>>([]);
  const tickRef = useRef(0);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      // 80/20 healthy/fault mix per tick — enough to populate both classes.
      for (let k = 0; k < 6; k++) {
        tickRef.current++;
        const t = tickRef.current;
        const truth: GroundTruth = Math.random() < 0.8 ? "healthy" : "fault";
        const reading = genReading(t, truth);
        const t0 = performance.now();
        const score: MaintenanceScore = engine.ingest(reading);
        const dt = performance.now() - t0;
        evaluator.record(truth, score, dt);
      }
      const list = engine.prioritized();
      // Build a tiny 2D latent path for the Digital Twin layer.
      // We don't have direct latent access here, so derive a 2D
      // signature from (riskScore, driftSigma) — sufficient for a live
      // phase-space mirror in the UI.
      const head = list[0];
      if (head) {
        setLatentPath((prev) => {
          const next = [...prev, [head.driftSigma, head.riskScore] as [number, number]];
          if (next.length > 96) next.shift();
          return next;
        });
      }
      setReport(evaluator.report());
    }, 200);
    return () => clearInterval(id);
  }, [engine, evaluator, running]);

  const reset = () => {
    engine.reset();
    seedExemplars(engine);
    evaluator.reset();
    setLatentPath([]);
    setReport(evaluator.report());
    tickRef.current = 0;
  };

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          PdM · Gating System
          <Badge variant="outline" className="text-[10px]">
            {report.unlocked ? "all gates passed" : "evaluating"}
          </Badge>
        </h3>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setRunning((r) => !r)}>
            {running ? "pause" : "resume"}
          </Button>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs"
            onClick={reset}>
            reset
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        <GateCard
          letter="A"
          title="False Positive Rate"
          fmt={(v) => `${(v * 100).toFixed(2)}%`}
          fmtThr={(v) => `< ${(v * 100).toFixed(0)}%`}
          status={report.A}
        />
        <GateCard
          letter="B"
          title="Prediction Accuracy"
          fmt={(v) => `${(v * 100).toFixed(1)}%`}
          fmtThr={(v) => `> ${(v * 100).toFixed(0)}%`}
          status={report.B}
          higherIsBetter
        />
        <GateCard
          letter="C"
          title="Inference Latency (p95)"
          fmt={(v) => `${v.toFixed(2)} ms`}
          fmtThr={(v) => `< ${v} ms`}
          status={report.C}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 text-[10px] tabular-nums text-muted-foreground sm:grid-cols-6">
        <Stat label="healthy" v={report.counts.healthy} />
        <Stat label="fault"   v={report.counts.fault} />
        <Stat label="TP"      v={report.counts.tp} />
        <Stat label="TN"      v={report.counts.tn} />
        <Stat label="FP"      v={report.counts.fp} />
        <Stat label="FN"      v={report.counts.fn} />
      </div>

      <DigitalTwinLayer unlocked={report.unlocked} path={latentPath} />
    </div>
  );
}

function GateCard({
  letter, title, status, fmt, fmtThr, higherIsBetter,
}: {
  letter: "A" | "B" | "C";
  title: string;
  status: GateReport["A"];
  fmt: (v: number) => string;
  fmtThr: (v: number) => string;
  higherIsBetter?: boolean;
}) {
  const tone = !status.ready
    ? "border-border bg-background/40 text-muted-foreground"
    : status.pass
      ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300"
      : "border-destructive/60 bg-destructive/10 text-destructive";
  const ratio = higherIsBetter
    ? Math.min(1, status.value / Math.max(1e-9, status.threshold))
    : Math.min(1, status.value / Math.max(1e-9, status.threshold));
  return (
    <div className={`rounded border px-3 py-2 ${tone}`}>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.16em] opacity-80">
        <span>Gate {letter}</span>
        <span>{!status.ready ? "warming" : status.pass ? "PASS" : "FAIL"}</span>
      </div>
      <div className="mt-0.5 text-[11px] opacity-90">{title}</div>
      <div className="mt-1 flex items-baseline justify-between">
        <span className="font-mono text-lg tabular-nums">{fmt(status.value)}</span>
        <span className="text-[10px] opacity-70">{fmtThr(status.threshold)}</span>
      </div>
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded bg-background/40">
        <div
          className="h-full bg-current opacity-70 transition-[width]"
          style={{ width: `${Math.max(2, Math.round(ratio * 100))}%` }}
        />
      </div>
    </div>
  );
}

function Stat({ label, v }: { label: string; v: number }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 px-2 py-1">
      <div className="text-[9px] uppercase tracking-[0.14em] opacity-70">{label}</div>
      <div className="font-mono">{v}</div>
    </div>
  );
}

function DigitalTwinLayer({
  unlocked, path,
}: { unlocked: boolean; path: Array<[number, number]> }) {
  if (!unlocked) {
    return (
      <div className="rounded border border-dashed border-border bg-background/40 p-3 text-[11px] text-muted-foreground">
        <div className="flex items-center justify-between">
          <span className="uppercase tracking-[0.18em]">Digital Twin Layer · locked</span>
          <span className="text-[10px] opacity-70">pass all gates to unlock</span>
        </div>
        <div className="mt-1 text-[10px] opacity-70">
          The Digital Twin Layer mirrors the asset in latent phase-space.
          It becomes available once the metric engine demonstrates production
          fitness across false-positive rate, accuracy, and latency.
        </div>
      </div>
    );
  }

  // Render the latent path as a live phase-space view (drift σ × risk).
  const w = 520, h = 120, pad = 8;
  const maxX = Math.max(1, ...path.map((p) => p[0]));
  const pts = path.map(([x, y], i) => {
    const px = pad + (i / Math.max(1, path.length - 1)) * (w - pad * 2);
    const py = h - pad - Math.min(1, y) * (h - pad * 2);
    return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
  }).join(" ");
  const dots = path.map(([x, y], i) => {
    const px = pad + Math.min(1, x / maxX) * (w - pad * 2);
    const py = h - pad - Math.min(1, y) * (h - pad * 2);
    return <circle key={i} cx={px} cy={py} r={1.4} className="fill-emerald-300/60" />;
  });

  return (
    <div className="rounded border border-emerald-500/40 bg-emerald-500/5 p-3">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-emerald-300">
        <span>Digital Twin Layer · unlocked</span>
        <span className="opacity-70">phase-space mirror · drift σ × risk</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="mt-2 h-28 w-full">
        <rect x={0} y={0} width={w} height={h} className="fill-background/40" />
        <path d={pts} fill="none" className="stroke-emerald-300/80" strokeWidth={1.2} />
        {dots}
      </svg>
      <div className="mt-1 text-[10px] text-emerald-300/80">
        live twin · {path.length} samples · advisory channel open
      </div>
    </div>
  );
}
