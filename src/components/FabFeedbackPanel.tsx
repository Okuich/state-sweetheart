// Fabrication Feedback Calibration System — UI panel
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  CHANNELS,
  CHANNEL_UNITS,
  correct,
  generateBatch,
  ingestBatch,
  initState,
  residuals,
  resetSuff,
  uncertainty,
  type CalibrationState,
  type Channel,
  type Observation,
} from "@/lib/fabFeedback";
import { scanImportBridge } from "@/lib/scanImportStore";

export function FabFeedbackPanel() {
  const [state, setState] = useState<CalibrationState>(() => initState());
  const [obs, setObs] = useState<Observation[]>([]);
  const [batchSize, setBatchSize] = useState(40);
  const [streaming, setStreaming] = useState(false);
  const [seed, setSeed] = useState(1);
  const [active, setActive] = useState<Channel>("dimensional");
  const tickRef = useRef<number | null>(null);

  // streaming ingest
  useEffect(() => {
    if (!streaming) {
      if (tickRef.current) {
        window.clearInterval(tickRef.current);
        tickRef.current = null;
      }
      return;
    }
    tickRef.current = window.setInterval(() => {
      const b = generateBatch(8, seed + Math.floor(Math.random() * 1e6));
      setObs((prev) => [...prev.slice(-1000), ...b]);
      setState((prev) => ingestBatch(prev, b));
    }, 600);
    return () => {
      if (tickRef.current) window.clearInterval(tickRef.current);
      tickRef.current = null;
    };
  }, [streaming, seed]);

  // ingest observations imported from external scan/quality reports
  useEffect(() => {
    return scanImportBridge.subscribe((batch) => {
      setObs((prev) => [...prev.slice(-1500), ...batch]);
      setState((prev) => ingestBatch(prev, batch));
    });
  }, []);

  const ingest = () => {
    const b = generateBatch(batchSize, seed + obs.length);
    setObs((prev) => [...prev.slice(-1500), ...b]);
    setState((prev) => ingestBatch(prev, b));
  };
  const reset = () => {
    setStreaming(false);
    resetSuff();
    setObs([]);
    setState(initState());
  };

  const stats = useMemo(
    () => CHANNELS.map((c) => residuals(obs, state[c], c)),
    [obs, state]
  );
  const totalReduction = useMemo(() => {
    const valid = stats.filter((s) => s.n > 0);
    return valid.length === 0
      ? 0
      : valid.reduce((a, s) => a + s.reduction, 0) / valid.length;
  }, [stats]);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
            module · midwater-feedback
          </div>
          <h2 className="font-display text-2xl md:text-3xl text-glow">
            Midwater <span className="text-primary">Feedback</span> Calibration
          </h2>
          <p className="text-xs text-muted-foreground max-w-xl mt-1">
            Online ridge least-squares per channel. Closes the loop on
            dimensional, thermal, tolerance and surface measurements —
            shrinks RMSE and posterior uncertainty as more parts are scanned.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={ingest} className="uppercase tracking-[0.16em] text-[10px]">
            ingest · {batchSize}
          </Button>
          <Button
            variant={streaming ? "default" : "outline"}
            onClick={() => setStreaming((s) => !s)}
            className={`uppercase tracking-[0.16em] text-[10px] ${streaming ? "bg-accent text-accent-foreground" : ""}`}
          >
            {streaming ? "streaming · on" : "streaming · off"}
          </Button>
          <Button variant="outline" onClick={reset} className="uppercase tracking-[0.16em] text-[10px]">
            reset
          </Button>
        </div>
      </header>

      {/* hyperparams */}
      <div className="grid gap-3 md:grid-cols-3 text-xs">
        <NumInput label="batch size" value={batchSize} onChange={setBatchSize} min={1} max={500} step={1} />
        <NumInput label="seed"       value={seed}      onChange={setSeed}      min={0} max={9999} step={1} />
        <Stat     label="avg RMSE reduction" value={`${(totalReduction * 100).toFixed(1)} %`} tone="ok" />
      </div>

      {/* per-channel cards */}
      <div className="grid gap-3 md:grid-cols-2">
        {CHANNELS.map((c) => {
          const m = state[c];
          const r = stats.find((s) => s.channel === c)!;
          const u = uncertainty(m);
          return (
            <button
              key={c}
              onClick={() => setActive(c)}
              className={`text-left rounded-lg border p-3 transition ${
                active === c ? "border-primary bg-primary/5" : "border-border bg-background/40 hover:border-foreground/30"
              }`}
            >
              <div className="flex items-baseline justify-between mb-2">
                <div className="text-sm font-mono">{c}</div>
                <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
                  {CHANNEL_UNITS[c]} · n={m.n}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px]">
                <Mini label="scale" value={`${m.scale.toFixed(3)} ± ${u.sdScale.toFixed(3)}`} />
                <Mini label="bias"  value={`${m.bias.toFixed(2)} ± ${u.sdBias.toFixed(2)}`} />
                <Mini label="rmse · raw" value={r.rmseRaw.toFixed(2)} tone="muted" />
                <Mini label="rmse · cal" value={r.rmseCorrected.toFixed(2)} tone="ok" />
              </div>
              {/* confidence bar */}
              <div className="mt-2 h-1 rounded-sm bg-muted overflow-hidden">
                <div className="h-full bg-primary"
                     style={{ width: `${(u.confidence * 100).toFixed(1)}%` }} />
              </div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground mt-1 flex justify-between">
                <span>confidence</span>
                <span className="text-primary">−{(r.reduction * 100).toFixed(1)}% rmse</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* scatter for active channel */}
      <ScatterCard
        channel={active}
        obs={obs.filter((o) => o.channel === active)}
        model={state[active]}
      />
    </div>
  );
}

// ─── scatter (predicted vs measured) + correction line ──────
function ScatterCard({
  channel, obs, model,
}: { channel: Channel; obs: Observation[]; model: ReturnType<typeof initState>[Channel] }) {
  const W = 720, H = 220, pad = 30;
  const data = obs.slice(-300);
  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-background/40 p-6 text-[11px] text-muted-foreground">
        no <span className="text-foreground/80">{channel}</span> observations yet —
        ingest a batch or enable streaming.
      </div>
    );
  }
  const xs = data.map((d) => d.predicted);
  const ys = data.map((d) => d.measured);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys, ...xs), yMax = Math.max(...ys, ...xs);
  const sx = (x: number) => pad + ((x - xMin) / (xMax - xMin || 1)) * (W - pad * 2);
  const sy = (y: number) => H - pad - ((y - yMin) / (yMax - yMin || 1)) * (H - pad * 2);

  // identity (raw) line and corrected line
  const idA = { x: xMin, y: xMin };
  const idB = { x: xMax, y: xMax };
  const corA = { x: xMin, y: correct(model, xMin) };
  const corB = { x: xMax, y: correct(model, xMax) };

  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
        scatter · {channel}  (predicted → measured, last {data.length})
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[220px]">
        {/* axes */}
        <line x1={pad} y1={H-pad} x2={W-pad} y2={H-pad} stroke="hsl(var(--border))" />
        <line x1={pad} y1={pad}   x2={pad}   y2={H-pad} stroke="hsl(var(--border))" />
        {/* identity */}
        <line x1={sx(idA.x)} y1={sy(idA.y)} x2={sx(idB.x)} y2={sy(idB.y)}
              stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" strokeWidth={1} />
        {/* correction line */}
        <line x1={sx(corA.x)} y1={sy(corA.y)} x2={sx(corB.x)} y2={sy(corB.y)}
              stroke="hsl(var(--primary))" strokeWidth={1.5} />
        {/* points */}
        {data.map((d, i) => (
          <circle key={i} cx={sx(d.predicted)} cy={sy(d.measured)} r={1.6}
                  fill="hsl(var(--accent))" opacity={0.75} />
        ))}
        {/* axis labels */}
        <text x={W-pad} y={H-pad+14} textAnchor="end" fontSize={9}
              fill="hsl(var(--muted-foreground))" className="font-mono">
          predicted ({CHANNEL_UNITS[channel]})
        </text>
        <text x={pad} y={pad-8} fontSize={9}
              fill="hsl(var(--muted-foreground))" className="font-mono">
          measured ({CHANNEL_UNITS[channel]})
        </text>
      </svg>
      <div className="text-[10px] font-mono text-muted-foreground mt-1">
        dashed · identity (no correction) &nbsp; solid · learned correction
        &nbsp; ŷ = {model.scale.toFixed(3)}·x + {model.bias.toFixed(2)}
      </div>
    </div>
  );
}

// ─── atoms ──────────────────────────────────────────────────
function NumInput({
  label, value, onChange, min, max, step,
}: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</span>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md border border-border bg-background/60 px-2 py-1 font-mono text-xs outline-none focus:ring-1 focus:ring-primary" />
    </label>
  );
}
function Stat({ label, value, tone }: { label: string; value: string; tone?: "ok" | "muted" }) {
  const cls = tone === "ok" ? "text-primary" : "text-foreground/80";
  return (
    <div className="rounded-md border border-border bg-card/60 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      <div className={`font-mono tabular-nums ${cls}`}>{value}</div>
    </div>
  );
}
function Mini({ label, value, tone }: { label: string; value: string; tone?: "ok" | "muted" }) {
  const cls = tone === "ok" ? "text-primary" : tone === "muted" ? "text-muted-foreground" : "text-foreground/85";
  return (
    <div className="rounded-md border border-border bg-card/40 px-1.5 py-1">
      <div className="text-[8px] uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className={`font-mono tabular-nums ${cls}`}>{value}</div>
    </div>
  );
}
