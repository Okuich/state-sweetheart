import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  calibrate, generateMeasurements, modelAt, TRUE_PARAMS,
  type FitResult, type MeasurementSample, type ModelParams,
} from "@/lib/calibration";

const MATERIALS: { name: string; init: ModelParams }[] = [
  { name: "Al-7075",  init: { k: 30,  c: 0.3, m: 1.2, A: 0.9 } },
  { name: "SS-316",   init: { k: 80,  c: 1.2, m: 1.6, A: 0.8 } },
  { name: "CFRP",     init: { k: 60,  c: 0.2, m: 0.7, A: 1.0 } },
  { name: "ABS",      init: { k: 15,  c: 0.8, m: 0.9, A: 1.0 } },
];

export function CalibrationPanel() {
  const [N, setN] = useState(120);
  const [sigma, setSigma] = useState(0.04);
  const [matIdx, setMatIdx] = useState(0);
  const [data, setData] = useState<MeasurementSample[]>(() => generateMeasurements(120, 6, 0.04));
  const [fit, setFit] = useState<FitResult | null>(null);
  const [running, setRunning] = useState(false);

  const ingest = () => setData(generateMeasurements(N, 6, sigma));

  const run = () => {
    setRunning(true);
    requestAnimationFrame(() => {
      const r = calibrate(MATERIALS[matIdx].init, data, 40);
      setFit(r);
      setRunning(false);
    });
  };

  const view = useMemo(() => {
    if (data.length < 2) return null;
    const W = 600, H = 160;
    const tMin = data[0].t, tMax = data[data.length - 1].t;
    const span = Math.max(1e-3, tMax - tMin);
    const fitVals = fit ? data.map((s) => modelAt(fit.params, s.t)) : [];
    const trueVals = data.map((s) => modelAt(TRUE_PARAMS, s.t));
    const all = data.map((d) => d.y).concat(fitVals, trueVals);
    const lo = Math.min(...all), hi = Math.max(...all);
    const ys = Math.max(1e-3, hi - lo);
    const xy = (t: number, v: number) => [
      ((t - tMin) / span) * W,
      H - ((v - lo) / ys) * (H - 8) - 4,
    ];
    const truePath = data.map((s, i) => {
      const [x, y] = xy(s.t, modelAt(TRUE_PARAMS, s.t));
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    const fitPath = fit ? data.map((s, i) => {
      const [x, y] = xy(s.t, modelAt(fit.params, s.t));
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ") : "";
    return { W, H, truePath, fitPath, points: data.map((s) => xy(s.t, s.y)) };
  }, [data, fit]);

  const residView = useMemo(() => {
    if (!fit) return null;
    const W = 600, H = 70;
    const max = Math.max(...fit.residuals.map(Math.abs), 1e-3);
    const path = fit.residuals.map((r, i) => {
      const x = (i / (fit.residuals.length - 1)) * W;
      const y = H / 2 - (r / max) * (H / 2 - 4);
      return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
    return { W, H, path };
  }, [fit]);

  const trueP = TRUE_PARAMS;

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            calibration · real-world fit
          </div>
          <h2 className="font-display text-2xl text-foreground">
            Fit the model to <span className="text-primary">measured</span> data.
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {fit ? <>R² · <span className={fit.r2 > 0.97 ? "text-primary" : "text-accent"}>{fit.r2.toFixed(4)}</span></> : "no fit"}
        </div>
      </div>

      {/* Sensor ingest controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Knob label="Samples" value={N} min={20} max={400} step={10}
          onChange={(v) => setN(v)} />
        <Knob label="Sensor σ" value={sigma} min={0} max={0.25} step={0.005}
          onChange={(v) => setSigma(v)} />
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">Material seed</div>
          <div className="grid grid-cols-2 gap-1">
            {MATERIALS.map((m, i) => (
              <button key={m.name} onClick={() => setMatIdx(i)}
                className={`text-[10px] uppercase tracking-[0.18em] px-2 py-1.5 rounded border transition ${
                  matIdx === i ? "border-primary bg-primary/10 text-foreground"
                  : "border-border bg-background/30 text-muted-foreground hover:text-foreground"
                }`}>
                {m.name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="flex gap-2">
        <Button onClick={ingest} variant="outline" className="uppercase tracking-[0.18em] text-[10px]">
          ingest sensors
        </Button>
        <Button onClick={run} disabled={running} className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "calibrating…" : fit ? "re-fit (LM)" : "calibrate"}
        </Button>
      </div>

      {/* Plot */}
      <div className="rounded-md border border-border bg-background/30 p-3">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
          measured · ground-truth · fitted
        </div>
        {view ? (
          <svg viewBox={`0 0 ${view.W} ${view.H}`} width="100%" height={view.H} className="block">
            <path d={view.truePath} fill="none" stroke="currentColor"
              className="text-muted-foreground/40" strokeDasharray="3 3" strokeWidth={1} />
            {view.points.map((p, i) => (
              <circle key={i} cx={p[0]} cy={p[1]} r={1.4} className="fill-accent/70" />
            ))}
            {fit && view.fitPath && (
              <path d={view.fitPath} fill="none" stroke="currentColor"
                className="text-primary" strokeWidth={1.6} />
            )}
          </svg>
        ) : null}
        <div className="mt-2 flex gap-4 text-[9px] uppercase tracking-[0.2em] text-muted-foreground">
          <span><span className="inline-block w-3 h-px align-middle bg-muted-foreground/40 mr-1" /> truth</span>
          <span><span className="inline-block w-1.5 h-1.5 align-middle rounded-full bg-accent/70 mr-1" /> measured</span>
          <span><span className="inline-block w-3 h-px align-middle bg-primary mr-1" /> fitted</span>
        </div>
      </div>

      {/* Residuals */}
      {fit && residView && (
        <div className="rounded-md border border-border bg-background/30 p-3">
          <div className="flex items-baseline justify-between mb-2">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              residuals · y - ŷ
            </div>
            <div className="text-[10px] font-mono text-muted-foreground">
              rmse · <span className="text-foreground tabular-nums">{fit.rmse.toFixed(4)}</span>
              <span className="ml-3">iters · <span className="text-foreground">{fit.iters}</span></span>
            </div>
          </div>
          <svg viewBox={`0 0 ${residView.W} ${residView.H}`} width="100%" height={residView.H}>
            <line x1={0} y1={residView.H / 2} x2={residView.W} y2={residView.H / 2}
              stroke="currentColor" className="text-muted-foreground/30" />
            <path d={residView.path} fill="none" stroke="currentColor"
              className="text-secondary" strokeWidth={1.2} />
          </svg>
        </div>
      )}

      {/* Parameter estimates */}
      {fit && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
          {(Object.keys(fit.params) as (keyof ModelParams)[]).map((k) => {
            const est = fit.params[k];
            const tru = trueP[k];
            const err = Math.abs(est - tru) / Math.max(Math.abs(tru), 1e-9) * 100;
            return (
              <div key={k} className="rounded border border-border/60 px-2 py-1.5">
                <div className="uppercase tracking-[0.18em] text-muted-foreground">{k}</div>
                <div className="font-mono text-foreground/95 tabular-nums">{est.toFixed(3)}</div>
                <div className="text-[9px] text-muted-foreground/80 font-mono">
                  truth {tru.toFixed(2)} · Δ <span className={err < 5 ? "text-primary" : err < 15 ? "text-accent" : "text-destructive"}>
                    {err.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function Knob({ label, value, min, max, step, onChange }: {
  label: string; value: number;
  min: number; max: number; step: number; onChange: (v: number) => void;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between text-[10px] uppercase tracking-[0.18em]">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-primary tabular-nums">{value.toFixed(step < 1 ? 3 : 0)}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => onChange(v)} />
    </div>
  );
}
