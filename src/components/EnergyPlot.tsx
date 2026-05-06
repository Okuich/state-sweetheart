// Real-time energy plots — verifies gravity transfers PE↔KE and that
// damping monotonically dissipates total mechanical energy.
import { useEffect, useRef, useState } from "react";
import type { EnergySample } from "./PhysicsCanvas";

const CAP = 480; // ~8s at 60fps

type Series = {
  t: Float32Array;
  KE: Float32Array;
  PE: Float32Array;
  E: Float32Array;
  drift: Float32Array;
  head: number;
  len: number;
  baseline: number;
};

function emptySeries(): Series {
  return {
    t: new Float32Array(CAP),
    KE: new Float32Array(CAP),
    PE: new Float32Array(CAP),
    E: new Float32Array(CAP),
    drift: new Float32Array(CAP),
    head: 0,
    len: 0,
    baseline: 0,
  };
}

export type EnergyPlotHandle = {
  push: (s: EnergySample) => void;
  clear: () => void;
};

export function useEnergyPlot(): {
  handleRef: React.MutableRefObject<EnergyPlotHandle | null>;
  Plot: React.FC<{ height?: number }>;
} {
  const handleRef = useRef<EnergyPlotHandle | null>(null);
  const Plot: React.FC<{ height?: number }> = ({ height = 180 }) => (
    <EnergyPlot height={height} register={(h) => (handleRef.current = h)} />
  );
  return { handleRef, Plot };
}

function EnergyPlot({
  height,
  register,
}: { height: number; register: (h: EnergyPlotHandle) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const seriesRef = useRef<Series>(emptySeries());
  const dirtyRef = useRef(false);
  const latestRef = useRef<EnergySample | null>(null);

  // expose imperative push/clear
  useEffect(() => {
    register({
      push: (sample) => {
        const s = seriesRef.current;
        if (s.len === 0) s.baseline = sample.E;
        s.t[s.head]     = sample.t;
        s.KE[s.head]    = sample.KE;
        s.PE[s.head]    = sample.PE;
        s.E[s.head]     = sample.E;
        s.drift[s.head] = sample.drift;
        s.head = (s.head + 1) % CAP;
        if (s.len < CAP) s.len += 1;
        latestRef.current = sample;
        dirtyRef.current = true;
      },
      clear: () => {
        seriesRef.current = emptySeries();
        latestRef.current = null;
        dirtyRef.current = true;
      },
    });
  }, [register]);

  // render loop — driven by rAF, throttled to 30fps for cheap drawing
  useEffect(() => {
    let raf = 0;
    let lastDraw = 0;
    const draw = (now: number) => {
      raf = requestAnimationFrame(draw);
      if (!dirtyRef.current || now - lastDraw < 33) return;
      lastDraw = now;
      dirtyRef.current = false;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.clientWidth;
      const cssH = canvas.clientHeight;
      if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
        canvas.width = cssW * dpr;
        canvas.height = cssH * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, cssW, cssH);
      drawSeries(ctx, seriesRef.current, cssW, cssH);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="rounded-lg border border-border bg-background/40 p-3 space-y-2">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
          energy · KE / PE / total · drift overlay
        </div>
        <Legend />
      </div>
      <canvas ref={canvasRef} style={{ width: "100%", height }} className="block" />
      <Readout latestRef={latestRef} />
    </div>
  );
}

function Legend() {
  const items: [string, string][] = [
    ["KE", "rgb(110,231,255)"],
    ["PE", "rgb(255,176,82)"],
    ["E_total", "rgb(160,255,178)"],
    ["drift", "rgb(255,108,140)"],
  ];
  return (
    <div className="flex gap-3">
      {items.map(([l, c]) => (
        <div key={l} className="flex items-center gap-1 text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
          <span style={{ background: c }} className="inline-block w-2.5 h-2.5 rounded-sm" />
          {l}
        </div>
      ))}
    </div>
  );
}

function Readout({ latestRef }: { latestRef: React.MutableRefObject<EnergySample | null> }) {
  // re-render on rAF tick for live numeric readout
  const [, force] = useTick();
  const s = latestRef.current;
  if (!s) {
    return <div className="text-[10px] text-muted-foreground">awaiting first frame…</div>;
  }
  const fmt = (n: number) => {
    const a = Math.abs(n);
    if (a >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (a >= 1e3) return (n / 1e3).toFixed(2) + "k";
    if (a >= 1)   return n.toFixed(2);
    return n.toExponential(1);
  };
  const rel = Math.abs(s.baseline) > 1e-9 ? (s.drift / Math.abs(s.baseline)) * 100 : 0;
  void force;
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px] font-mono">
      <Stat label="KE"      value={fmt(s.KE)}     accent="rgb(110,231,255)" />
      <Stat label="PE"      value={fmt(s.PE)}     accent="rgb(255,176,82)" />
      <Stat label="E_total" value={fmt(s.E)}      accent="rgb(160,255,178)" />
      <Stat label="Δ/E₀"    value={`${rel >= 0 ? "+" : ""}${rel.toFixed(2)}%`} accent="rgb(255,108,140)" />
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: string; accent: string }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 px-2 py-1">
      <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      <div className="tabular-nums" style={{ color: accent }}>{value}</div>
    </div>
  );
}

function useTick(intervalMs = 100) {
  const ref = useRef(0);
  // re-render at intervalMs
  useEffect(() => {
    const id = window.setInterval(() => {
      ref.current = (ref.current + 1) >>> 0;
      // trigger re-render via state setter substitute
      setRerender((x) => x + 1);
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs]);
  const [, setRerender] = useStateRef();
  return [ref.current, setRerender] as const;
}

function useStateRef() {
  // tiny useState shim avoiding a separate import line
  // eslint-disable-next-line react-hooks/rules-of-hooks
  const useStateImpl = require("react").useState as <T>(v: T) => [T, (v: T | ((p: T) => T)) => void];
  return useStateImpl(0);
}

function drawSeries(
  ctx: CanvasRenderingContext2D,
  s: Series,
  w: number,
  h: number,
) {
  // grid
  ctx.fillStyle = "rgba(255,255,255,0.02)";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h * i) / 4;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  if (s.len < 2) return;

  // pull arrays in chronological order
  const n = s.len;
  const start = (s.head - n + CAP) % CAP;
  const KE = new Float32Array(n);
  const PE = new Float32Array(n);
  const E  = new Float32Array(n);
  const D  = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const k = (start + i) % CAP;
    KE[i] = s.KE[k]; PE[i] = s.PE[k]; E[i] = s.E[k]; D[i] = s.drift[k];
  }

  // shared y-range for KE/PE/E (positive scales). Drift uses a separate
  // signed range painted as a thin overlay along the bottom band.
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    if (KE[i] < lo) lo = KE[i]; if (KE[i] > hi) hi = KE[i];
    if (PE[i] < lo) lo = PE[i]; if (PE[i] > hi) hi = PE[i];
    if (E[i]  < lo) lo = E[i];  if (E[i]  > hi) hi = E[i];
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-9) {
    lo -= 1; hi += 1;
  }
  const padE = (hi - lo) * 0.08;
  lo -= padE; hi += padE;

  let dMax = 1e-9;
  for (let i = 0; i < n; i++) if (Math.abs(D[i]) > dMax) dMax = Math.abs(D[i]);

  const plotH = h * 0.78;
  const driftH = h * 0.22;
  const driftY0 = plotH + driftH * 0.5;

  const xAt = (i: number) => (i / (n - 1)) * (w - 2) + 1;
  const yAt = (v: number) => plotH - ((v - lo) / (hi - lo)) * (plotH - 4) - 2;

  const drawLine = (arr: Float32Array, color: string, lw = 1.5) => {
    ctx.strokeStyle = color; ctx.lineWidth = lw;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = xAt(i), y = yAt(arr[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  drawLine(KE, "rgb(110,231,255)");
  drawLine(PE, "rgb(255,176,82)");
  drawLine(E,  "rgb(160,255,178)", 2);

  // drift band — zero line + signed line
  ctx.strokeStyle = "rgba(255,255,255,0.18)";
  ctx.beginPath();
  ctx.moveTo(0, driftY0); ctx.lineTo(w, driftY0); ctx.stroke();
  ctx.strokeStyle = "rgb(255,108,140)";
  ctx.lineWidth = 1.25;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = xAt(i);
    const y = driftY0 - (D[i] / dMax) * (driftH * 0.45);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // labels
  ctx.fillStyle = "rgba(255,255,255,0.5)";
  ctx.font = "9px ui-monospace,monospace";
  ctx.fillText("E", 4, 11);
  ctx.fillText("Δ", 4, plotH + 11);
}
