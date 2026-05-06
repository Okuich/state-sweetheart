import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  initPhaseField, resetPhaseField, stepPhaseField,
  type PhaseFieldState,
} from "@/lib/phaseFieldFracture";

type ViewMode = "damage" | "crack" | "psi";

/** Inferno-ish gradient: 0 → near-black, 1 → bright yellow. */
function colormap(v: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(255 * Math.min(1, 0.05 + 1.5 * t));
  const g = Math.round(255 * Math.max(0, t * t * 0.95 - 0.05));
  const b = Math.round(255 * Math.max(0, 0.45 * (1 - t) - 0.1) + 60 * Math.max(0, t - 0.7));
  return [r, g, b];
}

export function FractureVisualizationPanel() {
  const stateRef = useRef<PhaseFieldState>(initPhaseField({ W: 96, H: 96 }));
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [, force] = useState(0);
  const repaint = () => force((n) => n + 1);

  const [running, setRunning] = useState(false);
  const [loadRate, setLoadRate] = useState(0.02);
  const [view, setView] = useState<ViewMode>("damage");
  const [resPow, setResPow] = useState(96);
  const [ellMul, setEllMul] = useState(4);

  const rebuild = (W: number, ell: number) => {
    stateRef.current = initPhaseField({ W, H: W, ell: ell / W });
    repaint();
  };

  // RAF loop.
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const tick = () => {
      stepPhaseField(stateRef.current, loadRate);
      paint();
      // Auto-stop when fully cracked (crack covers > 25% of grid) to be polite.
      const last = stateRef.current.crackSizeHistory.at(-1) ?? 0;
      if (last > 0.25 * stateRef.current.d.length) {
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [running, loadRate]);

  const stepOnce = () => { stepPhaseField(stateRef.current, loadRate); paint(); repaint(); };
  const reset = () => { resetPhaseField(stateRef.current); paint(); repaint(); };

  // Canvas paint.
  const paint = () => {
    const cv = canvasRef.current;
    const s = stateRef.current;
    if (!cv) return;
    if (cv.width !== s.W || cv.height !== s.H) {
      cv.width = s.W; cv.height = s.H;
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(s.W, s.H);
    const N = s.d.length;
    if (view === "damage") {
      for (let i = 0; i < N; i++) {
        const [r, g, b] = colormap(s.d[i]);
        img.data[i * 4 + 0] = r; img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
      }
    } else if (view === "crack") {
      // Binary crack mask atop a faint gray.
      for (let i = 0; i < N; i++) {
        if (s.d[i] > s.dThreshold) {
          img.data[i * 4 + 0] = 255; img.data[i * 4 + 1] = 60;
          img.data[i * 4 + 2] = 80;  img.data[i * 4 + 3] = 255;
        } else {
          const g = 30 + Math.round(40 * s.d[i]);
          img.data[i * 4 + 0] = g; img.data[i * 4 + 1] = g;
          img.data[i * 4 + 2] = g; img.data[i * 4 + 3] = 255;
        }
      }
    } else {
      // ψ⁺ normalized
      let max = 1e-9;
      for (let i = 0; i < N; i++) if (s.psi[i] > max) max = s.psi[i];
      for (let i = 0; i < N; i++) {
        const [r, g, b] = colormap(s.psi[i] / max);
        img.data[i * 4 + 0] = r; img.data[i * 4 + 1] = g;
        img.data[i * 4 + 2] = b; img.data[i * 4 + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  };

  // Repaint when view changes too.
  useEffect(paint, [view]);

  // Growth chart polyline.
  const chart = useMemo(() => {
    const s = stateRef.current;
    const W = 360, H = 80, pad = 4;
    const hist = s.crackSizeHistory;
    if (hist.length < 2) return { W, H, path: "", maxLabel: "0" };
    const max = Math.max(1, ...hist);
    const pts = hist.map((v, i) => {
      const x = pad + (i / (hist.length - 1)) * (W - 2 * pad);
      const y = H - pad - (v / max) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return { W, H, path: `M${pts.join(" L")}`, maxLabel: max.toString() };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stateRef.current.crackSizeHistory.length]);

  const s = stateRef.current;

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Fracture visualization (phase-field gate)
        </h3>
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" onClick={stepOnce} className="h-7 px-2 text-xs">
            step
          </Button>
          <Button variant={running ? "default" : "outline"} size="sm"
                  onClick={() => setRunning((r) => !r)} className="h-7 px-3 text-xs">
            {running ? "pause" : "play"}
          </Button>
          <Button variant="ghost" size="sm" onClick={reset} className="h-7 px-2 text-xs">
            reset
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[auto_1fr]">
        <div className="rounded border border-border/60 bg-background/40 p-2">
          <canvas
            ref={canvasRef}
            className="block h-[360px] w-[360px] image-rendering-pixelated"
            style={{ imageRendering: "pixelated" }}
          />
          <div className="mt-1 flex justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            <span>step {s.step}</span>
            <span>load {s.loadFactor.toFixed(2)}</span>
            <span>{s.W}×{s.H}</span>
          </div>
        </div>

        <div className="space-y-3">
          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">view mode</div>
            <Select value={view} onValueChange={(v) => setView(v as ViewMode)}>
              <SelectTrigger className="h-9 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="damage" className="text-xs">damage d ∈ [0,1]</SelectItem>
                <SelectItem value="crack"  className="text-xs">crack mask (d &gt; threshold)</SelectItem>
                <SelectItem value="psi"    className="text-xs">driving energy ψ⁺</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">load increment / step</span>
              <span className="tabular-nums">{loadRate.toFixed(3)}</span>
            </div>
            <Slider value={[loadRate]} min={0.005} max={0.1} step={0.005}
                    onValueChange={(v) => setLoadRate(v[0])} />
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">grid resolution</span>
              <span className="tabular-nums">{resPow}²</span>
            </div>
            <Slider value={[resPow]} min={32} max={160} step={16}
                    onValueChange={(v) => { setResPow(v[0]); rebuild(v[0], ellMul); }} />
          </div>

          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">ℓ / h (smear width)</span>
              <span className="tabular-nums">{ellMul}</span>
            </div>
            <Slider value={[ellMul]} min={2} max={10} step={1}
                    onValueChange={(v) => { setEllMul(v[0]); rebuild(resPow, v[0]); }} />
          </div>

          <div className="rounded border border-border/60 bg-background/40 p-2 grid grid-cols-3 gap-2 text-[11px]">
            <div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">cracked cells</div>
              <div className="tabular-nums">{s.crackSizeHistory.at(-1) ?? 0}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">mean d</div>
              <div className="tabular-nums">{(s.meanDamageHistory.at(-1) ?? 0).toFixed(4)}</div>
            </div>
            <div>
              <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">peak d</div>
              <div className="tabular-nums">
                {Array.from(s.d).reduce((a, b) => Math.max(a, b), 0).toFixed(3)}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Crack growth over time */}
      <div className="rounded border border-border/60 bg-background/40 p-2">
        <div className="mb-1 flex justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          <span>crack growth (cells &gt; threshold)</span>
          <span>max {chart.maxLabel}</span>
        </div>
        <svg viewBox={`0 0 ${chart.W} ${chart.H}`} className="h-20 w-full">
          <path d={chart.path} fill="none" stroke="hsl(var(--primary))" strokeWidth={1.5} />
        </svg>
      </div>
    </div>
  );
}
