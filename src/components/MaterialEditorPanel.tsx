import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  ACTIVE_PARAMS, DEFAULT_PARAMS, gradient, strainEnergy,
  type MaterialKind, type MaterialParams,
} from "@/lib/materialModel";

const KINDS: { value: MaterialKind; label: string }[] = [
  { value: "hookean",      label: "Hookean (linear elastic)" },
  { value: "neo-hookean",  label: "Neo-Hookean (nonlinear elastic)" },
  { value: "viscoelastic", label: "Viscoelastic (Maxwell)" },
  { value: "plastic",      label: "Plastic (J2 + hardening)" },
  { value: "fracture",     label: "Fracture (phase-field)" },
];

const PARAM_META: Record<keyof MaterialParams, { label: string; min: number; max: number; step: number; unit?: string }> = {
  mu:          { label: "μ  shear modulus",       min: 1,    max: 1e5, step: 1,    unit: "Pa" },
  lambda:      { label: "λ  Lamé 2",              min: 0,    max: 1e5, step: 1,    unit: "Pa" },
  eta:         { label: "η  viscosity",           min: 0,    max: 1e3, step: 0.1,  unit: "Pa·s" },
  tau:         { label: "τ  relaxation time",     min: 1e-3, max: 1,   step: 1e-3, unit: "s" },
  yieldStress: { label: "σ_y  yield stress",      min: 1,    max: 1e4, step: 1,    unit: "Pa" },
  hardening:   { label: "H  strain hardening",    min: 0,    max: 5e3, step: 1,    unit: "Pa" },
  epsFrac:     { label: "ε_frac  fracture strain",min: 1e-3, max: 0.5, step: 1e-3 },
  Gc:          { label: "G_c  fracture energy",   min: 1,    max: 500, step: 1,    unit: "J/m²" },
};

export function MaterialEditorPanel() {
  const [kind, setKind] = useState<MaterialKind>("hookean");
  const [params, setParams] = useState<MaterialParams>(DEFAULT_PARAMS);
  const [probe, setProbe] = useState(0.05);

  const active = ACTIVE_PARAMS[kind];

  const setOne = (k: keyof MaterialParams, v: number) =>
    setParams((p) => ({ ...p, [k]: v }));

  const reset = () => setParams(DEFAULT_PARAMS);

  const grad = useMemo(() => gradient(kind, params, probe), [kind, params, probe]);

  const curve = useMemo(() => {
    const W = 360, H = 110, N = 80;
    const epsMax = 0.3;
    const samples = Array.from({ length: N + 1 }, (_, i) => {
      const e = -epsMax + (2 * epsMax * i) / N;
      return { e, psi: strainEnergy(kind, params, e) };
    });
    const finite = samples.filter((s) => Number.isFinite(s.psi));
    const psiMax = Math.max(1e-9, ...finite.map((s) => s.psi));
    const pts = samples.map((s) => {
      const x = ((s.e + epsMax) / (2 * epsMax)) * W;
      const y = H - (Math.min(s.psi, psiMax) / psiMax) * (H - 6) - 3;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    const probeX = ((probe + epsMax) / (2 * epsMax)) * W;
    return { W, H, path: `M${pts.join(" L")}`, probeX };
  }, [kind, params, probe]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Material editor
        </h3>
        <Button variant="ghost" size="sm" onClick={reset} className="h-7 px-2 text-xs">
          reset
        </Button>
      </div>

      <Select value={kind} onValueChange={(v) => setKind(v as MaterialKind)}>
        <SelectTrigger className="h-9 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {KINDS.map((k) => (
            <SelectItem key={k.value} value={k.value} className="text-xs">
              {k.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      {/* Strain-energy preview */}
      <div className="rounded border border-border/60 bg-background/40 p-2">
        <div className="mb-1 flex justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          <span>ψ(ε) preview</span>
          <span>ε ∈ [-0.3, 0.3]</span>
        </div>
        <svg viewBox={`0 0 ${curve.W} ${curve.H}`} className="h-24 w-full">
          <line x1={curve.W / 2} x2={curve.W / 2} y1={0} y2={curve.H}
                stroke="hsl(var(--border))" strokeWidth={1} />
          <path d={curve.path} fill="none"
                stroke="hsl(var(--primary))" strokeWidth={1.5} />
          <line x1={curve.probeX} x2={curve.probeX} y1={0} y2={curve.H}
                stroke="hsl(var(--primary))" strokeOpacity={0.4} strokeDasharray="2 2" />
        </svg>
        <div className="mt-2 space-y-1">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            <span>probe ε</span>
            <span>{probe.toFixed(3)}</span>
          </div>
          <Slider value={[probe]} min={-0.3} max={0.3} step={0.005}
                  onValueChange={(v) => setProbe(v[0])} />
        </div>
      </div>

      {/* Parameter sliders for the active model */}
      <div className="space-y-3">
        {active.map((k) => {
          const meta = PARAM_META[k];
          const v = params[k];
          return (
            <div key={k} className="space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-muted-foreground">{meta.label}</span>
                <span className="tabular-nums">
                  {v.toFixed(v < 1 ? 4 : 1)}{meta.unit ? ` ${meta.unit}` : ""}
                </span>
              </div>
              <Slider value={[v]} min={meta.min} max={meta.max} step={meta.step}
                      onValueChange={(arr) => setOne(k, arr[0])} />
            </div>
          );
        })}
      </div>

      {/* Differentiable gradient readout */}
      <div className="rounded border border-border/60 bg-background/40 p-2 text-[11px]">
        <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          ∂ψ/∂θ at probe ε
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-0.5 tabular-nums">
          {active.map((k) => (
            <div key={k} className="flex justify-between">
              <span className="text-muted-foreground">∂/∂{k}</span>
              <span>{(grad[k] ?? 0).toExponential(2)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
