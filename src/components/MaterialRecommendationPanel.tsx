/**
 * Material Recommendation Engine panel.
 *
 *   - constraint sliders (yield, stiffness, density, Tmax, cost, designStress)
 *   - environment + fabrication chips
 *   - multi-objective weight sliders
 *   - ranked candidates table with feasibility, safety, lifecycle
 *   - "substitute for…" view that surfaces the nearest metric-space
 *     alternatives to a chosen baseline material
 *
 * Pulls live rows from `public.materials` when the table has data,
 * and falls back to BUILTIN_MATERIALS so the engine renders on a
 * fresh project.
 */
import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { supabase } from "@/integrations/supabase/client";
import {
  BUILTIN_MATERIALS, buildIndex, recommend, substitute,
  DEFAULT_WEIGHTS,
  type CorpusIndex, type DesignConstraints, type Environment,
  type Fabrication, type MaterialRecord, type MaterialScore,
  type ObjectiveWeights,
} from "@/lib/materials";

const ENVIRONMENTS: Environment[] = ["marine", "high_temp", "cryogenic", "uv_exposed", "chemical", "abrasive"];
const FABRICATIONS: Fabrication[] = [
  "machining", "casting", "injection_molding", "additive_dmls",
  "additive_fdm", "sheet_forming", "extrusion", "layup", "sintering",
];

type DbMaterial = {
  id: string;
  name: string;
  family: string;
  density_kg_m3: number | null;
  youngs_modulus_gpa: number | null;
  yield_strength_mpa: number | null;
  ultimate_strength_mpa: number | null;
  thermal_conductivity_w_mk: number | null;
  cost_usd_per_kg: number | null;
  metadata: Record<string, unknown> | null;
};

function mapFamily(f: string): MaterialRecord["family"] {
  const v = f.toLowerCase();
  if (v.includes("metal") || v.includes("alloy") || v.includes("steel")) return "metal";
  if (v.includes("polymer") || v.includes("plastic")) return "polymer";
  if (v.includes("ceramic")) return "ceramic";
  if (v.includes("composite")) return "composite";
  if (v.includes("elastomer") || v.includes("rubber")) return "elastomer";
  return "other";
}

function fromDb(row: DbMaterial): MaterialRecord | null {
  if (!row.yield_strength_mpa || !row.youngs_modulus_gpa) return null;
  const md = row.metadata ?? {};
  const pick = (k: string, fallback: number) =>
    typeof md[k] === "number" ? (md[k] as number) : fallback;
  return {
    id: row.id,
    name: row.name,
    family: mapFamily(row.family),
    density: (row.density_kg_m3 ?? 7800) / 1000,
    youngsModulus: row.youngs_modulus_gpa,
    yieldStrength: row.yield_strength_mpa,
    ultimateStrength: row.ultimate_strength_mpa ?? row.yield_strength_mpa * 1.2,
    fatigueLimit: pick("fatigueLimit", row.yield_strength_mpa * 0.4),
    thermalConductivity: row.thermal_conductivity_w_mk ?? pick("thermalConductivity", 30),
    thermalExpansion: pick("thermalExpansion", 12),
    maxServiceTempC: pick("maxServiceTempC", 300),
    costPerKg: row.cost_usd_per_kg ?? pick("costPerKg", 5),
    embodiedCO2: pick("embodiedCO2", 4),
    corrosionResistance: pick("corrosionResistance", 0.6),
    weatherResistance: pick("weatherResistance", 0.6),
    fabrication: Array.isArray(md.fabrication) ? (md.fabrication as Fabrication[]) : ["machining"],
  };
}

export function MaterialRecommendationPanel() {
  const [corpus, setCorpus] = useState<MaterialRecord[]>(BUILTIN_MATERIALS);
  const [source, setSource] = useState<"built-in" | "db">("built-in");

  // Pull DB rows once on mount; merge with built-ins.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const { data, error } = await supabase
        .from("materials")
        .select("id,name,family,density_kg_m3,youngs_modulus_gpa,yield_strength_mpa,ultimate_strength_mpa,thermal_conductivity_w_mk,cost_usd_per_kg,metadata");
      if (error || !data || data.length === 0 || cancelled) return;
      const dbMats = (data as DbMaterial[])
        .map(fromDb)
        .filter((m): m is MaterialRecord => m !== null);
      if (dbMats.length === 0) return;
      setCorpus([...BUILTIN_MATERIALS, ...dbMats]);
      setSource("db");
    })();
    return () => { cancelled = true; };
  }, []);

  const index: CorpusIndex = useMemo(() => buildIndex(corpus), [corpus]);

  const [constraints, setConstraints] = useState<DesignConstraints>({
    minYield: 250,
    minStiffness: 50,
    maxDensity: 8,
    minServiceTempC: 100,
    maxCostPerKg: 30,
    designStress: 120,
    designCycles: 1_000_000,
    environment: [],
    fabrication: [],
  });
  const [weights, setWeights] = useState<ObjectiveWeights>(DEFAULT_WEIGHTS);
  const [subFor, setSubFor] = useState<string | null>(null);

  const ranked: MaterialScore[] = useMemo(
    () => recommend(index, constraints, weights),
    [index, constraints, weights],
  );
  const subs: MaterialScore[] = useMemo(
    () => subFor ? substitute(index, subFor, constraints, weights, 5) : [],
    [index, subFor, constraints, weights],
  );

  const feasible = ranked.filter((r) => r.feasible).length;

  const upd = <K extends keyof DesignConstraints>(k: K, v: DesignConstraints[K]) =>
    setConstraints((c) => ({ ...c, [k]: v }));
  const updW = <K extends keyof ObjectiveWeights>(k: K, v: ObjectiveWeights[K]) =>
    setWeights((w) => ({ ...w, [k]: v }));

  const toggleEnv = (e: Environment) =>
    upd("environment", constraints.environment?.includes(e)
      ? constraints.environment.filter((x) => x !== e)
      : [...(constraints.environment ?? []), e]);
  const toggleFab = (f: Fabrication) =>
    upd("fabrication", constraints.fabrication?.includes(f)
      ? constraints.fabrication.filter((x) => x !== f)
      : [...(constraints.fabrication ?? []), f]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Material · metric intelligence layer
          <Badge variant="outline" className="text-[10px]">
            {corpus.length} materials · {source}
          </Badge>
          <Badge variant="outline" className="text-[10px]">
            {feasible} feasible
          </Badge>
        </h3>
        {subFor && (
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setSubFor(null)}>
            close substitutes
          </Button>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="space-y-3 rounded border border-border/60 bg-background/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            design constraints
          </div>
          <CSlider label="min yield" unit="MPa" min={0} max={1500} step={10}
            value={constraints.minYield ?? 0} onChange={(v) => upd("minYield", v)} />
          <CSlider label="min stiffness" unit="GPa" min={0} max={400} step={5}
            value={constraints.minStiffness ?? 0} onChange={(v) => upd("minStiffness", v)} />
          <CSlider label="max density" unit="g/cm³" min={0.5} max={10} step={0.1}
            value={constraints.maxDensity ?? 10} onChange={(v) => upd("maxDensity", v)} />
          <CSlider label="min service T" unit="°C" min={0} max={1500} step={10}
            value={constraints.minServiceTempC ?? 0} onChange={(v) => upd("minServiceTempC", v)} />
          <CSlider label="max cost" unit="$/kg" min={0} max={120} step={1}
            value={constraints.maxCostPerKg ?? 120} onChange={(v) => upd("maxCostPerKg", v)} />
          <CSlider label="design stress" unit="MPa" min={0} max={800} step={5}
            value={constraints.designStress ?? 0} onChange={(v) => upd("designStress", v)} />

          <div>
            <div className="mb-1 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">environment</div>
            <div className="flex flex-wrap gap-1">
              {ENVIRONMENTS.map((e) => (
                <Chip key={e} on={!!constraints.environment?.includes(e)} onClick={() => toggleEnv(e)}>{e}</Chip>
              ))}
            </div>
          </div>
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">fabrication</div>
            <div className="flex flex-wrap gap-1">
              {FABRICATIONS.map((f) => (
                <Chip key={f} on={!!constraints.fabrication?.includes(f)} onClick={() => toggleFab(f)}>
                  {f.replace(/_/g, " ")}
                </Chip>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-3 rounded border border-border/60 bg-background/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            objective weights (normalized)
          </div>
          <CSlider label="performance" min={0} max={1} step={0.05}
            value={weights.performance} onChange={(v) => updW("performance", v)} />
          <CSlider label="cost" min={0} max={1} step={0.05}
            value={weights.cost} onChange={(v) => updW("cost", v)} />
          <CSlider label="weight (mass)" min={0} max={1} step={0.05}
            value={weights.weight} onChange={(v) => updW("weight", v)} />
          <CSlider label="sustainability" min={0} max={1} step={0.05}
            value={weights.sustainability} onChange={(v) => updW("sustainability", v)} />
          <CSlider label="fatigue" min={0} max={1} step={0.05}
            value={weights.fatigue} onChange={(v) => updW("fatigue", v)} />

          <div className="pt-2 text-[10px] text-muted-foreground">
            Stress safety, thermal safety, and Basquin lifecycle are computed
            per material against current design stress and service temperature.
          </div>
        </div>
      </div>

      {/* Substitutes view */}
      {subFor && (
        <div className="rounded border border-emerald-500/40 bg-emerald-500/5 p-2">
          <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-emerald-300">
            substitutes for {index.records.find((r) => r.id === subFor)?.name ?? subFor}
          </div>
          <ScoreTable rows={subs} showSimilarity onSub={setSubFor} />
        </div>
      )}

      <div>
        <div className="mb-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          ranked candidates
        </div>
        <ScoreTable rows={ranked.slice(0, 12)} onSub={setSubFor} />
      </div>
    </div>
  );
}

function CSlider({
  label, unit, value, onChange, min, max, step,
}: { label: string; unit?: string; value: number; onChange: (v: number) => void;
     min: number; max: number; step: number }) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground">
        <span>{label}</span>
        <span className="text-primary tabular-nums">
          {value.toFixed(step < 1 ? 2 : 0)}{unit && <span className="text-muted-foreground ml-0.5">{unit}</span>}
        </span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step}
        onValueChange={([v]) => onChange(v)} />
    </div>
  );
}

function Chip({ on, children, onClick }: {
  on: boolean; children: React.ReactNode; onClick: () => void;
}) {
  return (
    <button onClick={onClick}
      className={`rounded border px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] transition-colors ${
        on
          ? "border-primary bg-primary/15 text-primary"
          : "border-border/60 text-muted-foreground hover:border-foreground/40 hover:text-foreground"
      }`}>
      {children}
    </button>
  );
}

function fmtCycles(n: number): string {
  if (!Number.isFinite(n)) return "∞";
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}G`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return `${Math.round(n)}`;
}

function ScoreTable({
  rows, showSimilarity, onSub,
}: { rows: MaterialScore[]; showSimilarity?: boolean; onSub?: (id: string) => void }) {
  if (rows.length === 0) {
    return (
      <div className="rounded border border-border/60 bg-background/40 p-3 text-[11px] italic text-muted-foreground">
        no candidates
      </div>
    );
  }
  return (
    <div className="space-y-1">
      {rows.map((r) => {
        const m = r.material;
        const tone = r.feasible
          ? "border-border/60 bg-background/40"
          : "border-destructive/40 bg-destructive/5";
        return (
          <div key={m.id}
            className={`rounded border px-2 py-1.5 text-[11px] ${tone}`}>
            <div className="grid grid-cols-[1.4fr_auto_auto_auto_auto] items-center gap-2">
              <div className="min-w-0">
                <div className="truncate font-medium">
                  {m.name}
                  <span className="ml-2 text-[10px] text-muted-foreground">{m.family}</span>
                </div>
                <div className="truncate text-[10px] text-muted-foreground">
                  σy {m.yieldStrength} MPa · E {m.youngsModulus} GPa · ρ {m.density} · ${m.costPerKg}/kg · Tmax {m.maxServiceTempC}°C
                </div>
              </div>
              <Bar v={r.score} w={70} label="score" />
              <span className="font-mono text-[10px] tabular-nums text-muted-foreground"
                title={`stress ${r.stressSafety === Infinity ? "∞" : r.stressSafety.toFixed(2)} · thermal ${r.thermalSafety.toFixed(2)}`}>
                SF {Number.isFinite(r.stressSafety) ? r.stressSafety.toFixed(2) : "∞"}
              </span>
              <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
                life {fmtCycles(r.lifecycleCycles)}
              </span>
              <div className="flex items-center gap-1">
                {showSimilarity && r.similarity != null && (
                  <span className="rounded border border-emerald-500/40 px-1.5 py-0.5 font-mono text-[10px] text-emerald-300">
                    sim {r.similarity.toFixed(2)}
                  </span>
                )}
                {onSub && (
                  <button onClick={() => onSub(m.id)}
                    className="rounded border border-border/60 px-1.5 py-0.5 text-[9px] uppercase tracking-[0.14em] hover:border-foreground/40 hover:text-foreground">
                    substitutes
                  </button>
                )}
              </div>
            </div>
            <div className="mt-1 grid grid-cols-5 gap-1">
              {(["performance", "cost", "weight", "sustainability", "fatigue"] as const).map((k) => (
                <Bar key={k} v={r.breakdown[k]} label={k} />
              ))}
            </div>
            {!r.feasible && (
              <div className="mt-1 text-[10px] text-destructive/90">
                · {r.violations.slice(0, 3).join(" · ")}
                {r.violations.length > 3 ? ` · +${r.violations.length - 3} more` : ""}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function Bar({ v, label, w }: { v: number; label: string; w?: number }) {
  const pct = Math.max(2, Math.round(Math.max(0, Math.min(1, v)) * 100));
  return (
    <div className="min-w-0" style={w ? { width: w } : undefined}>
      <div className="flex items-baseline justify-between text-[9px] uppercase tracking-[0.12em] text-muted-foreground">
        <span className="truncate">{label}</span>
        <span className="tabular-nums">{(v * 100).toFixed(0)}</span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded bg-background/60">
        <div className="h-full bg-primary/70" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
