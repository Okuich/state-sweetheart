/**
 * Topology → Physics OS bridge.
 *
 * Runs analyzeTopology on a chosen preset and previews how its priors map
 * into SimParams (timestep, damping, contact stiffness, sub-step / refinement
 * recommendations). One click applies the patch to the live simulation.
 */

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh, type RefinementSeed } from "@/lib/meshing/octree";
import {
  analyzeTopology, FEATURE_LABELS,
  type TopologyResult, type FeatureClass,
} from "@/lib/topology";
import type { SimParams } from "@/components/PhysicsCanvas";

interface Preset { label: string; seeds: RefinementSeed[] }

const PRESETS: Preset[] = [
  { label: "block · plain", seeds: [] },
  { label: "overhang shelf", seeds: [{ kind: "overhang", center: [0, 0.55, 0], radius: 0.35, weight: 1 }] },
  { label: "thin shell", seeds: [{ kind: "thin_wall", center: [0, 0, 0], radius: 0.6, weight: 1 }] },
  { label: "stress notch", seeds: [{ kind: "sharp", center: [0.7, 0, 0], radius: 0.25, weight: 1 }] },
  { label: "thermal hotspot", seeds: [{ kind: "hotspot", center: [0, 0, 0], radius: 0.3, weight: 1 }] },
  { label: "fillet pocket + bore", seeds: [
    { kind: "fillet", center: [-0.4, 0, 0], radius: 0.3, weight: 1 },
    { kind: "hole", center: [0.4, 0, 0], radius: 0.3, weight: 1 },
  ] },
];

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };

const FEATURE_ORDER: FeatureClass[] = [
  "bulk", "boundary", "thin_wall", "overhang",
  "cavity", "stress_concentrator", "thermal_bottleneck", "symmetry_seed",
];

const FEATURE_COLORS: Record<FeatureClass, string> = {
  bulk: "bg-muted-foreground/30",
  boundary: "bg-primary/60",
  thin_wall: "bg-amber-500",
  overhang: "bg-fuchsia-500",
  cavity: "bg-destructive",
  stress_concentrator: "bg-red-500",
  thermal_bottleneck: "bg-orange-500",
  symmetry_seed: "bg-emerald-500",
};

export type SimParamsPatch = Partial<SimParams>;

export interface PriorMapping {
  dtScale: number;
  damping: number;
  contactBeta: number;
  contactsEnabled: boolean;
  adaptiveSubSteps: boolean;
  subSteps: number;
  maxSubSteps: number;
  /** Driver narrative for the UI ("why these values"). */
  rationale: string[];
}

/** Map TopologyResult.priors + features into a SimParams patch. */
export function mapPriorsToSimParams(r: TopologyResult): PriorMapping {
  const p = r.priors;
  const hints = p.refinementHints;
  const maxHint = Math.max(...FEATURE_ORDER.map((c) => hints[c]));
  const subSteps = clamp(Math.round(maxHint * 2), 1, 8);
  const maxSubSteps = clamp(Math.round(maxHint * 4), 4, 16);
  const adaptiveSubSteps = maxHint > 1.3;
  const contactsEnabled = p.contactStiffness > 0.45;

  const m = r.manufacturability;
  const rationale: string[] = [];
  if (m.drivers.stressPenalty > 0.2) rationale.push(`stress concentrators → smaller dt (${p.timestepScale.toFixed(2)}×)`);
  if (m.drivers.thinWallPenalty > 0.2) rationale.push(`thin walls → higher damping (${p.damping.toFixed(2)})`);
  if (m.drivers.overhangPenalty > 0.2) rationale.push(`overhangs → adaptive sub-steps`);
  if (m.drivers.cavityPenalty > 0.2) rationale.push(`cavities → contact β ${p.contactStiffness.toFixed(2)}`);
  if (rationale.length === 0) rationale.push("bulk-dominant geometry → relaxed defaults");

  return {
    dtScale: round(p.timestepScale, 2),
    damping: round(p.damping, 2),
    contactBeta: round(p.contactStiffness, 2),
    contactsEnabled,
    adaptiveSubSteps,
    subSteps,
    maxSubSteps,
    rationale,
  };
}

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }
function round(v: number, dp: number) { const k = 10 ** dp; return Math.round(v * k) / k; }

export function TopologyPriorsPanel({
  currentParams,
  onApplyPatch,
}: {
  currentParams: SimParams;
  onApplyPatch: (patch: SimParamsPatch) => void;
}) {
  const [presetIdx, setPresetIdx] = useState(2);
  const [maxDepth, setMaxDepth] = useState(4);
  const [partitions, setPartitions] = useState(4);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<TopologyResult | null>(null);

  const run = () => {
    setRunning(true);
    setTimeout(() => {
      const mesh = buildOctreeMesh(BBOX, PRESETS[presetIdx].seeds, {
        minDepth: 2, maxDepth, refineThreshold: 0.3,
      });
      setResult(analyzeTopology(mesh, { partitionCount: partitions }));
      setRunning(false);
    }, 0);
  };

  const mapping = useMemo(() => (result ? mapPriorsToSimParams(result) : null), [result]);

  const apply = () => {
    if (!mapping) return;
    onApplyPatch({
      dtScale: mapping.dtScale,
      damping: mapping.damping,
      contactBeta: mapping.contactBeta,
      contactsEnabled: mapping.contactsEnabled,
      adaptiveSubSteps: mapping.adaptiveSubSteps,
      subSteps: mapping.subSteps,
      maxSubSteps: mapping.maxSubSteps,
    });
  };

  return (
    <section className="rounded-xl border border-border bg-card/40 p-5 space-y-5">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold">Topology → Physics OS bridge</h2>
          <p className="text-sm text-muted-foreground">
            Preview adaptive meshing hints, timestep / damping priors, and contact-stiffness suggestions derived from the topology engine, then apply them to the live simulation in one click.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={apply} disabled={!mapping || running}>Apply to simulation</Button>
          <Button onClick={run} disabled={running}>{running ? "Analyzing…" : "Analyze"}</Button>
        </div>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
        {PRESETS.map((p, i) => (
          <button
            key={p.label}
            onClick={() => setPresetIdx(i)}
            className={`rounded-md border px-3 py-2 text-left text-xs transition ${i === presetIdx ? "border-primary bg-primary/10" : "border-border hover:bg-muted/40"}`}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <SliderRow label="Octree max depth" value={maxDepth} min={3} max={6} step={1} onChange={setMaxDepth} fmt={(v) => `${v}`} />
        <SliderRow label="Partitions" value={partitions} min={1} max={8} step={1} onChange={setPartitions} fmt={(v) => `${v}`} />
      </div>

      {result && mapping && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Adaptive meshing hints</h3>
            <RefinementHints r={result} />
            <h3 className="text-sm font-semibold pt-2">Driver rationale</h3>
            <ul className="text-xs text-muted-foreground space-y-1 list-disc pl-4">
              {mapping.rationale.map((r) => <li key={r}>{r}</li>)}
            </ul>
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold">SimParams patch · current → suggested</h3>
            <Diff label="dtScale" current={currentParams.dtScale} suggested={mapping.dtScale} fmt={(v) => v.toFixed(2)} />
            <Diff label="damping" current={currentParams.damping} suggested={mapping.damping} fmt={(v) => v.toFixed(2)} />
            <Diff label="contactBeta" current={currentParams.contactBeta} suggested={mapping.contactBeta} fmt={(v) => v.toFixed(2)} />
            <Diff label="contactsEnabled" current={String(currentParams.contactsEnabled)} suggested={String(mapping.contactsEnabled)} />
            <Diff label="adaptiveSubSteps" current={String(currentParams.adaptiveSubSteps)} suggested={String(mapping.adaptiveSubSteps)} />
            <Diff label="subSteps" current={currentParams.subSteps} suggested={mapping.subSteps} fmt={(v) => `${v}`} />
            <Diff label="maxSubSteps" current={currentParams.maxSubSteps} suggested={mapping.maxSubSteps} fmt={(v) => `${v}`} />
          </div>
        </div>
      )}
    </section>
  );
}

function SliderRow({ label, value, min, max, step, onChange, fmt }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (n: number) => void; fmt: (n: number) => string;
}) {
  return (
    <label className="space-y-1 block">
      <div className="flex justify-between"><span>{label}</span><span className="font-mono text-muted-foreground">{fmt(value)}</span></div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={(v) => onChange(v[0])} />
    </label>
  );
}

function RefinementHints({ r }: { r: TopologyResult }) {
  const max = 2;
  return (
    <div className="space-y-1.5">
      {FEATURE_ORDER.map((c) => {
        const v = r.priors.refinementHints[c];
        const count = r.features.counts[c];
        return (
          <div key={c} className="flex items-center gap-2 text-xs">
            <span className="w-32 text-muted-foreground">{FEATURE_LABELS[c]}</span>
            <div className="flex-1 h-2 rounded bg-muted overflow-hidden">
              <div className={`h-full rounded ${FEATURE_COLORS[c]}`} style={{ width: `${(v / max) * 100}%`, opacity: count > 0 ? 1 : 0.3 }} />
            </div>
            <span className="w-12 text-right font-mono">{v.toFixed(2)}×</span>
            <span className="w-10 text-right font-mono text-muted-foreground/70">n={count}</span>
          </div>
        );
      })}
    </div>
  );
}

function Diff<T extends string | number>({ label, current, suggested, fmt }: {
  label: string; current: T; suggested: T; fmt?: (v: T) => string;
}) {
  const f = (v: T) => (fmt ? fmt(v) : String(v));
  const changed = String(current) !== String(suggested);
  return (
    <div className="flex items-center gap-2 text-xs font-mono border-b border-border/40 py-1">
      <span className="w-36 text-muted-foreground">{label}</span>
      <span className="w-20 text-right">{f(current)}</span>
      <span className="text-muted-foreground/60">→</span>
      <span className={`w-20 text-right ${changed ? "text-primary font-semibold" : "text-muted-foreground"}`}>{f(suggested)}</span>
      {changed && <span className="text-[10px] uppercase tracking-wide text-primary/80">patch</span>}
    </div>
  );
}
