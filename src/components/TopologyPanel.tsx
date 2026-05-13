import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh, type RefinementSeed } from "@/lib/meshing/octree";
import {
  analyzeTopology, cosine, FEATURE_LABELS,
  type TopologyResult, type FeatureClass,
} from "@/lib/topology";

interface Preset {
  label: string;
  seeds: RefinementSeed[];
}

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

const PARTITION_COLORS = ["bg-primary", "bg-accent", "bg-emerald-500", "bg-fuchsia-500", "bg-amber-500", "bg-sky-500", "bg-teal-500", "bg-rose-500"];

interface CorpusEntry { id: string; preset: number; result: TopologyResult }

export function TopologyPanel() {
  const [presetIdx, setPresetIdx] = useState(1);
  const [maxDepth, setMaxDepth] = useState(4);
  const [partitions, setPartitions] = useState(4);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<TopologyResult | null>(null);
  const [corpus, setCorpus] = useState<CorpusEntry[]>([]);

  const run = () => {
    setRunning(true);
    setTimeout(() => {
      const mesh = buildOctreeMesh(BBOX, PRESETS[presetIdx].seeds, {
        minDepth: 2, maxDepth, refineThreshold: 0.3,
      });
      const r = analyzeTopology(mesh, { partitionCount: partitions });
      setResult(r);
      setRunning(false);
    }, 0);
  };

  const indexCorpus = () => {
    setRunning(true);
    setTimeout(() => {
      const entries: CorpusEntry[] = PRESETS.map((p, i) => {
        const mesh = buildOctreeMesh(BBOX, p.seeds, { minDepth: 2, maxDepth, refineThreshold: 0.3 });
        return { id: p.label, preset: i, result: analyzeTopology(mesh, { partitionCount: partitions }) };
      });
      setCorpus(entries);
      setRunning(false);
    }, 0);
  };

  const retrieval = useMemo(() => {
    if (!result || corpus.length === 0) return [];
    return corpus
      .map((c) => ({ id: c.id, similarity: cosine(result.embedding.vector, c.result.embedding.vector) }))
      .sort((a, b) => b.similarity - a.similarity);
  }, [result, corpus]);

  return (
    <section className="rounded-xl border border-border bg-card/40 p-5 space-y-5">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold">Topology Intelligence</h2>
          <p className="text-sm text-muted-foreground">
            Leaf-graph topology, feature classification, manufacturability scoring, structural embeddings, distributed partitioning, and Physics OS priors.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={indexCorpus} disabled={running}>Index corpus</Button>
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

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Topology graph</h3>
            <Stats r={result} />
            <h3 className="text-sm font-semibold pt-2">Feature distribution</h3>
            <FeatureBars r={result} />
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Manufacturability</h3>
            <ManuBars r={result} />
            <h3 className="text-sm font-semibold pt-2">Physics priors</h3>
            <PriorBars r={result} />
            <h3 className="text-sm font-semibold pt-2">Partition layout · imbalance {(result.partition.imbalance * 100).toFixed(1)}% · cut {result.partition.edgeCut}</h3>
            <PartitionStrip r={result} />
            <h3 className="text-sm font-semibold pt-2">Structural embedding (40-d)</h3>
            <EmbeddingStrip vec={result.embedding.vector} slices={result.embedding.slices} />
          </div>
        </div>
      )}

      {retrieval.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Geometry retrieval (cosine)</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
            {retrieval.map((r, i) => (
              <div key={r.id} className="flex items-center gap-2 text-xs">
                <span className="w-6 font-mono text-muted-foreground">#{i + 1}</span>
                <span className="w-40 truncate">{r.id}</span>
                <div className="flex-1 h-1.5 rounded bg-muted">
                  <div className="h-full rounded bg-primary" style={{ width: `${Math.max(0, r.similarity) * 100}%` }} />
                </div>
                <span className="w-12 text-right font-mono">{r.similarity.toFixed(3)}</span>
              </div>
            ))}
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

function Stats({ r }: { r: TopologyResult }) {
  const rows: [string, string][] = [
    ["nodes", `${r.graph.nodes.length}`],
    ["edges", `${r.graph.edges.length}`],
    ["mean valence", `${(r.graph.edges.length * 2 / Math.max(1, r.graph.nodes.length)).toFixed(2)}`],
    ["graph build", `${r.graph.buildMs} ms`],
    ["pipeline total", `${r.totalMs} ms`],
    ["symmetry", `${(r.features.symmetryScore * 100).toFixed(1)}%`],
    ["min wall", `${r.features.minWallThickness.toFixed(3)}`],
    ["avg thin wall", `${r.features.avgThinWallThickness.toFixed(3)}`],
  ];
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs font-mono">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between border-b border-border/40 py-0.5">
          <span className="text-muted-foreground">{k}</span><span>{v}</span>
        </div>
      ))}
    </div>
  );
}

function FeatureBars({ r }: { r: TopologyResult }) {
  const total = Math.max(1, r.graph.nodes.length);
  const order: FeatureClass[] = ["bulk", "boundary", "thin_wall", "overhang", "cavity", "stress_concentrator", "thermal_bottleneck", "symmetry_seed"];
  return (
    <div className="space-y-1.5">
      {order.map((c) => {
        const v = r.features.counts[c];
        return (
          <div key={c} className="flex items-center gap-2 text-xs">
            <span className="w-32 text-muted-foreground">{FEATURE_LABELS[c]}</span>
            <div className="flex-1 h-2 rounded bg-muted">
              <div className={`h-full rounded ${FEATURE_COLORS[c]}`} style={{ width: `${(v / total) * 100}%` }} />
            </div>
            <span className="w-10 text-right font-mono">{v}</span>
          </div>
        );
      })}
    </div>
  );
}

function ManuBars({ r }: { r: TopologyResult }) {
  const m = r.manufacturability;
  const rows: [string, number, string][] = [
    ["feasibility", m.feasibility, "bg-emerald-500"],
    ["machining access", m.machiningAccess, "bg-sky-500"],
    ["support fraction", m.supportFraction, "bg-amber-500"],
    ["thermal distortion", m.thermalDistortionRisk, "bg-orange-500"],
    ["assembly complexity", m.assemblyComplexity, "bg-fuchsia-500"],
  ];
  return (
    <div className="space-y-1.5">
      {rows.map(([k, v, c]) => (
        <div key={k} className="flex items-center gap-2 text-xs">
          <span className="w-36 text-muted-foreground">{k}</span>
          <div className="flex-1 h-2 rounded bg-muted">
            <div className={`h-full rounded ${c}`} style={{ width: `${Math.min(100, v * 100)}%` }} />
          </div>
          <span className="w-10 text-right font-mono">{(v * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

function PriorBars({ r }: { r: TopologyResult }) {
  const p = r.priors;
  return (
    <div className="grid grid-cols-3 gap-2 text-xs">
      <Stat label="timestep" value={p.timestepScale.toFixed(2)} />
      <Stat label="damping" value={p.damping.toFixed(2)} />
      <Stat label="contact k" value={p.contactStiffness.toFixed(2)} />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border px-2 py-1.5">
      <div className="text-muted-foreground text-[10px] uppercase tracking-wide">{label}</div>
      <div className="font-mono text-sm">{value}</div>
    </div>
  );
}

function PartitionStrip({ r }: { r: TopologyResult }) {
  const total = Math.max(1, r.graph.nodes.length);
  return (
    <div className="space-y-2">
      <div className="flex h-6 w-full overflow-hidden rounded border border-border">
        {r.partition.resident.map((res, i) => (
          <div key={i} className={`${PARTITION_COLORS[i % PARTITION_COLORS.length]} flex items-center justify-center text-[10px] font-mono text-background`} style={{ width: `${(res.length / total) * 100}%` }}>
            {res.length}
          </div>
        ))}
      </div>
      <div className="text-xs text-muted-foreground">
        halo {r.partition.halos.reduce((a, b) => a + b.length, 0)} · partitions {r.partition.partitionCount}
      </div>
    </div>
  );
}

function EmbeddingStrip({ vec, slices }: { vec: Float32Array; slices: { curvature: [number, number]; features: [number, number]; structural: [number, number]; manuf: [number, number] } }) {
  const max = Math.max(...Array.from(vec).map(Math.abs), 1e-6);
  const labels = [
    { range: slices.curvature, label: "curvature" },
    { range: slices.features, label: "features" },
    { range: slices.structural, label: "structural" },
    { range: slices.manuf, label: "manuf" },
  ];
  return (
    <div className="space-y-1">
      <div className="flex h-6 gap-px">
        {Array.from(vec).map((v, i) => {
          const t = Math.abs(v) / max;
          const hue = v >= 0 ? 200 : 10;
          return <div key={i} className="flex-1" style={{ backgroundColor: `hsl(${hue} 80% ${20 + t * 50}%)` }} title={`${i}: ${v.toFixed(3)}`} />;
        })}
      </div>
      <div className="flex text-[10px] text-muted-foreground">
        {labels.map((l) => (
          <div key={l.label} className="text-center" style={{ width: `${((l.range[1] - l.range[0]) / vec.length) * 100}%` }}>{l.label}</div>
        ))}
      </div>
    </div>
  );
}
