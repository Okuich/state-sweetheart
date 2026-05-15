import { useMemo, useState } from "react";
import type { TopoNode, Vec3 } from "@/lib/topology/types";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh, type RefinementSeed } from "@/lib/meshing/octree";
import {
  analyzeTopology, cosine, FEATURE_LABELS,
  type TopologyResult, type FeatureClass, type StructuralEmbedding,
} from "@/lib/topology";

interface QueryPreset { label: string; seeds: RefinementSeed[] }

const QUERY_PRESETS: QueryPreset[] = [
  { label: "block · plain", seeds: [] },
  { label: "overhang shelf", seeds: [{ kind: "overhang", center: [0, 0.55, 0], radius: 0.35, weight: 1 }] },
  { label: "thin shell", seeds: [{ kind: "thin_wall", center: [0, 0, 0], radius: 0.6, weight: 1 }] },
  { label: "stress notch", seeds: [{ kind: "sharp", center: [0.7, 0, 0], radius: 0.25, weight: 1 }] },
  { label: "thermal hotspot", seeds: [{ kind: "hotspot", center: [0, 0, 0], radius: 0.3, weight: 1 }] },
  { label: "fillet pocket + bore", seeds: [
    { kind: "fillet", center: [-0.4, 0, 0], radius: 0.3, weight: 1 },
    { kind: "hole", center: [0.4, 0, 0], radius: 0.3, weight: 1 },
  ] },
  { label: "twin overhang rib", seeds: [
    { kind: "overhang", center: [-0.4, 0.5, 0], radius: 0.25, weight: 1 },
    { kind: "thin_wall", center: [0.4, 0.4, 0], radius: 0.25, weight: 1 },
  ] },
  { label: "hot bore", seeds: [
    { kind: "hole", center: [0, 0, 0], radius: 0.35, weight: 1 },
    { kind: "hotspot", center: [0, 0, 0], radius: 0.4, weight: 0.6 },
  ] },
  { label: "sharp shelf", seeds: [
    { kind: "sharp", center: [0.5, 0.5, 0], radius: 0.3, weight: 1 },
    { kind: "overhang", center: [0.0, 0.55, 0], radius: 0.25, weight: 0.7 },
  ] },
  { label: "contact pad", seeds: [
    { kind: "contact", center: [0, -0.7, 0], radius: 0.4, weight: 1 },
  ] },
];

const BBOX = { min: [-1, -1, -1] as [number, number, number], max: [1, 1, 1] as [number, number, number] };

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

const FEATURE_ORDER: FeatureClass[] = [
  "bulk", "boundary", "thin_wall", "overhang", "cavity", "stress_concentrator", "thermal_bottleneck", "symmetry_seed",
];

interface IndexEntry { id: string; result: TopologyResult }

interface SliceScore { curvature: number; features: number; structural: number; manuf: number }

interface RetrievalHit {
  id: string;
  similarity: number;
  slices: SliceScore;
  matchedFeatures: FeatureClass[];
  result: TopologyResult;
}

function sliceCosine(q: Float32Array, c: Float32Array, range: [number, number]): number {
  const a = q.subarray(range[0], range[1]);
  const b = c.subarray(range[0], range[1]);
  return cosine(a as Float32Array, b as Float32Array);
}

function topMatchingFeatures(q: StructuralEmbedding, c: StructuralEmbedding, k = 3): FeatureClass[] {
  const [s, e] = q.slices.features;
  const scored: { f: FeatureClass; agree: number }[] = [];
  for (let i = s; i < e; i++) {
    const qi = q.vector[i];
    const ci = c.vector[i];
    // Reward features both sides activate strongly.
    const agree = Math.min(qi, ci) - 0.25 * Math.abs(qi - ci);
    scored.push({ f: FEATURE_ORDER[i - s], agree });
  }
  scored.sort((a, b) => b.agree - a.agree);
  return scored.filter((x) => x.agree > 0.005).slice(0, k).map((x) => x.f);
}

export function GeometryQueryPanel() {
  const [maxDepth, setMaxDepth] = useState(4);
  const [topK, setTopK] = useState(5);
  const [mode, setMode] = useState<"preset" | "paste">("preset");
  const [presetIdx, setPresetIdx] = useState(1);
  const [paste, setPaste] = useState<string>(JSON.stringify(QUERY_PRESETS[2].seeds, null, 2));
  const [pasteError, setPasteError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [index, setIndex] = useState<IndexEntry[]>([]);
  const [query, setQuery] = useState<{ id: string; result: TopologyResult } | null>(null);

  const buildIndex = () => {
    setRunning(true);
    setTimeout(() => {
      const entries = QUERY_PRESETS.map((p) => {
        const mesh = buildOctreeMesh(BBOX, p.seeds, { minDepth: 2, maxDepth, refineThreshold: 0.3 });
        return { id: p.label, result: analyzeTopology(mesh, { partitionCount: 4 }) };
      });
      setIndex(entries);
      setRunning(false);
    }, 0);
  };

  const runQuery = () => {
    let seeds: RefinementSeed[];
    let id: string;
    if (mode === "preset") {
      seeds = QUERY_PRESETS[presetIdx].seeds;
      id = `query · ${QUERY_PRESETS[presetIdx].label}`;
    } else {
      try {
        const parsed = JSON.parse(paste);
        if (!Array.isArray(parsed)) throw new Error("expected an array of seeds");
        seeds = parsed as RefinementSeed[];
        setPasteError(null);
        id = "query · pasted";
      } catch (e) {
        setPasteError((e as Error).message);
        return;
      }
    }
    setRunning(true);
    setTimeout(() => {
      const mesh = buildOctreeMesh(BBOX, seeds, { minDepth: 2, maxDepth, refineThreshold: 0.3 });
      const result = analyzeTopology(mesh, { partitionCount: 4 });
      setQuery({ id, result });
      setRunning(false);
    }, 0);
  };

  const hits: RetrievalHit[] = useMemo(() => {
    if (!query || index.length === 0) return [];
    const qe = query.result.embedding;
    return index.map((c) => {
      const ce = c.result.embedding;
      const similarity = cosine(qe.vector, ce.vector);
      const slices: SliceScore = {
        curvature: sliceCosine(qe.vector, ce.vector, qe.slices.curvature),
        features: sliceCosine(qe.vector, ce.vector, qe.slices.features),
        structural: sliceCosine(qe.vector, ce.vector, qe.slices.structural),
        manuf: sliceCosine(qe.vector, ce.vector, qe.slices.manuf),
      };
      const matchedFeatures = topMatchingFeatures(qe, ce);
      return { id: c.id, similarity, slices, matchedFeatures, result: c.result };
    }).sort((a, b) => b.similarity - a.similarity).slice(0, topK);
  }, [query, index, topK]);

  return (
    <section className="rounded-xl border border-border bg-card/40 p-5 space-y-5">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold">Geometry Query · Similarity Search</h2>
          <p className="text-sm text-muted-foreground">
            Paste a seed spec or pick a preset, then retrieve the most similar parts from the index with per-feature match highlights.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={buildIndex} disabled={running}>
            {index.length === 0 ? "Build index" : `Re-index · ${index.length}`}
          </Button>
          <Button onClick={runQuery} disabled={running || index.length === 0}>
            {running ? "Searching…" : "Search"}
          </Button>
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        <SliderRow label="Octree max depth" value={maxDepth} min={3} max={6} step={1} onChange={setMaxDepth} fmt={(v) => `${v}`} />
        <SliderRow label="Top-K results" value={topK} min={1} max={QUERY_PRESETS.length} step={1} onChange={setTopK} fmt={(v) => `${v}`} />
      </div>

      <div className="flex gap-2 text-xs">
        <button
          onClick={() => setMode("preset")}
          className={`rounded-md border px-3 py-1.5 ${mode === "preset" ? "border-primary bg-primary/10" : "border-border hover:bg-muted/40"}`}
        >Preset query</button>
        <button
          onClick={() => setMode("paste")}
          className={`rounded-md border px-3 py-1.5 ${mode === "paste" ? "border-primary bg-primary/10" : "border-border hover:bg-muted/40"}`}
        >Paste seeds JSON</button>
      </div>

      {mode === "preset" ? (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
          {QUERY_PRESETS.map((p, i) => (
            <button
              key={p.label}
              onClick={() => setPresetIdx(i)}
              className={`rounded-md border px-3 py-2 text-left text-xs transition ${i === presetIdx ? "border-primary bg-primary/10" : "border-border hover:bg-muted/40"}`}
            >{p.label}</button>
          ))}
        </div>
      ) : (
        <div className="space-y-1">
          <textarea
            value={paste}
            onChange={(e) => setPaste(e.target.value)}
            spellCheck={false}
            rows={8}
            className="w-full rounded-md border border-border bg-background/60 p-2 font-mono text-[11px]"
            placeholder='[{"kind":"thin_wall","center":[0,0,0],"radius":0.5,"weight":1}]'
          />
          {pasteError && <div className="text-xs text-destructive">JSON error: {pasteError}</div>}
          <div className="text-[10px] text-muted-foreground">
            Schema: array of {`{ kind: "hole"|"fillet"|"sharp"|"hotspot"|"overhang"|"thin_wall"|"contact", center:[x,y,z], radius, weight }`}
          </div>
        </div>
      )}

      {query && hits.length === 0 && (
        <div className="text-xs text-muted-foreground">No hits — build the index first.</div>
      )}

      {query && hits.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold">Top {hits.length} matches for <span className="font-mono">{query.id}</span></h3>
            <span className="text-[11px] text-muted-foreground font-mono">
              N={query.result.graph.nodes.length} · pipeline {query.result.totalMs}ms
            </span>
          </div>
          <div className="space-y-2">
            {hits.map((h, i) => (
              <HitRow key={h.id} rank={i + 1} hit={h} query={query.result} />
            ))}
          </div>
        </div>
      )}
    </section>
  );
}

function HitRow({ rank, hit, query }: { rank: number; hit: RetrievalHit; query: TopologyResult }) {
  const total = Math.max(1, hit.result.graph.nodes.length);
  const qTotal = Math.max(1, query.graph.nodes.length);
  const matched = new Set(hit.matchedFeatures);
  const [showPreview, setShowPreview] = useState(rank === 1);
  return (
    <div className="rounded-lg border border-border bg-muted/10 p-3 space-y-2">
      <div className="flex items-center gap-3">
        <span className="w-6 font-mono text-muted-foreground text-xs">#{rank}</span>
        <span className="flex-1 truncate text-sm font-medium">{hit.id}</span>
        <div className="w-32 h-1.5 rounded bg-muted">
          <div className="h-full rounded bg-primary" style={{ width: `${Math.max(0, hit.similarity) * 100}%` }} />
        </div>
        <span className="w-14 text-right font-mono text-sm">{hit.similarity.toFixed(3)}</span>
        <button
          onClick={() => setShowPreview((s) => !s)}
          className="rounded-md border border-border px-2 py-1 text-[10px] hover:bg-muted/40"
          title="Toggle 3D preview of candidate geometry with matched features highlighted"
        >{showPreview ? "Hide 3D" : "Show 3D"}</button>
      </div>

      <div className="grid grid-cols-4 gap-2 text-[10px]">
        <SliceChip label="curvature" v={hit.slices.curvature} />
        <SliceChip label="features" v={hit.slices.features} />
        <SliceChip label="structural" v={hit.slices.structural} />
        <SliceChip label="manuf" v={hit.slices.manuf} />
      </div>

      {showPreview && (
        <CandidatePreview3D nodes={hit.result.graph.nodes} matched={matched} />
      )}

      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
          Matching features {matched.size > 0 && (
            <span className="ml-1">— {Array.from(matched).map((f) => FEATURE_LABELS[f]).join(" · ")}</span>
          )}
        </div>
        {FEATURE_ORDER.map((f) => {
          const q = query.features.counts[f] / qTotal;
          const c = hit.result.features.counts[f] / total;
          const isMatch = matched.has(f);
          return (
            <div key={f} className={`flex items-center gap-2 text-[11px] rounded px-1 ${isMatch ? "bg-primary/10 ring-1 ring-primary/40" : ""}`}>
              <span className={`w-32 ${isMatch ? "text-foreground font-medium" : "text-muted-foreground"}`}>
                {FEATURE_LABELS[f]}
              </span>
              <div className="flex-1 grid grid-cols-2 gap-1">
                <div className="h-1.5 rounded bg-muted overflow-hidden">
                  <div className={`h-full ${FEATURE_COLORS[f]} opacity-60`} style={{ width: `${Math.min(100, q * 100)}%` }} title={`query ${(q * 100).toFixed(1)}%`} />
                </div>
                <div className="h-1.5 rounded bg-muted overflow-hidden">
                  <div className={`h-full ${FEATURE_COLORS[f]}`} style={{ width: `${Math.min(100, c * 100)}%` }} title={`candidate ${(c * 100).toFixed(1)}%`} />
                </div>
              </div>
              <span className="w-20 text-right font-mono text-muted-foreground">
                {(q * 100).toFixed(0)}→{(c * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
        <div className="flex gap-3 text-[10px] text-muted-foreground pl-32">
          <span>← query</span><span>candidate →</span>
        </div>
      </div>
    </div>
  );
}

const FEATURE_HEX: Record<FeatureClass, string> = {
  bulk: "#cbd5e1",
  boundary: "#94a3b8",
  thin_wall: "#f59e0b",
  overhang: "#d946ef",
  cavity: "#ef4444",
  stress_concentrator: "#ef4444",
  thermal_bottleneck: "#fb923c",
  symmetry_seed: "#10b981",
};

interface ProjectionViewProps {
  nodes: TopoNode[];
  matched: Set<FeatureClass>;
  hAxis: 0 | 1 | 2;
  vAxis: 0 | 1 | 2;
  dAxis: 0 | 1 | 2;
  label: string;
}

function ProjectionView({ nodes, matched, hAxis, vAxis, dAxis, label }: ProjectionViewProps) {
  const size = 180;
  const pad = 8;
  if (nodes.length === 0) return null;
  let minH = Infinity, maxH = -Infinity, minV = Infinity, maxV = -Infinity;
  let minD = Infinity, maxD = -Infinity, maxR = 0;
  for (const n of nodes) {
    const c = n.center as Vec3;
    if (c[hAxis] < minH) minH = c[hAxis];
    if (c[hAxis] > maxH) maxH = c[hAxis];
    if (c[vAxis] < minV) minV = c[vAxis];
    if (c[vAxis] > maxV) maxV = c[vAxis];
    if (c[dAxis] < minD) minD = c[dAxis];
    if (c[dAxis] > maxD) maxD = c[dAxis];
    if (n.radius > maxR) maxR = n.radius;
  }
  const spanH = Math.max(1e-6, maxH - minH + 2 * maxR);
  const spanV = Math.max(1e-6, maxV - minV + 2 * maxR);
  const spanD = Math.max(1e-6, maxD - minD);
  const scale = (size - 2 * pad) / Math.max(spanH, spanV);
  const cx0 = pad + (size - 2 * pad - spanH * scale) / 2 - (minH - maxR) * scale;
  const cy0 = pad + (size - 2 * pad - spanV * scale) / 2 - (minV - maxR) * scale;
  const order = nodes.map((_, i) => i).sort((a, b) => nodes[a].center[dAxis] - nodes[b].center[dAxis]);

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} className="rounded-md border border-border bg-background/40">
        {order.map((i) => {
          const n = nodes[i];
          const c = n.center as Vec3;
          const x = cx0 + c[hAxis] * scale;
          const y = size - (cy0 + c[vAxis] * scale);
          const r = Math.max(1, n.radius * scale);
          const isMatch = matched.has(n.feature);
          const depth = (c[dAxis] - minD) / spanD;
          if (!isMatch) {
            const alpha = 0.08 + 0.18 * depth;
            return (
              <rect key={i} x={x - r} y={y - r} width={r * 2} height={r * 2} fill={`rgba(148,163,184,${alpha.toFixed(3)})`} />
            );
          }
          const col = FEATURE_HEX[n.feature];
          return (
            <g key={i}>
              <rect x={x - r * 1.6} y={y - r * 1.6} width={r * 3.2} height={r * 3.2} fill={col} opacity={0.25} />
              <rect x={x - r} y={y - r} width={r * 2} height={r * 2} fill={col} stroke="white" strokeWidth={0.5} opacity={0.9} />
            </g>
          );
        })}
      </svg>
      <span className="text-[9px] uppercase tracking-wide text-muted-foreground font-mono">{label}</span>
    </div>
  );
}

type ViewMode = "xy" | "xz" | "zy" | "all";
const VIEW_MODES: { id: ViewMode; label: string }[] = [
  { id: "all", label: "All" },
  { id: "xy", label: "XY" },
  { id: "xz", label: "XZ" },
  { id: "zy", label: "ZY" },
];

function CandidatePreview3D({ nodes, matched }: { nodes: TopoNode[]; matched: Set<FeatureClass> }) {
  const [mode, setMode] = useState<ViewMode>("all");
  const matchedCount = useMemo(
    () => nodes.reduce((s, n) => s + (matched.has(n.feature) ? 1 : 0), 0),
    [nodes, matched],
  );
  const big = mode !== "all";
  return (
    <div className="rounded-md border border-border bg-background/30 p-2 space-y-2">
      <div className="flex items-center justify-between text-[10px] text-muted-foreground gap-2">
        <span className="uppercase tracking-wide">3D preview · matched features highlighted</span>
        <div className="flex items-center gap-1">
          {VIEW_MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              onClick={() => setMode(m.id)}
              className={`px-1.5 py-0.5 rounded border text-[9px] font-mono uppercase tracking-wide transition ${
                mode === m.id
                  ? "border-primary bg-primary/15 text-foreground"
                  : "border-border hover:bg-muted/40"
              }`}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>
      <div className="text-[10px] text-muted-foreground font-mono text-right">
        {matchedCount}/{nodes.length} leaves · {Array.from(matched).map((f) => FEATURE_LABELS[f]).join(" · ") || "—"}
      </div>
      <div className="flex flex-wrap gap-3 justify-center">
        {(mode === "all" || mode === "xy") && (
          <ProjectionView nodes={nodes} matched={matched} hAxis={0} vAxis={1} dAxis={2} label="XY · front" size={big ? 220 : 140} />
        )}
        {(mode === "all" || mode === "xz") && (
          <ProjectionView nodes={nodes} matched={matched} hAxis={0} vAxis={2} dAxis={1} label="XZ · top" size={big ? 220 : 140} />
        )}
        {(mode === "all" || mode === "zy") && (
          <ProjectionView nodes={nodes} matched={matched} hAxis={2} vAxis={1} dAxis={0} label="ZY · side" size={big ? 220 : 140} />
        )}
      </div>
      {matched.size > 0 && (
        <div className="flex flex-wrap gap-2 text-[10px] justify-center pt-1">
          {Array.from(matched).map((f) => (
            <span key={f} className="inline-flex items-center gap-1 font-mono">
              <span className="inline-block w-2.5 h-2.5 rounded-sm" style={{ background: FEATURE_HEX[f] }} />
              {FEATURE_LABELS[f]}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function SliceChip({ label, v }: { label: string; v: number }) {
  const pct = Math.max(0, v) * 100;
  return (
    <div className="rounded border border-border bg-background/40 px-2 py-1">
      <div className="flex items-center justify-between">
        <span className="text-muted-foreground uppercase tracking-wide">{label}</span>
        <span className="font-mono">{v.toFixed(2)}</span>
      </div>
      <div className="mt-1 h-1 rounded bg-muted">
        <div className="h-full rounded bg-primary/70" style={{ width: `${pct}%` }} />
      </div>
    </div>
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
