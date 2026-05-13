import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh } from "@/lib/meshing/octree";
import { buildAdjacency } from "@/lib/meshing/adjacency";
import {
  buildBVH,
  bvhAllPairs,
  bvhFromOctree,
  buildSpatialHash,
  broadphasePairs,
  buildKDTree,
  knn,
  rayQuery,
  pointInSolid,
  analyzeFabrication,
  FAB_CLASS_LABELS,
  type BVH,
  type FabIntelReport,
} from "@/lib/spatial";

interface BenchResult {
  bvh: BVH;
  bvhBuildMs: number;
  hashBuildMs: number;
  hashPairs: number;
  hashPairsMs: number;
  bvhPairs: number;
  bvhPairsMs: number;
  knnQps: number;
  rayQps: number;
  pointInSolidMs: number;
  fab: FabIntelReport;
  leafCount: number;
  primCount: number;
}

const PRESETS = [
  { label: "uniform · 64³", minDepth: 2, maxDepth: 2, seedKind: null },
  { label: "fillet pocket", minDepth: 2, maxDepth: 5, seedKind: "fillet" as const },
  { label: "thermal hotspot", minDepth: 2, maxDepth: 5, seedKind: "hotspot" as const },
  { label: "overhang shell", minDepth: 2, maxDepth: 4, seedKind: "overhang" as const },
];

const FAB_COLORS = ["bg-muted-foreground/30", "bg-primary", "bg-accent", "bg-destructive", "bg-destructive/60"];

export function SpatialAccelPanel() {
  const [presetIdx, setPresetIdx] = useState(1);
  const [cellSize, setCellSize] = useState(0.15);
  const [running, setRunning] = useState(false);
  const [bench, setBench] = useState<BenchResult | null>(null);

  const run = () => {
    setRunning(true);
    requestAnimationFrame(() => {
      const preset = PRESETS[presetIdx];
      const seeds = preset.seedKind
        ? [{ kind: preset.seedKind, center: [0.5, 0.5, 0.5] as [number, number, number], radius: 0.3, weight: 1 }]
        : [];
      const mesh = buildOctreeMesh(
        { min: [0, 0, 0], max: [1, 1, 1] },
        seeds,
        { minDepth: preset.minDepth, maxDepth: preset.maxDepth, refineThreshold: 0.3, maxLeaves: 20_000 },
      );
      const adj = buildAdjacency(mesh);

      const tBvh = performance.now();
      const bvh = bvhFromOctree(mesh);
      const bvhBuildMs = performance.now() - tBvh;

      const tHash = performance.now();
      const lo = new Float32Array(mesh.leaves.length * 3);
      const hi = new Float32Array(mesh.leaves.length * 3);
      for (let i = 0; i < mesh.leaves.length; i++) {
        const n = mesh.nodes[mesh.leaves[i]];
        lo.set(n.bbox.min, i * 3);
        hi.set(n.bbox.max, i * 3);
      }
      const hash = buildSpatialHash(lo, hi, cellSize);
      const hashBuildMs = performance.now() - tHash;

      const tHP = performance.now();
      const hp = broadphasePairs(hash, lo, hi, 200_000);
      const hashPairsMs = performance.now() - tHP;

      const tBP = performance.now();
      const bp = bvhAllPairs(bvh, 200_000);
      const bvhPairsMs = performance.now() - tBP;

      // KDTree on leaf centroids → kNN benchmark
      const cents = new Float32Array(mesh.leaves.length * 3);
      for (let i = 0; i < mesh.leaves.length; i++) {
        const n = mesh.nodes[mesh.leaves[i]];
        cents[i * 3]     = (n.bbox.min[0] + n.bbox.max[0]) / 2;
        cents[i * 3 + 1] = (n.bbox.min[1] + n.bbox.max[1]) / 2;
        cents[i * 3 + 2] = (n.bbox.min[2] + n.bbox.max[2]) / 2;
      }
      const kd = buildKDTree(cents);
      const KNN_N = 2000;
      const tKnn = performance.now();
      for (let i = 0; i < KNN_N; i++) {
        const t = i / KNN_N;
        knn(kd, [t, 1 - t, (t * 7) % 1], 8);
      }
      const knnMs = performance.now() - tKnn;

      const RAY_N = 1000;
      const tRay = performance.now();
      for (let i = 0; i < RAY_N; i++) {
        const a = (i / RAY_N) * Math.PI * 2;
        rayQuery(bvh, [-0.5, 0.5 + 0.3 * Math.cos(a), 0.5 + 0.3 * Math.sin(a)], [1, 0, 0]);
      }
      const rayMs = performance.now() - tRay;

      const tPiS = performance.now();
      let inside = 0;
      for (let i = 0; i < 256; i++) {
        const p: [number, number, number] = [Math.random(), Math.random(), Math.random()];
        if (pointInSolid(bvh, p)) inside++;
      }
      void inside;
      const pisMs = performance.now() - tPiS;

      const fab = analyzeFabrication(mesh, adj);

      setBench({
        bvh,
        bvhBuildMs,
        hashBuildMs,
        hashPairs: hp.length / 2,
        hashPairsMs,
        bvhPairs: bp.length / 2,
        bvhPairsMs,
        knnQps: KNN_N / Math.max(0.001, knnMs / 1000),
        rayQps: RAY_N / Math.max(0.001, rayMs / 1000),
        pointInSolidMs: pisMs / 256,
        fab,
        leafCount: mesh.leaves.length,
        primCount: bvh.primCount,
      });
      setRunning(false);
    });
  };

  const fabHist = useMemo(() => {
    if (!bench) return null;
    const counts = new Array(FAB_CLASS_LABELS.length).fill(0);
    bench.fab.classification.forEach((c) => { counts[c] = (counts[c] ?? 0) + 1; });
    return counts;
  }, [bench]);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            spatial · BVH · KD-tree · spatial hash · fab intel
          </div>
          <h2 className="font-display text-2xl text-foreground">
            GPU-native <span className="text-primary">acceleration</span> for geometry queries.
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {bench ? <>BVH · <span className="text-foreground">{bench.bvh.nodeCount.toLocaleString()} nodes · d{bench.bvh.maxDepth}</span></> : "no acceleration structures"}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {PRESETS.map((p, i) => (
          <button
            key={p.label}
            onClick={() => setPresetIdx(i)}
            className={`text-left rounded-md border px-3 py-2 transition ${
              presetIdx === i
                ? "border-primary bg-primary/10 text-foreground"
                : "border-border bg-background/40 text-muted-foreground hover:text-foreground"
            }`}
          >
            <div className="text-[10px] uppercase tracking-[0.2em]">{p.label}</div>
            <div className="text-[10px] font-mono text-muted-foreground/80">depth {p.minDepth}–{p.maxDepth}</div>
          </button>
        ))}
      </div>

      <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
        <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
          <span className="text-muted-foreground">spatial-hash cell size</span>
          <span className="text-foreground font-mono">{cellSize.toFixed(2)}</span>
        </div>
        <Slider value={[cellSize]} min={0.05} max={0.5} step={0.01} onValueChange={(v) => setCellSize(v[0])} />
      </div>

      <div className="flex gap-2">
        <Button onClick={run} disabled={running} className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "benchmarking…" : bench ? "re-run benchmark" : "build + benchmark"}
        </Button>
      </div>

      {bench && (
        <>
          {/* Build + throughput stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
            <Stat label="BVH primitives" value={bench.primCount.toLocaleString()} />
            <Stat label="BVH build" value={`${bench.bvhBuildMs.toFixed(1)} ms`} />
            <Stat label="hash build" value={`${bench.hashBuildMs.toFixed(1)} ms`} />
            <Stat label="hash buckets" value={`${bench.bvh.primCount}/cell≈${(bench.primCount / 64).toFixed(1)}`} />
            <Stat label="kNN qps" value={`${(bench.knnQps / 1000).toFixed(1)}k/s`} highlight />
            <Stat label="ray qps" value={`${(bench.rayQps / 1000).toFixed(1)}k/s`} highlight />
            <Stat label="point-in-solid" value={`${bench.pointInSolidMs.toFixed(2)} ms/q`} />
            <Stat label="BVH depth" value={String(bench.bvh.maxDepth)} />
          </div>

          {/* Broadphase comparison */}
          <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
              broadphase pair generation
            </div>
            {[
              { label: "spatial hash", pairs: bench.hashPairs, ms: bench.hashPairsMs, color: "bg-primary" },
              { label: "BVH self-traverse", pairs: bench.bvhPairs, ms: bench.bvhPairsMs, color: "bg-accent" },
            ].map((row) => {
              const max = Math.max(bench.hashPairs, bench.bvhPairs, 1);
              return (
                <div key={row.label} className="flex items-center gap-2 text-[10px] font-mono">
                  <span className="w-32 text-muted-foreground">{row.label}</span>
                  <div className="flex-1 h-2 bg-muted-foreground/15 rounded">
                    <div className={`${row.color} h-full rounded`} style={{ width: `${(row.pairs / max) * 100}%` }} />
                  </div>
                  <span className="w-20 text-right text-foreground/90 tabular-nums">{row.pairs.toLocaleString()}</span>
                  <span className="w-16 text-right text-muted-foreground tabular-nums">{row.ms.toFixed(1)} ms</span>
                </div>
              );
            })}
          </div>

          {/* Fabrication intelligence */}
          <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
            <div className="flex items-baseline justify-between">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                fabrication intelligence · leaf classification
              </div>
              <div className="text-[10px] font-mono text-muted-foreground">{bench.fab.ms} ms · {bench.leafCount.toLocaleString()} leaves</div>
            </div>
            {fabHist && (
              <div className="flex h-3 w-full rounded overflow-hidden">
                {fabHist.map((c, i) => {
                  const pct = (c / bench.leafCount) * 100;
                  if (pct === 0) return null;
                  return (
                    <div
                      key={i}
                      className={FAB_COLORS[i]}
                      style={{ width: `${pct}%` }}
                      title={`${FAB_CLASS_LABELS[i]}: ${c}`}
                    />
                  );
                })}
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-1 text-[10px] font-mono pt-1">
              {FAB_CLASS_LABELS.map((label, i) => (
                <div key={label} className="flex items-center gap-1.5">
                  <span className={`w-2 h-2 rounded-sm ${FAB_COLORS[i]}`} />
                  <span className="text-muted-foreground">{label}</span>
                  <span className="text-foreground/90 tabular-nums ml-auto">{fabHist?.[i] ?? 0}</span>
                </div>
              ))}
            </div>
            <ul className="text-[10px] font-mono text-muted-foreground/95 space-y-0.5 pt-1">
              {bench.fab.notes.map((n, i) => <li key={i}>· {n}</li>)}
            </ul>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-1 text-[10px] pt-1">
              <Stat label="trapped clusters" value={String(bench.fab.trappedClusters)} />
              <Stat label="overhang leaves" value={String(bench.fab.overhangLeaves)} />
              <Stat label="inaccessible" value={String(bench.fab.inaccessibleLeaves)} />
              <Stat label="dense clusters" value={String(bench.fab.collisionClusters)} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`rounded border px-2 py-1 ${highlight ? "border-primary/60 bg-primary/[0.04]" : "border-border/60"}`}>
      <div className="uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className={`font-mono tabular-nums ${highlight ? "text-primary" : "text-foreground/90"}`}>{value}</div>
    </div>
  );
}
