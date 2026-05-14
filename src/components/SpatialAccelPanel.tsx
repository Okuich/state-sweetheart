import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh } from "@/lib/meshing/octree";
import { buildAdjacency } from "@/lib/meshing/adjacency";
import {
  bvhAllPairs,
  bvhFromOctree,
  buildSpatialHash,
  broadphasePairs,
  buildKDTree,
  knn,
  rayQuery,
  pointInSolid,
  analyzeFabrication,
  mortonCodes,
  mortonOrder,
  FAB_CLASS_LABELS,
  type BVH,
  type FabIntelReport,
} from "@/lib/spatial";
import { planDistributed, type DistPartResult } from "@/lib/distpart";
import { runPersistentSuite, type PersistentKernelSuite, type KernelMetrics } from "@/lib/gpu/persistentKernels";

interface PartitionThroughput {
  rank: number;
  leaves: number;
  knnQps: number;
  rayQps: number;
  bbox: { min: [number, number, number]; max: [number, number, number] };
}

interface MultiGpuBench {
  partitionCount: number;
  perPartition: PartitionThroughput[];
  /** Σ per-partition qps (theoretical aggregate). */
  aggregateKnnQps: number;
  aggregateRayQps: number;
  /** Σ per-partition / max(per-partition) — load-balance speedup. */
  knnSpeedup: number;
  raySpeedup: number;
  dist: DistPartResult;
}

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
  multiGpu: MultiGpuBench;
  kernels: PersistentKernelSuite;
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
  const [partitionCount, setPartitionCount] = useState(4);
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

      // ── Multi-GPU partition-aware throughput ─────────────────────────────
      // Distribute leaves into P morton chunks (cache-local per rank).
      const P = Math.max(1, Math.min(16, partitionCount));
      const codes = mortonCodes(cents, mesh.bbox);
      const order = mortonOrder(codes);
      const leafPart = new Uint16Array(mesh.leaves.length);
      const chunk = Math.ceil(mesh.leaves.length / P);
      for (let i = 0; i < order.length; i++) leafPart[order[i]] = Math.min(P - 1, Math.floor(i / chunk));

      // Per-partition AABB.
      const perPartition: PartitionThroughput[] = [];
      for (let p = 0; p < P; p++) {
        const minP: [number, number, number] = [Infinity, Infinity, Infinity];
        const maxP: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        let count = 0;
        for (let i = 0; i < leafPart.length; i++) {
          if (leafPart[i] !== p) continue;
          count++;
          for (let a = 0; a < 3; a++) {
            if (lo[i * 3 + a] < minP[a]) minP[a] = lo[i * 3 + a];
            if (hi[i * 3 + a] > maxP[a]) maxP[a] = hi[i * 3 + a];
          }
        }
        if (!count) {
          perPartition.push({ rank: p, leaves: 0, knnQps: 0, rayQps: 0, bbox: { min: [0, 0, 0], max: [0, 0, 0] } });
          continue;
        }

        // Per-rank kNN throughput inside its AABB.
        const PK = 400;
        const t1 = performance.now();
        for (let i = 0; i < PK; i++) {
          const u = i / PK;
          const x = minP[0] + (maxP[0] - minP[0]) * u;
          const y = minP[1] + (maxP[1] - minP[1]) * (1 - u);
          const z = minP[2] + (maxP[2] - minP[2]) * ((u * 11) % 1);
          knn(kd, [x, y, z], 8);
        }
        const knnPMs = performance.now() - t1;

        // Per-rank ray throughput shooting through the AABB.
        const PR = 250;
        const cx = (minP[0] + maxP[0]) / 2;
        const cy = (minP[1] + maxP[1]) / 2;
        const cz = (minP[2] + maxP[2]) / 2;
        const t2 = performance.now();
        for (let i = 0; i < PR; i++) {
          const a = (i / PR) * Math.PI * 2;
          rayQuery(bvh, [cx - 1, cy + 0.2 * Math.cos(a), cz + 0.2 * Math.sin(a)], [1, 0, 0]);
        }
        const rayPMs = performance.now() - t2;

        perPartition.push({
          rank: p,
          leaves: count,
          knnQps: PK / Math.max(0.001, knnPMs / 1000),
          rayQps: PR / Math.max(0.001, rayPMs / 1000),
          bbox: { min: minP, max: maxP },
        });
      }

      const aggregateKnnQps = perPartition.reduce((a, p) => a + p.knnQps, 0);
      const aggregateRayQps = perPartition.reduce((a, p) => a + p.rayQps, 0);
      const maxKnn = Math.max(...perPartition.map((p) => p.knnQps), 1);
      const maxRay = Math.max(...perPartition.map((p) => p.rayQps), 1);
      const knnSpeedup = aggregateKnnQps / maxKnn;
      const raySpeedup = aggregateRayQps / maxRay;

      // Halo + scaling telemetry from distpart engine.
      const dist = planDistributed({
        mesh,
        adj,
        partitionCount: P,
        algorithm: "kway",
        traversal: true,
        haloSyncIterations: 8,
        broadphase: true,
      });

      // ── Persistent GPU traversal kernels (BVH/hash/KD) ────────────────────
      const kernels = runPersistentSuite({
        bvh,
        hash,
        kd,
        bbox: mesh.bbox,
        queryCount: 4096,
        k: 8,
      });

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
        multiGpu: {
          partitionCount: P,
          perPartition,
          aggregateKnnQps,
          aggregateRayQps,
          knnSpeedup,
          raySpeedup,
          dist,
        },
        kernels,
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
            spatial · BVH · KD-tree · spatial hash · fab intel · multi-GPU
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

      <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
        <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
            <span className="text-muted-foreground">spatial-hash cell size</span>
            <span className="text-foreground font-mono">{cellSize.toFixed(2)}</span>
          </div>
          <Slider value={[cellSize]} min={0.05} max={0.5} step={0.01} onValueChange={(v) => setCellSize(v[0])} />
        </div>
        <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
            <span className="text-muted-foreground">GPU partitions</span>
            <span className="text-foreground font-mono">{partitionCount}× rank</span>
          </div>
          <Slider value={[partitionCount]} min={2} max={16} step={1} onValueChange={(v) => setPartitionCount(v[0])} />
        </div>
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

          {/* Multi-GPU scaling + halo cost */}
          <MultiGpuSection bench={bench.multiGpu} />

          {/* Persistent GPU traversal kernels */}
          <PersistentKernelsSection suite={bench.kernels} />

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

function MultiGpuSection({ bench }: { bench: MultiGpuBench }) {
  const { dist, perPartition, partitionCount, aggregateKnnQps, aggregateRayQps, knnSpeedup, raySpeedup } = bench;
  const maxKnn = Math.max(...perPartition.map((p) => p.knnQps), 1);
  const maxRay = Math.max(...perPartition.map((p) => p.rayQps), 1);
  const maxLeaves = Math.max(...perPartition.map((p) => p.leaves), 1);
  const halo = dist.halo;
  const sync = dist.haloSync;
  const trav = dist.traversal;
  const syncBytesGB = halo.syncBytes / 1e9;
  const idealKnn = perPartition.length ? maxKnn * perPartition.length : 0;
  const balanceEff = idealKnn ? (aggregateKnnQps / idealKnn) * 100 : 0;

  return (
    <div className="rounded-md border border-border bg-background/30 p-3 space-y-3">
      <div className="flex items-baseline justify-between">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          multi-GPU scaling · halo sync · per-rank throughput
        </div>
        <div className="text-[10px] font-mono text-muted-foreground">
          {partitionCount}× ranks · k-way · cut {halo.edgeCut.toLocaleString()}
        </div>
      </div>

      {/* Top-line scaling stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
        <Stat label="kNN aggregate" value={`${(aggregateKnnQps / 1000).toFixed(1)}k/s`} highlight />
        <Stat label="ray aggregate" value={`${(aggregateRayQps / 1000).toFixed(1)}k/s`} highlight />
        <Stat label="kNN speedup" value={`${knnSpeedup.toFixed(2)}× / ${partitionCount}×`} />
        <Stat label="ray speedup" value={`${raySpeedup.toFixed(2)}× / ${partitionCount}×`} />
        <Stat label="load balance" value={`${balanceEff.toFixed(0)}%`} />
        <Stat label="halo sync bytes" value={`${syncBytesGB < 0.01 ? `${(halo.syncBytes / 1e6).toFixed(2)} MB` : `${syncBytesGB.toFixed(3)} GB`}`} />
        <Stat label="comm rounds" value={String(halo.rounds.length)} />
        <Stat label="sync latency" value={sync ? `${sync.avgUs.toFixed(1)} µs/it` : "—"} />
      </div>

      {/* Halo sync iteration trace */}
      {sync && (
        <div className="space-y-1">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            <span>halo sync trace · {sync.iterations.length} iterations</span>
            <span className="font-mono text-foreground/80">
              eff. {sync.effectiveGBs.toFixed(1)} GB/s · stale {sync.maxStaleness}
            </span>
          </div>
          <div className="flex items-end gap-0.5 h-12 bg-muted-foreground/5 rounded p-1">
            {sync.iterations.map((it) => {
              const maxUs = Math.max(...sync.iterations.map((x) => x.iterUs), 1);
              const h = (it.iterUs / maxUs) * 100;
              return (
                <div
                  key={it.iter}
                  className="flex-1 bg-accent/70 rounded-sm"
                  style={{ height: `${h}%`, minHeight: 2 }}
                  title={`it ${it.iter}: ${it.iterUs.toFixed(1)} µs · ${(it.bytes / 1024).toFixed(0)} KB · ${it.packets} pkts`}
                />
              );
            })}
          </div>
        </div>
      )}

      {/* Traversal speedup */}
      {trav && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
          <Stat label="BFS super-steps" value={String(trav.steps.length)} />
          <Stat label="traversal speedup" value={`${trav.speedup.toFixed(2)}×`} highlight />
          <Stat label="halo bytes" value={`${(trav.totalHaloBytes / 1024).toFixed(1)} KB`} />
          <Stat label="wall-clock" value={`${trav.estimatedUs.toFixed(0)} µs`} />
        </div>
      )}

      {/* Per-partition throughput */}
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          per-rank throughput · leaves · kNN qps · ray qps
        </div>
        <div className="space-y-0.5 font-mono text-[10px]">
          <div className="flex items-center gap-2 text-muted-foreground/70 pb-0.5 border-b border-border/40">
            <span className="w-10">rank</span>
            <span className="w-16 text-right">leaves</span>
            <div className="flex-1 flex gap-1">
              <span className="flex-1 text-[9px] uppercase tracking-[0.15em]">kNN qps</span>
              <span className="flex-1 text-[9px] uppercase tracking-[0.15em]">ray qps</span>
            </div>
          </div>
          {perPartition.map((p) => (
            <div key={p.rank} className="flex items-center gap-2">
              <span className="w-10 text-foreground/80">G{p.rank}</span>
              <span className="w-16 text-right text-muted-foreground tabular-nums">
                {p.leaves.toLocaleString()}
              </span>
              <div className="flex-1 flex gap-1 items-center">
                <div className="flex-1 h-2 bg-muted-foreground/10 rounded relative">
                  <div
                    className="absolute inset-y-0 left-0 bg-primary/80 rounded"
                    style={{ width: `${(p.knnQps / maxKnn) * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right text-foreground/90 tabular-nums">
                  {(p.knnQps / 1000).toFixed(1)}k
                </span>
                <div className="flex-1 h-2 bg-muted-foreground/10 rounded relative">
                  <div
                    className="absolute inset-y-0 left-0 bg-accent/80 rounded"
                    style={{ width: `${(p.rayQps / maxRay) * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right text-foreground/90 tabular-nums">
                  {(p.rayQps / 1000).toFixed(1)}k
                </span>
              </div>
            </div>
          ))}
        </div>
        <div className="text-[9px] text-muted-foreground/70 font-mono pt-1">
          load skew {((maxLeaves / Math.max(1, perPartition.reduce((a, p) => a + p.leaves, 0) / perPartition.length)) - 1 ) * 100 | 0}% · imbalance {dist.assignment.stats.imbalance.toFixed(2)}×
        </div>
      </div>
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
