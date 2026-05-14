import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  buildSparseSDF, distance, gradient, nearestSurface, sphereCollide,
  partitionSDF, buildEmbedding, createGpuSdfBackend,
  type SDFPrim, type AdaptiveHint, type SparseSDF, type SDFPartitionPlan, type SDFEmbedding,
  type GpuMode,
} from "@/lib/sdf";

interface Stat { min: number; avg: number; max: number; runs: number }

function makeStat(samples: number[]): Stat {
  if (!samples.length) return { min: 0, avg: 0, max: 0, runs: 0 };
  let mn = Infinity, mx = -Infinity, s = 0;
  for (const v of samples) { mn = Math.min(mn, v); mx = Math.max(mx, v); s += v; }
  return { min: mn, avg: s / samples.length, max: mx, runs: samples.length };
}

interface GpuBench {
  available: boolean;
  reason?: string;
  brickCount?: number;
  levelCount?: number;
  distQps?: Stat;
  gradQps?: Stat;
  collideQps?: Stat;
  nearestQps?: Stat;
  distGpuMs?: number;
  gradGpuMs?: number;
  collideGpuMs?: number;
  nearestGpuMs?: number;
  speedupDist?: number;
  speedupGrad?: number;
  speedupCollide?: number;
  // CPU↔GPU correctness (max / mean absolute error per mode).
  errDistMax?: number;     errDistMean?: number;
  errGradMax?: number;     errGradMean?: number;
  errCollideMax?: number;  errCollideMean?: number;
  errNearestMax?: number;  errNearestMean?: number;
  errSampleN?: number;
}

interface BenchResult {
  sdf: SparseSDF;
  partition: SDFPartitionPlan;
  embedding: SDFEmbedding;
  buildMs: number;
  partitionMs: number;
  embeddingMs: number;
  distQps: Stat;
  gradQps: Stat;
  collideQps: Stat;
  nearestErr: number;
  gpu?: GpuBench;
  config: BenchConfig;
}

interface BenchConfig {
  cpuQueries: number;
  gpuQueries: number;
  warmup: number;
  radius: number;
  newtonIters: number;
  batchSize: number;
  runs: number;
}

const PRESETS: { label: string; prims: SDFPrim[]; hints: AdaptiveHint[] }[] = [
  {
    label: "sphere · solid",
    prims: [{ kind: "sphere", center: [0, 0, 0], radius: 0.55 }],
    hints: [{ kind: "high_curvature", center: [0, 0, 0], radius: 0.7, weight: 0.8 }],
  },
  {
    label: "torus · genus-1",
    prims: [{ kind: "torus", center: [0, 0, 0], major: 0.45, minor: 0.18 }],
    hints: [{ kind: "thin_wall", center: [0.45, 0, 0], radius: 0.35, weight: 0.9 }],
  },
  {
    label: "block + bore",
    prims: [
      { kind: "box", center: [0, 0, 0], half: [0.6, 0.4, 0.4] },
      { kind: "cylinder", center: [0, 0, 0], axis: [0, 1, 0], radius: 0.18, height: 1.0 },
    ],
    hints: [{ kind: "stress", center: [0, 0, 0], radius: 0.4, weight: 0.9 }],
  },
  {
    label: "thin shell + hotspot",
    prims: [
      { kind: "sphere", center: [0, 0, 0], radius: 0.55 },
      { kind: "sphere", center: [0, 0, 0], radius: 0.48 },
    ],
    hints: [{ kind: "thermal", center: [0.5, 0, 0], radius: 0.3, weight: 1 }],
  },
];

const BBOX = { min: [-0.8, -0.8, -0.8] as [number, number, number], max: [0.8, 0.8, 0.8] as [number, number, number] };

export function SDFPanel() {
  const [presetIdx, setPresetIdx] = useState(2);
  const [voxelSize, setVoxelSize] = useState(0.06);
  const [bandWidth, setBandWidth] = useState(3);
  const [adaptive, setAdaptive] = useState(1);
  const [partitionCount, setPartitionCount] = useState(4);

  // Benchmark configuration
  const [cpuQueries, setCpuQueries] = useState(5000);
  const [gpuQueries, setGpuQueries] = useState(50000);
  const [warmup, setWarmup] = useState(2);
  const [radius, setRadius] = useState(0.05);
  const [newtonIters, setNewtonIters] = useState(6);
  const [batchSize, setBatchSize] = useState(50000);
  const [runs, setRuns] = useState(5);

  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<BenchResult | null>(null);

  const preset = PRESETS[presetIdx];

  const run = () => {
    setRunning(true);
    setTimeout(async () => {
      const config: BenchConfig = { cpuQueries, gpuQueries, warmup, radius, newtonIters, batchSize, runs };

      const t0 = performance.now();
      const sdf = buildSparseSDF(BBOX, preset.prims, {
        voxelSize, bandWidth, hints: preset.hints, maxAdaptiveLevels: adaptive,
      });
      const buildMs = performance.now() - t0;

      const t1 = performance.now();
      const partition = partitionSDF(sdf, partitionCount, 1);
      const partitionMs = performance.now() - t1;

      const t2 = performance.now();
      const embedding = buildEmbedding(sdf);
      const embeddingMs = performance.now() - t2;

      // Random points used by both CPU and GPU paths.
      const N = Math.max(1, cpuQueries);
      const pts: [number, number, number][] = [];
      for (let i = 0; i < N; i++) {
        pts.push([
          BBOX.min[0] + Math.random() * (BBOX.max[0] - BBOX.min[0]),
          BBOX.min[1] + Math.random() * (BBOX.max[1] - BBOX.min[1]),
          BBOX.min[2] + Math.random() * (BBOX.max[2] - BBOX.min[2]),
        ]);
      }

      // CPU warmup
      for (let w = 0; w < warmup; w++) {
        for (let i = 0; i < Math.min(N, 256); i++) {
          distance(sdf, pts[i]); gradient(sdf, pts[i]); sphereCollide(sdf, pts[i], radius);
        }
      }

      // CPU benchmark — multi-run for min/avg/max stats.
      const distSamples: number[] = [], gradSamples: number[] = [], collSamples: number[] = [];
      for (let r = 0; r < Math.max(1, runs); r++) {
        let acc = 0;
        const td0 = performance.now();
        for (const p of pts) acc += distance(sdf, p);
        const distMs = performance.now() - td0;
        distSamples.push(N / Math.max(0.001, distMs / 1000));

        let ga = 0;
        const tg0 = performance.now();
        for (const p of pts) { const g = gradient(sdf, p); ga += g[0]; }
        const gradMs = performance.now() - tg0;
        gradSamples.push(N / Math.max(0.001, gradMs / 1000));

        let hits = 0;
        const tc0 = performance.now();
        for (const p of pts) if (sphereCollide(sdf, p, radius).hit) hits++;
        const colMs = performance.now() - tc0;
        collSamples.push(N / Math.max(0.001, colMs / 1000));
        void acc; void ga; void hits;
      }
      const distQps = makeStat(distSamples);
      const gradQps = makeStat(gradSamples);
      const collideQps = makeStat(collSamples);

      // Nearest-surface accuracy on a fixed off-surface probe.
      const rNear = nearestSurface(sdf, [0.7, 0, 0], newtonIters);
      const nearestErr = Math.abs(distance(sdf, rNear.point));

      // GPU benchmark
      let gpu: GpuBench | undefined;
      try {
        const back = await createGpuSdfBackend(sdf);
        if (!back.available) {
          gpu = { available: false, reason: back.reason };
        } else {
          const NG = Math.max(1, gpuQueries);
          const gflat = new Float32Array(NG * 3);
          for (let i = 0; i < NG; i++) {
            gflat[i * 3]     = BBOX.min[0] + Math.random() * (BBOX.max[0] - BBOX.min[0]);
            gflat[i * 3 + 1] = BBOX.min[1] + Math.random() * (BBOX.max[1] - BBOX.min[1]);
            gflat[i * 3 + 2] = BBOX.min[2] + Math.random() * (BBOX.max[2] - BBOX.min[2]);
          }
          const bs = Math.max(1, Math.min(batchSize, NG));

          // GPU warmup
          for (let w = 0; w < warmup; w++) {
            await back.run("distance", gflat.subarray(0, Math.min(NG, 1024) * 3));
          }

          // Run a mode end-to-end as configured `runs` times. Each run dispatches
          // ceil(N/batchSize) batches and returns total wall-clock + gpu time.
          const runMode = async (
            mode: GpuMode,
            buf: Float32Array,
            count: number,
          ) => {
            const totals: number[] = [];
            let lastGpuMs = 0, totalMs = 0;
            for (let r = 0; r < Math.max(1, runs); r++) {
              const t = performance.now();
              let gpuAcc = 0;
              for (let off = 0; off < count; off += bs) {
                const sliceN = Math.min(bs, count - off);
                const slice = buf.subarray(off * 3, (off + sliceN) * 3);
                const res = await back.run(mode, slice, { radius, iters: newtonIters });
                gpuAcc += res.gpuMs;
              }
              const ms = performance.now() - t;
              totals.push(count / Math.max(0.001, ms / 1000));
              lastGpuMs = gpuAcc;
              totalMs = ms;
            }
            return { qps: makeStat(totals), gpuMs: lastGpuMs, totalMs };
          };

          const nN = Math.min(5000, NG);
          const d = await runMode("distance", gflat, NG);
          const g = await runMode("gradient", gflat, NG);
          const c = await runMode("collide",  gflat, NG);
          const n = await runMode("nearest",  gflat.subarray(0, nN * 3), nN);

          // CPU↔GPU correctness check on a shared sample.
          const NC = Math.min(2000, NG);
          const sample = gflat.subarray(0, NC * 3);
          const dRef = await back.run("distance", sample);
          const gRef = await back.run("gradient", sample);
          const cRef = await back.run("collide",  sample, { radius });
          const nRef = await back.run("nearest",  sample, { iters: newtonIters });
          let edMax = 0, edSum = 0, egMax = 0, egSum = 0, ecMax = 0, ecSum = 0, enMax = 0, enSum = 0;
          for (let i = 0; i < NC; i++) {
            const p: [number, number, number] = [sample[i * 3], sample[i * 3 + 1], sample[i * 3 + 2]];
            const dCpu = distance(sdf, p);
            const eD = Math.abs(dCpu - dRef.out[i * 4]);
            edMax = Math.max(edMax, eD); edSum += eD;
            const gCpu = gradient(sdf, p);
            const eG = Math.hypot(gCpu[0] - gRef.out[i * 4], gCpu[1] - gRef.out[i * 4 + 1], gCpu[2] - gRef.out[i * 4 + 2]);
            egMax = Math.max(egMax, eG); egSum += eG;
            const cCpu = sphereCollide(sdf, p, radius);
            const eC = Math.abs((cCpu.hit ? cCpu.depth : 0) - cRef.out[i * 4]);
            ecMax = Math.max(ecMax, eC); ecSum += eC;
            const nCpu = nearestSurface(sdf, p, newtonIters);
            const eN = Math.hypot(nCpu.point[0] - nRef.out[i * 4], nCpu.point[1] - nRef.out[i * 4 + 1], nCpu.point[2] - nRef.out[i * 4 + 2]);
            enMax = Math.max(enMax, eN); enSum += eN;
          }

          gpu = {
            available: true,
            brickCount: back.brickCount,
            levelCount: back.levelCount,
            distQps: d.qps, gradQps: g.qps, collideQps: c.qps, nearestQps: n.qps,
            distGpuMs: d.gpuMs, gradGpuMs: g.gpuMs, collideGpuMs: c.gpuMs, nearestGpuMs: n.gpuMs,
            speedupDist:    d.qps.avg / Math.max(1, distQps.avg),
            speedupGrad:    g.qps.avg / Math.max(1, gradQps.avg),
            speedupCollide: c.qps.avg / Math.max(1, collideQps.avg),
            errSampleN: NC,
            errDistMax: edMax,    errDistMean: edSum / NC,
            errGradMax: egMax,    errGradMean: egSum / NC,
            errCollideMax: ecMax, errCollideMean: ecSum / NC,
            errNearestMax: enMax, errNearestMean: enSum / NC,
          };
          back.destroy();
        }
      } catch (e) {
        gpu = { available: false, reason: e instanceof Error ? e.message : String(e) };
      }

      setResult({ sdf, partition, embedding, buildMs, partitionMs, embeddingMs, distQps, gradQps, collideQps, nearestErr, gpu, config });
      setRunning(false);
    }, 0);
  };

  const partitionColors = useMemo(
    () => ["bg-primary", "bg-accent", "bg-destructive/70", "bg-emerald-500", "bg-sky-500", "bg-amber-500", "bg-fuchsia-500", "bg-teal-500"],
    [],
  );

  return (
    <section className="rounded-xl border border-border bg-card/40 p-5 space-y-5">
      <header className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <h2 className="text-lg font-semibold">SDF Infrastructure</h2>
          <p className="text-sm text-muted-foreground">
            Sparse narrow-band signed distance fields with adaptive bricks, GPU-shaped queries, multi-GPU partitioning, and retrieval embeddings.
          </p>
        </div>
        <Button onClick={run} disabled={running}>{running ? "Baking…" : "Bake SDF"}</Button>
      </header>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
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

      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
        <SliderRow label="Voxel size" value={voxelSize} min={0.03} max={0.15} step={0.01} onChange={setVoxelSize} fmt={(v) => v.toFixed(2)} />
        <SliderRow label="Narrow band" value={bandWidth} min={1} max={6} step={1} onChange={setBandWidth} fmt={(v) => `${v} vx`} />
        <SliderRow label="Adaptive levels" value={adaptive} min={0} max={3} step={1} onChange={setAdaptive} fmt={(v) => `${v}`} />
        <SliderRow label="Partitions (GPUs)" value={partitionCount} min={1} max={8} step={1} onChange={setPartitionCount} fmt={(v) => `${v}`} />
      </div>

      <div className="rounded-lg border border-border/60 bg-muted/10 p-3 space-y-3">
        <div className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Benchmark configuration</div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          <SliderRow label="CPU queries"    value={cpuQueries} min={500}  max={20000}  step={500}  onChange={setCpuQueries} fmt={(v) => `${(v/1000).toFixed(1)}k`} />
          <SliderRow label="GPU queries"    value={gpuQueries} min={5000} max={500000} step={5000} onChange={setGpuQueries} fmt={(v) => `${(v/1000).toFixed(0)}k`} />
          <SliderRow label="Warmup passes"  value={warmup}     min={0}    max={5}      step={1}    onChange={setWarmup}     fmt={(v) => `${v}`} />
          <SliderRow label="Runs / mode"    value={runs}       min={1}    max={20}     step={1}    onChange={setRuns}       fmt={(v) => `${v}`} />
          <SliderRow label="Sphere radius"  value={radius}     min={0.01} max={0.2}    step={0.01} onChange={setRadius}     fmt={(v) => v.toFixed(2)} />
          <SliderRow label="Newton iters"   value={newtonIters} min={1}   max={16}     step={1}    onChange={setNewtonIters} fmt={(v) => `${v}`} />
          <SliderRow label="GPU batch size" value={batchSize}  min={1000} max={200000} step={1000} onChange={setBatchSize}  fmt={(v) => `${(v/1000).toFixed(0)}k`} />
        </div>
      </div>

      {result && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Field statistics</h3>
            <Stats stats={result} />
            <h3 className="text-sm font-semibold pt-2">Narrow-band histogram</h3>
            <Histogram data={result.embedding.topology.bandHistogram} />
          </div>

          <div className="space-y-3">
            <h3 className="text-sm font-semibold">Partition layout (axis {["X", "Y", "Z"][result.partition.axis]})</h3>
            <div className="flex h-6 w-full overflow-hidden rounded border border-border">
              {result.partition.resident.map((r, i) => (
                <div key={i} className={`${partitionColors[i % partitionColors.length]} flex items-center justify-center text-[10px] font-mono text-background`} style={{ width: `${(r.length / Math.max(1, result.sdf.bricks.length)) * 100}%` }}>
                  {r.length}
                </div>
              ))}
            </div>
            <div className="text-xs text-muted-foreground">
              imbalance {(result.partition.imbalance * 100).toFixed(1)}% · halo bricks {result.partition.totalHalo}
            </div>

          </div>

          <div className="lg:col-span-2 space-y-3 rounded-lg border border-border bg-muted/10 p-3">
            <div className="flex items-baseline justify-between flex-wrap gap-2">
              <h3 className="text-sm font-semibold">Distributed partition dashboard</h3>
              <span className="text-[10px] font-mono text-muted-foreground">
                P={result.partition.partitionCount} · halo=1 brick · split-axis {["X","Y","Z"][result.partition.axis]}
              </span>
            </div>
            <PartitionDashboard partition={result.partition} colors={partitionColors} />
          </div>

          <div className="space-y-3">

            <h3 className="text-sm font-semibold pt-2">Manufacturability prior</h3>
            <FabBars m={result.embedding.manufacturability} />

            <h3 className="text-sm font-semibold pt-2">Geometry embedding (32-d)</h3>
            <EmbeddingStrip vec={result.embedding.geometry} />
          </div>

          <div className="lg:col-span-2 space-y-2 rounded-lg border border-border bg-muted/20 p-3">
            <div className="flex items-baseline justify-between flex-wrap gap-2">
              <h3 className="text-sm font-semibold">CPU benchmark</h3>
              <span className="text-[10px] font-mono text-muted-foreground">
                N={result.config.cpuQueries.toLocaleString()} · warmup={result.config.warmup} · runs={result.config.runs} · r={result.config.radius.toFixed(2)} · iters={result.config.newtonIters}
              </span>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs font-mono">
              <CpuStatCell label="distance" qps={result.distQps} />
              <CpuStatCell label="gradient" qps={result.gradQps} />
              <CpuStatCell label="sphere collide" qps={result.collideQps} />
            </div>
          </div>

          {result.gpu && (
            <div className="lg:col-span-2 space-y-2 rounded-lg border border-border bg-muted/20 p-3">
              <div className="flex items-baseline justify-between flex-wrap gap-2">
                <h3 className="text-sm font-semibold">WebGPU backend</h3>
                {result.gpu.available ? (
                  <span className="text-[10px] font-mono text-muted-foreground">
                    {result.gpu.brickCount} bricks · {result.gpu.levelCount} levels · N={result.config.gpuQueries.toLocaleString()} · batch={result.config.batchSize.toLocaleString()} · runs={result.config.runs}
                  </span>
                ) : (
                  <span className="text-[10px] font-mono text-amber-500">unavailable: {result.gpu.reason}</span>
                )}
              </div>
              {result.gpu.available && (
                <>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                    <GpuCell label="distance" qps={result.gpu.distQps!} gpuMs={result.gpu.distGpuMs!} speedup={result.gpu.speedupDist!} />
                    <GpuCell label="gradient" qps={result.gpu.gradQps!} gpuMs={result.gpu.gradGpuMs!} speedup={result.gpu.speedupGrad!} />
                    <GpuCell label="sphere collide" qps={result.gpu.collideQps!} gpuMs={result.gpu.collideGpuMs!} speedup={result.gpu.speedupCollide!} />
                    <GpuCell label="nearest (Newton)" qps={result.gpu.nearestQps!} gpuMs={result.gpu.nearestGpuMs!} />
                  </div>
                  {result.gpu.errSampleN !== undefined && (
                    <div className="space-y-1 pt-1">
                      <div className="flex items-baseline justify-between">
                        <h4 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                          GPU↔CPU correctness
                        </h4>
                        <span className="text-[10px] font-mono text-muted-foreground">
                          n={result.gpu.errSampleN} samples · |Δ| in world units
                        </span>
                      </div>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono">
                        <ErrCell label="distance"       max={result.gpu.errDistMax!}    mean={result.gpu.errDistMean!} />
                        <ErrCell label="gradient"       max={result.gpu.errGradMax!}    mean={result.gpu.errGradMean!} />
                        <ErrCell label="sphere collide" max={result.gpu.errCollideMax!} mean={result.gpu.errCollideMean!} />
                        <ErrCell label="nearest"        max={result.gpu.errNearestMax!} mean={result.gpu.errNearestMean!} />
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
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

function Stats({ stats }: { stats: BenchResult }) {
  const { sdf } = stats;
  const rows: [string, string][] = [
    ["allocated bricks", `${sdf.stats.allocatedBricks} / ${sdf.stats.rootBrickCount}`],
    ["surface bricks", `${sdf.stats.surfaceBricks}`],
    ["adaptive bricks", `${sdf.stats.adaptiveBricks}`],
    ["voxels", `${sdf.stats.voxelCount.toLocaleString()}`],
    ["sparsity", `${(sdf.stats.sparsity * 100).toFixed(1)}%`],
    ["levels", `${sdf.stats.levels}`],
    ["build", `${stats.buildMs.toFixed(1)} ms`],
    ["partition", `${stats.partitionMs.toFixed(1)} ms`],
    ["embedding", `${stats.embeddingMs.toFixed(1)} ms`],
    ["distance qps (avg)", fmtQps(stats.distQps.avg)],
    ["gradient qps (avg)", fmtQps(stats.gradQps.avg)],
    ["collide qps (avg)",  fmtQps(stats.collideQps.avg)],
    ["nearest err", `${stats.nearestErr.toExponential(1)}`],
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

function Histogram({ data }: { data: number[] }) {
  const max = Math.max(...data, 1);
  return (
    <div className="flex h-16 items-end gap-0.5">
      {data.map((v, i) => (
        <div key={i} className="flex-1 bg-primary/70 rounded-sm" style={{ height: `${(v / max) * 100}%` }} title={`${v}`} />
      ))}
    </div>
  );
}

function FabBars({ m }: { m: SDFEmbedding["manufacturability"] }) {
  const rows: [string, number, string][] = [
    ["score", m.score, "bg-emerald-500"],
    ["overhang risk", m.overhangRisk, "bg-amber-500"],
    ["thin-wall risk", m.thinWallRisk, "bg-destructive"],
    ["trapped vol risk", m.trappedVolumeRisk, "bg-fuchsia-500"],
  ];
  return (
    <div className="space-y-1.5">
      {rows.map(([k, v, c]) => (
        <div key={k} className="flex items-center gap-2 text-xs">
          <span className="w-32 text-muted-foreground">{k}</span>
          <div className="flex-1 h-2 rounded bg-muted">
            <div className={`h-full rounded ${c}`} style={{ width: `${Math.min(100, v * 100)}%` }} />
          </div>
          <span className="w-10 text-right font-mono">{(v * 100).toFixed(0)}%</span>
        </div>
      ))}
    </div>
  );
}

function EmbeddingStrip({ vec }: { vec: Float32Array }) {
  const max = Math.max(...Array.from(vec).map(Math.abs), 1e-6);
  return (
    <div className="flex h-6 gap-px">
      {Array.from(vec).map((v, i) => {
        const t = Math.abs(v) / max;
        const hue = v >= 0 ? 200 : 10;
        return (
          <div key={i} className="flex-1" style={{ backgroundColor: `hsl(${hue} 80% ${20 + t * 50}%)` }} title={`${i}: ${v.toFixed(3)}`} />
        );
      })}
    </div>
  );
}

function fmtQps(v: number): string {
  if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (v >= 1e3) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(0);
}

function GpuCell({ label, qps, gpuMs, speedup }: { label: string; qps: Stat; gpuMs: number; speedup?: number }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 p-2 space-y-0.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm">{fmtQps(qps.avg)} qps</div>
      <div className="text-[10px] text-muted-foreground">
        min {fmtQps(qps.min)} · max {fmtQps(qps.max)} · n={qps.runs}
      </div>
      <div className="text-[10px] text-muted-foreground">
        gpu {gpuMs.toFixed(2)} ms{speedup !== undefined ? ` · ${speedup.toFixed(1)}× cpu` : ""}
      </div>
    </div>
  );
}

function CpuStatCell({ label, qps }: { label: string; qps: Stat }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 p-2 space-y-0.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm">{fmtQps(qps.avg)} qps</div>
      <div className="text-[10px] text-muted-foreground">
        min {fmtQps(qps.min)} · max {fmtQps(qps.max)} · n={qps.runs}
      </div>
    </div>
  );
}

function ErrCell({ label, max, mean }: { label: string; max: number; mean: number }) {
  // Heuristic threshold: SDF voxel-scale errors above ~5e-3 in world units start to matter.
  const warn = max > 5e-3;
  return (
    <div className={`rounded border p-2 space-y-0.5 ${warn ? "border-amber-500/60 bg-amber-500/5" : "border-border/60 bg-background/40"}`}>
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-sm">max {max.toExponential(2)}</div>
      <div className="text-[10px] text-muted-foreground">mean {mean.toExponential(2)}</div>
    </div>
  );
}

/**
 * Distributed-partition dashboard: per-partition stats, P×P comm
 * matrix (rows = receiver, cols = sender — entry counts halo bricks
 * pulled from sender to receiver), and a halo-set strip per partition.
 */
function PartitionDashboard({
  partition,
  colors,
}: { partition: SDFPartitionPlan; colors: string[] }) {
  const { partitionCount: P, resident, halos, commMatrix } = partition;

  const totalBricks = resident.reduce((a, b) => a + b.length, 0);
  const haloOut = new Array(P).fill(0);
  const haloIn = new Array(P).fill(0);
  let maxFlow = 0;
  for (let r = 0; r < P; r++) {
    for (let c = 0; c < P; c++) {
      const v = commMatrix[r * P + c];
      haloIn[r] += v;
      haloOut[c] += v;
      if (v > maxFlow) maxFlow = v;
    }
  }
  const meanRes = totalBricks / Math.max(1, P);

  const cellStyle = (v: number): React.CSSProperties => {
    if (v === 0) return { backgroundColor: "hsl(var(--muted) / 0.25)" };
    const t = v / Math.max(1, maxFlow);
    return { backgroundColor: `hsl(var(--primary) / ${(0.18 + t * 0.7).toFixed(2)})` };
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-2 space-y-2">
        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Per-partition stats</div>
        <div className="overflow-x-auto rounded border border-border/60">
          <table className="w-full text-[11px] font-mono">
            <thead className="bg-muted/30 text-muted-foreground">
              <tr>
                <th className="px-2 py-1 text-left">part</th>
                <th className="px-2 py-1 text-right">resident</th>
                <th className="px-2 py-1 text-right">share %</th>
                <th className="px-2 py-1 text-right">halo in</th>
                <th className="px-2 py-1 text-right">halo out</th>
                <th className="px-2 py-1 text-right">vs mean</th>
              </tr>
            </thead>
            <tbody>
              {resident.map((r, i) => {
                const share = totalBricks > 0 ? (r.length / totalBricks) * 100 : 0;
                const skew = meanRes > 0 ? ((r.length - meanRes) / meanRes) * 100 : 0;
                const skewColor = Math.abs(skew) > 25 ? "text-amber-500" : skew >= 0 ? "text-emerald-500" : "text-sky-400";
                return (
                  <tr key={i} className="border-t border-border/40">
                    <td className="px-2 py-1">
                      <span className="inline-flex items-center gap-1.5">
                        <span className={`inline-block h-2 w-2 rounded-sm ${colors[i % colors.length]}`} />
                        P{i}
                      </span>
                    </td>
                    <td className="px-2 py-1 text-right">{r.length}</td>
                    <td className="px-2 py-1 text-right text-muted-foreground">{share.toFixed(1)}</td>
                    <td className="px-2 py-1 text-right">{haloIn[i]}</td>
                    <td className="px-2 py-1 text-right">{haloOut[i]}</td>
                    <td className={`px-2 py-1 text-right ${skewColor}`}>{skew >= 0 ? "+" : ""}{skew.toFixed(0)}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        <div className="text-[11px] uppercase tracking-wide text-muted-foreground pt-1">Halo brick sets</div>
        <div className="space-y-1">
          {halos.map((h, i) => {
            const max = Math.max(...halos.map((x) => x.length), 1);
            return (
              <div key={i} className="flex items-center gap-2 text-[11px] font-mono">
                <span className="w-6 text-muted-foreground">P{i}</span>
                <div className="flex-1 h-2 rounded bg-muted/40 overflow-hidden">
                  <div className={`h-full ${colors[i % colors.length]} opacity-70`} style={{ width: `${(h.length / max) * 100}%` }} />
                </div>
                <span className="w-12 text-right">{h.length}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
          P×P comm matrix <span className="normal-case">(row = receiver, col = sender)</span>
        </div>
        <div
          className="grid gap-px rounded border border-border/60 bg-border/40 p-px"
          style={{ gridTemplateColumns: `auto repeat(${P}, minmax(0, 1fr))` }}
        >
          <div className="bg-muted/30" />
          {Array.from({ length: P }, (_, c) => (
            <div key={`h-${c}`} className="bg-muted/30 text-center text-[10px] font-mono text-muted-foreground py-0.5">
              P{c}
            </div>
          ))}
          {Array.from({ length: P }, (_, r) => (
            <div key={`row-${r}`} className="contents">
              <div className="bg-muted/30 text-center text-[10px] font-mono text-muted-foreground px-1 flex items-center justify-center">P{r}</div>
              {Array.from({ length: P }, (_, c) => {
                const v = commMatrix[r * P + c];
                return (
                  <div
                    key={`c-${r}-${c}`}
                    className="aspect-square flex items-center justify-center text-[9px] font-mono"
                    style={cellStyle(v)}
                    title={`P${c} → P${r}: ${v} halo bricks`}
                  >
                    {v > 0 ? v : ""}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
        <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono">
          <span>0</span>
          <div className="flex-1 mx-2 h-1.5 rounded bg-gradient-to-r from-muted/40 to-primary/80" />
          <span>{maxFlow}</span>
        </div>
      </div>
    </div>
  );
}
