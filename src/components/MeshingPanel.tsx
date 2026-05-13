import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  generateMesh,
  seedsFromFeatures,
  type AABB,
  type MeshingResult,
  type RefinementSeed,
} from "@/lib/meshing";
import { MeshViewer3D } from "./MeshViewer3D";
import { useServerFn } from "@tanstack/react-start";
import { exportMeshFn } from "@/lib/meshing/export.functions";
import { toast } from "sonner";

type ExportFormat = "vtk" | "obj" | "json";

const BBOX: AABB = { min: [0, 0, 0], max: [1, 1, 1] };

const PRESETS: { label: string; seeds: { kind: string }[] }[] = [
  { label: "bracket · holes + fillets", seeds: [
    { kind: "hole" }, { kind: "hole" }, { kind: "hole" },
    { kind: "fillet" }, { kind: "fillet" }, { kind: "chamfer" },
  ] },
  { label: "thermal hotspot", seeds: [{ kind: "hotspot" }, { kind: "hotspot" }] },
  { label: "thin-wall shell", seeds: [{ kind: "thin_wall" }, { kind: "thin_wall" }, { kind: "thin_wall" }] },
  { label: "uniform · no seeds", seeds: [] },
];

function placeSeeds(kinds: { kind: string }[]): { kind: string; center?: number[]; radius?: number; weight?: number }[] {
  const n = Math.max(1, kinds.length);
  return kinds.map((k, i) => {
    const a = ((i + 0.5) / n) * Math.PI * 2;
    return {
      kind: k.kind,
      center: [0.5 + 0.3 * Math.cos(a), 0.5 + 0.3 * Math.sin(a), 0.5 + 0.2 * Math.sin(a * 2)],
      radius: 0.18,
      weight: 0.7,
    };
  });
}

function cls(s: number, hi = 0.7, mid = 0.4) {
  return s >= hi ? "text-primary" : s >= mid ? "text-accent" : "text-destructive";
}

export function MeshingPanel() {
  const [presetIdx, setPresetIdx] = useState(0);
  const [maxDepth, setMaxDepth] = useState(5);
  const [partitionCount, setPartitionCount] = useState(8);
  const [result, setResult] = useState<MeshingResult | null>(null);
  const [running, setRunning] = useState(false);
  const [exporting, setExporting] = useState<ExportFormat | null>(null);
  const exportFn = useServerFn(exportMeshFn);

  const lastInput = useMemo(() => {
    const placed = placeSeeds(PRESETS[presetIdx].seeds);
    const seeds: RefinementSeed[] = seedsFromFeatures(BBOX, placed);
    return {
      bbox: BBOX,
      seeds,
      octree: { minDepth: 2, maxDepth, refineThreshold: 0.3, maxLeaves: 30_000 },
      partitionCount,
    };
  }, [presetIdx, maxDepth, partitionCount]);

  const handleExport = async (format: ExportFormat) => {
    setExporting(format);
    try {
      const res = await exportFn({ data: { ...lastInput, format } });
      if (!res.ok) {
        toast.error(`Export failed: ${res.error}`);
        return;
      }
      const blob = new Blob([res.content], { type: res.mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = res.filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
      toast.success(
        `${format.toUpperCase()} ready · ${(res.bytes / 1024).toFixed(1)} KB · ${res.summary.tets.toLocaleString()} tets`,
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(null);
    }
  };

  const run = () => {
    setRunning(true);
    requestAnimationFrame(() => {
      const placed = placeSeeds(PRESETS[presetIdx].seeds);
      const seeds: RefinementSeed[] = seedsFromFeatures(BBOX, placed);
      const r = generateMesh({
        bbox: BBOX,
        seeds,
        octree: { minDepth: 2, maxDepth, refineThreshold: 0.3, maxLeaves: 30_000 },
        partitionCount,
      });
      setResult(r);
      setRunning(false);
    });
  };

  const partitionView = useMemo(() => {
    if (!result) return null;
    const P = result.summary.partition.partitionCount;
    const cm = result.summary.partition.commMatrix;
    return { P, cm };
  }, [result]);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            meshing · octree + adaptive + partitioning
          </div>
          <h2 className="font-display text-2xl text-foreground">
            From CAD to <span className="text-primary">simulation-ready</span> volumes.
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {result ? <>build · <span className="text-foreground">{result.summary.totalMs} ms</span></> : "no mesh"}
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
            <div className="text-[10px] font-mono text-muted-foreground/80">{p.seeds.length} seeds</div>
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
            <span className="text-muted-foreground">max octree depth</span>
            <span className="text-foreground font-mono">{maxDepth}</span>
          </div>
          <Slider value={[maxDepth]} min={2} max={7} step={1} onValueChange={(v) => setMaxDepth(v[0])} />
        </div>
        <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
            <span className="text-muted-foreground">distributed partitions</span>
            <span className="text-foreground font-mono">{partitionCount}</span>
          </div>
          <Slider value={[partitionCount]} min={1} max={16} step={1} onValueChange={(v) => setPartitionCount(v[0])} />
        </div>
      </div>

      <div className="flex gap-2">
        <Button onClick={run} disabled={running} className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "meshing…" : result ? "re-mesh" : "generate mesh"}
        </Button>
      </div>

      {result && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
            <Stat label="leaves" value={result.summary.octree.leafCount.toLocaleString()} />
            <Stat label="tets" value={result.summary.tets.count.toLocaleString()} />
            <Stat label="vertices" value={result.summary.tets.vertexCount.toLocaleString()} />
            <Stat label="edges" value={result.summary.adjacency.edgeCount.toLocaleString()} />
            <Stat label="boundary leaves" value={result.summary.octree.boundaryLeaves.toLocaleString()} />
            <Stat label="warp stride" value={String(result.summary.adjacency.warpStride)} />
            <Stat label="inverted tets" value={String(result.summary.tets.invertedTets)} />
            <Stat label="non-manifold" value={String(result.summary.tets.nonManifoldFaces)} />
          </div>

          <MeshViewer3D result={result} />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {/* Quality histogram */}
            <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                aspect-ratio histogram · log2 bins
              </div>
              <div className="flex items-end gap-1 h-24">
                {result.summary.tets.aspectHist.map((c, i) => {
                  const max = Math.max(...result.summary.tets.aspectHist, 1);
                  const h = (c / max) * 100;
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1">
                      <div
                        className={`w-full rounded-t ${i < 3 ? "bg-primary" : i < 6 ? "bg-accent" : "bg-destructive"}`}
                        style={{ height: `${h}%` }}
                      />
                      <span className="text-[9px] font-mono text-muted-foreground">{i}</span>
                    </div>
                  );
                })}
              </div>
              <div className="grid grid-cols-3 gap-2 text-[10px] font-mono pt-1">
                <Stat label="mean aspect" value={result.summary.tets.meanAspect.toFixed(2)} />
                <Stat label="worst aspect" value={result.summary.tets.worstAspect.toFixed(2)} />
                <Stat label="convergence" value={result.summary.tets.convergenceScore.toFixed(3)} />
              </div>
            </div>

            {/* Partition sizes + halos */}
            <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                partition balance · halo regions
              </div>
              <div className="space-y-1">
                {result.summary.partition.sizes.map((s, p) => {
                  const max = Math.max(...result.summary.partition.sizes);
                  const halo = result.summary.partition.haloSizes[p];
                  return (
                    <div key={p} className="flex items-center gap-2 text-[10px] font-mono">
                      <span className="w-8 text-muted-foreground">P{p}</span>
                      <div className="flex-1 h-2 bg-muted-foreground/15 rounded overflow-hidden flex">
                        <div className="bg-primary h-full" style={{ width: `${(s / max) * 100}%` }} />
                        <div className="bg-accent h-full opacity-70" style={{ width: `${(halo / max) * 100}%` }} />
                      </div>
                      <span className="w-12 text-right text-foreground/90 tabular-nums">{s}</span>
                      <span className="w-10 text-right text-accent tabular-nums">+{halo}</span>
                    </div>
                  );
                })}
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono pt-1">
                <Stat label="edge cut" value={result.summary.partition.edgeCut.toLocaleString()} />
                <Stat label="imbalance" value={result.summary.partition.imbalance.toFixed(3)} />
              </div>
            </div>
          </div>

          {/* Communication matrix */}
          {partitionView && partitionView.P > 1 && (
            <div className="rounded-md border border-border bg-background/30 p-3">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
                inter-partition communication matrix
              </div>
              <div
                className="grid gap-0.5"
                style={{ gridTemplateColumns: `repeat(${partitionView.P}, minmax(0, 1fr))` }}
              >
                {Array.from(partitionView.cm).map((v, i) => {
                  const max = Math.max(...partitionView.cm, 1);
                  const a = v / max;
                  return (
                    <div
                      key={i}
                      className="aspect-square rounded-sm border border-border/40"
                      style={{ background: `color-mix(in oklab, hsl(var(--primary)) ${a * 100}%, transparent)` }}
                      title={`P${Math.floor(i / partitionView.P)}→P${i % partitionView.P}: ${v}`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          <div className={`text-[11px] font-mono ${cls(result.summary.tets.convergenceScore)}`}>
            solver convergence proxy: {(result.summary.tets.convergenceScore * 100).toFixed(1)}%
            {result.summary.tets.convergenceScore >= 0.7 && " · stable for FEM/thermal"}
            {result.summary.tets.convergenceScore < 0.4 && " · re-mesh recommended"}
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border/60 px-2 py-1">
      <div className="uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className="font-mono text-foreground/90 tabular-nums">{value}</div>
    </div>
  );
}
