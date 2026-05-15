import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh, type RefinementSeed, type OctreeMesh } from "@/lib/meshing/octree";
import { buildAdjacency, type AdjacencyTensors } from "@/lib/meshing/adjacency";
import { partitionMesh, type PartitionPlan } from "@/lib/meshing/partition";
import { RefinementOverlay3D } from "@/components/RefinementOverlay3D";
import {
  runAdaptivePass,
  runDistributedRefinement,
  exportRefinedMesh,
  sharedPriorStore,
  physicsFeedbackBus,
  DEFAULT_WEIGHTS,
  type AdaptivePassResult,
  type DistributedAction,
  type RefinementExportFormat,
} from "@/lib/refinement";

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };

const PRESET_SEEDS: Record<string, RefinementSeed[]> = {
  "stress + thermal": [
    { kind: "sharp", center: [0.55, 0.4, 0], radius: 0.18, weight: 1 },
    { kind: "hotspot", center: [-0.4, 0.1, 0.3], radius: 0.25, weight: 0.9 },
    { kind: "contact", center: [0, -0.55, 0], radius: 0.15, weight: 0.7 },
  ],
  "deformation": [
    { kind: "overhang", center: [0.3, -0.4, 0.4], radius: 0.3, weight: 1 },
    { kind: "thin_wall", center: [-0.3, 0.3, -0.3], radius: 0.22, weight: 0.8 },
  ],
  "uniform bulk": [
    { kind: "fillet", center: [0, 0, 0], radius: 0.6, weight: 0.4 },
  ],
};

const FIELD_LABELS = ["stress", "thermal", "contact", "deform", "curve", "resid"];
const FIELD_COLORS = ["#ff5e7e", "#ffb347", "#7ad7ff", "#9b6bff", "#5cffb8", "#ffd34d"];

interface PassRow {
  step: number;
  baseLeaves: number;
  refinedLeaves: number;
  added: number;
  meanErr: number;
  p95Err: number;
  haloAdd: number;
  imbalance: number;
  ms: number;
  repart: boolean;
  source: "physics" | "synthetic" | "explicit";
}

export function RefinementPanel() {
  const [presetName, setPresetName] = useState<keyof typeof PRESET_SEEDS>("stress + thermal");
  const [splitThr, setSplitThr] = useState(0.45);
  const [extraDepth, setExtraDepth] = useState(2);
  const [partitions, setPartitions] = useState(8);
  const [feedbackMode, setFeedbackMode] = useState<"physics" | "auto" | "synthetic">("physics");
  const [distributed, setDistributed] = useState(false);
  const [imbThr, setImbThr] = useState(1.15);
  const [weights, setWeights] = useState<{ stress: number; thermal: number; deformation: number; contact: number }>({
    stress: 1.0, thermal: 0.6, deformation: 0.8, contact: 0.9,
  });
  const setWeight = (k: keyof typeof weights, v: number) => setWeights((w) => ({ ...w, [k]: v }));
  const [last, setLast] = useState<AdaptivePassResult | null>(null);
  const [history, setHistory] = useState<PassRow[]>([]);
  const [actions, setActions] = useState<DistributedAction[]>([]);
  const [running, setRunning] = useState(false);
  const [step, setStep] = useState(0);
  const [priorsCount, setPriorsCount] = useState(0);
  const [busTick, setBusTick] = useState(0);
  // Live mesh / partition / adjacency for distributed mode.
  const [liveMesh, setLiveMesh] = useState<OctreeMesh | null>(null);
  const [livePart, setLivePart] = useState<PartitionPlan | null>(null);
  const [liveAdj, setLiveAdj] = useState<AdjacencyTensors | null>(null);
  // Snapshot of (baseMesh, partition) used for the most recent pass — needed
  // so the export flow can build a refinement mask aligned to that base.
  const [lastBaseMesh, setLastBaseMesh] = useState<OctreeMesh | null>(null);
  const [lastPartition, setLastPartition] = useState<PartitionPlan | null>(null);

  // Re-render at 2 Hz so the snapshot age indicator stays current.
  useMemo(() => {
    const id = setInterval(() => setBusTick((t) => t + 1), 500);
    return () => clearInterval(id);
  }, []);
  void busTick;

  const seeds = PRESET_SEEDS[presetName];

  const baseSetup = useMemo(() => {
    const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
    const adj = buildAdjacency(mesh);
    const part = partitionMesh(mesh, adj, partitions);
    return { mesh, adj, part };
  }, [seeds, partitions]);

  const runPass = () => {
    setRunning(true);
    try {
      let r: AdaptivePassResult;
      let act: DistributedAction | null = null;
      if (distributed) {
        const mesh = liveMesh ?? baseSetup.mesh;
        const adj = liveAdj ?? baseSetup.adj;
        const part = livePart ?? baseSetup.part;
        const dr = runDistributedRefinement({
          bbox: BBOX,
          baseSeeds: seeds,
          mesh,
          partition: part,
          adjacency: adj,
          step,
          feedbackSource: feedbackMode,
          imbalanceThreshold: imbThr,
          options: { splitThreshold: splitThr, extraDepth, maxNewLeaves: 5000, weights: { ...DEFAULT_WEIGHTS, ...weights } },
        });
        r = dr.adaptive;
        act = dr.action;
        setLiveMesh(dr.mesh);
        setLivePart(dr.partition);
        setLiveAdj(dr.adjacency);
        setLastBaseMesh(mesh);
        setLastPartition(dr.partition);
        setActions((a) => [...a, dr.action].slice(-10));
      } else {
        r = runAdaptivePass({
          bbox: BBOX,
          baseSeeds: seeds,
          baseMesh: baseSetup.mesh,
          basePartition: baseSetup.part,
          step,
          feedbackSource: feedbackMode,
          options: { splitThreshold: splitThr, extraDepth, maxNewLeaves: 5000, weights: { ...DEFAULT_WEIGHTS, ...weights } },
        });
        setLastBaseMesh(baseSetup.mesh);
        setLastPartition(baseSetup.part);
      }
      setLast(r);
      setStep((s) => s + 1);
      setPriorsCount(sharedPriorStore().size());
      const row: PassRow = {
        step,
        baseLeaves: r.pass.baseLeafCount,
        refinedLeaves: r.pass.refinedLeafCount,
        added: r.pass.added,
        meanErr: r.error.combined.length
          ? Array.from(r.error.combined).reduce((a, b) => a + b, 0) / r.error.combined.length
          : 0,
        p95Err: r.pass.plan.stats.p95Error,
        haloAdd: r.haloDelta.addedHalo,
        imbalance: act ? act.imbalanceAfter : r.repartition.imbalance,
        ms: r.totalMs,
        repart: act ? act.repartitioned : r.shouldRepartition,
        source: r.fieldSource,
      };
      setHistory((h) => [...h, row].slice(-10));
    } finally {
      setRunning(false);
    }
  };

  const reset = () => {
    setHistory([]);
    setActions([]);
    setLast(null);
    setStep(0);
    setLiveMesh(null);
    setLivePart(null);
    setLiveAdj(null);
    setLastBaseMesh(null);
    setLastPartition(null);
    sharedPriorStore().clear();
    setPriorsCount(0);
  };

  const exportPass = (format: RefinementExportFormat) => {
    if (!last || !lastBaseMesh || !lastPartition) return;
    const file = exportRefinedMesh(
      { result: last, baseMesh: lastBaseMesh, partition: lastPartition, source: `physics-os/refinement/${presetName}` },
      format,
    );
    const blob = new Blob([file.content], { type: file.mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = file.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const dominantHist = useMemo(() => {
    const hist = new Array(6).fill(0);
    if (!last) return hist;
    for (let i = 0; i < last.error.dominant.length; i++) hist[last.error.dominant[i]]++;
    const max = Math.max(...hist, 1);
    return hist.map((v) => v / max);
  }, [last]);

  const priorTags = sharedPriorStore().summary().tagCounts;

  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.32em] text-muted-foreground">
            Adaptive Refinement Engine
          </div>
          <div className="font-display text-2xl text-foreground">
            Mesh follows physics. <span className="text-primary">Where it must.</span>
          </div>
        </div>
        <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          <FeedbackBadge mode={feedbackMode} last={last} />
          <button
            onClick={() => setFeedbackMode((m) => m === "physics" ? "auto" : m === "auto" ? "synthetic" : "physics")}
            className="rounded border border-border px-2 py-0.5 hover:text-foreground"
            title="Cycle Physics OS feedback mode: physics (require real solver fields) → auto (use real if fresh, else synthetic) → synthetic (mock fields only)"
          >
            feedback · <span className="text-primary">{feedbackMode}</span>
          </button>
          <button
            onClick={() => { setDistributed((d) => !d); reset(); }}
            className={`rounded border px-2 py-0.5 hover:text-foreground ${distributed ? "border-primary text-primary" : "border-border"}`}
            title="When ON, halo sync runs every pass and repartition fires automatically when imbalance > threshold"
          >
            distributed · <span className={distributed ? "text-primary" : "text-muted-foreground"}>{distributed ? "on" : "off"}</span>
          </button>
          <span>step <span className="text-primary tabular-nums">{step}</span> · priors <span className="text-accent tabular-nums">{priorsCount}</span></span>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">scenario</div>
          <div className="flex flex-wrap gap-1">
            {Object.keys(PRESET_SEEDS).map((k) => (
              <Button
                key={k}
                variant={k === presetName ? "default" : "outline"}
                className="text-[10px] uppercase tracking-[0.16em] h-7"
                onClick={() => { setPresetName(k as keyof typeof PRESET_SEEDS); reset(); }}
              >
                {k}
              </Button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span>split threshold</span><span className="text-primary tabular-nums">{splitThr.toFixed(2)}</span>
          </div>
          <Slider value={[splitThr]} min={0.1} max={0.9} step={0.05} onValueChange={([v]) => setSplitThr(v)} />
          <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span>extra depth</span><span className="text-primary tabular-nums">{extraDepth}</span>
          </div>
          <Slider value={[extraDepth]} min={0} max={4} step={1} onValueChange={([v]) => setExtraDepth(v)} />
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span>partitions</span><span className="text-primary tabular-nums">{partitions}</span>
          </div>
          <Slider value={[partitions]} min={1} max={16} step={1} onValueChange={([v]) => setPartitions(v)} />
          <div className={`flex justify-between text-[10px] uppercase tracking-[0.2em] ${distributed ? "text-muted-foreground" : "text-muted-foreground/40"}`}>
            <span>imbalance trigger</span><span className={distributed ? "text-primary tabular-nums" : "tabular-nums"}>{imbThr.toFixed(2)}×</span>
          </div>
          <Slider value={[imbThr]} min={1.05} max={2} step={0.05} onValueChange={([v]) => setImbThr(v)} disabled={!distributed} />
          <div className="flex gap-2 pt-2">
            <Button onClick={runPass} disabled={running} className="flex-1 uppercase tracking-[0.18em] text-[10px]">
              {running ? "refining…" : distributed ? "step distributed" : "run adaptive pass"}
            </Button>
            <Button onClick={reset} variant="outline" className="uppercase tracking-[0.18em] text-[10px]">
              reset
            </Button>
          </div>
          <div className="flex gap-2">
            <Button
              onClick={() => exportPass("json")}
              disabled={!last}
              variant="outline"
              className="flex-1 uppercase tracking-[0.18em] text-[10px]"
              title="Refined mesh + refinement mask + plan stats (Fabrication OS)"
            >
              ↓ json
            </Button>
            <Button
              onClick={() => exportPass("vtk")}
              disabled={!last}
              variant="outline"
              className="flex-1 uppercase tracking-[0.18em] text-[10px]"
              title="ParaView-compatible UnstructuredGrid with mask scalars"
            >
              ↓ vtk
            </Button>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-md border border-border bg-background/40 p-4 space-y-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            dominant field per leaf
          </div>
          <div className="space-y-1">
            {FIELD_LABELS.map((label, i) => (
              <div key={label} className="flex items-center gap-2 text-[10px] font-mono">
                <span className="w-12 text-muted-foreground uppercase tracking-[0.14em]">{label}</span>
                <div className="flex-1 h-2 rounded-sm bg-muted/40 overflow-hidden">
                  <div
                    className="h-full rounded-sm transition-all"
                    style={{ width: `${dominantHist[i] * 100}%`, background: FIELD_COLORS[i] }}
                  />
                </div>
                <span className="text-foreground/70 tabular-nums w-10 text-right">
                  {(dominantHist[i] * 100).toFixed(0)}%
                </span>
              </div>
            ))}
          </div>
          {last && (
            <div className="grid grid-cols-2 gap-2 pt-2 text-[10px] font-mono">
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">leaves</div>
                <div className="text-foreground tabular-nums">
                  {last.pass.baseLeafCount} → <span className="text-primary">{last.pass.refinedLeafCount}</span>
                </div>
              </div>
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">added</div>
                <div className="text-accent tabular-nums">+{last.pass.added}</div>
              </div>
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">p95 err</div>
                <div className="text-foreground tabular-nums">{last.pass.plan.stats.p95Error.toFixed(3)}</div>
              </div>
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">pass ms</div>
                <div className="text-foreground tabular-nums">{last.totalMs}</div>
              </div>
            </div>
          )}
        </div>

        <div className="rounded-md border border-border bg-background/40 p-4 space-y-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            partition · halo · repartition
          </div>
          {last ? (
            <>
              <div className="flex gap-1">
                {Array.from(last.repartition.newSizes).map((s, i) => {
                  const max = Math.max(...last.repartition.newSizes, 1);
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center gap-1">
                      <div className="w-full h-16 bg-muted/30 rounded-sm overflow-hidden flex items-end">
                        <div
                          className="w-full bg-primary/70 transition-all"
                          style={{ height: `${(s / max) * 100}%` }}
                        />
                      </div>
                      <div className="text-[9px] font-mono text-muted-foreground tabular-nums">{s}</div>
                    </div>
                  );
                })}
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">halo +</div>
                  <div className="text-accent tabular-nums">+{last.haloDelta.addedHalo}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">imbalance</div>
                  <div className={last.shouldRepartition ? "text-destructive tabular-nums" : "text-primary tabular-nums"}>
                    {last.repartition.imbalance.toFixed(3)}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">comm saved</div>
                  <div className="text-foreground tabular-nums">{last.repartition.commSaved}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">repartition</div>
                  <div className={last.shouldRepartition ? "text-destructive" : "text-muted-foreground"}>
                    {last.shouldRepartition ? "RECOMMENDED" : "stable"}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-[11px] text-muted-foreground/70">
              run a pass to see partition load and halo growth.
            </div>
          )}
        </div>
      </div>

      {last && (
        <PhysicsChannelStrip last={last} weights={weights} onWeightChange={setWeight} />
      )}

      {last && lastBaseMesh && (
        <RefinementOverlay3D baseMesh={lastBaseMesh} result={last} height={340} />
      )}

      {distributed && (
        <div className="rounded-md border border-primary/40 bg-primary/5 p-4 space-y-2">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span>distributed action log · trigger {imbThr.toFixed(2)}×</span>
            <span>
              repartitions · <span className="text-primary tabular-nums">{actions.filter((a) => a.repartitioned).length}</span>
              {" · "}halo syncs · <span className="text-accent tabular-nums">{actions.filter((a) => a.haloSynced).length}</span>
            </span>
          </div>
          {actions.length === 0 ? (
            <div className="text-[11px] text-muted-foreground/70 py-3 text-center">
              step the distributed engine — halo syncs and repartitions will be triggered automatically.
            </div>
          ) : (
            <div className="overflow-hidden rounded-sm border border-border">
              <table className="w-full text-[10px] font-mono">
                <thead className="bg-muted/30 text-muted-foreground uppercase tracking-[0.14em]">
                  <tr>
                    <th className="text-left px-2 py-1">step</th>
                    <th className="text-left px-2 py-1">reason</th>
                    <th className="text-right px-2 py-1">imb</th>
                    <th className="text-right px-2 py-1">migrated</th>
                    <th className="text-right px-2 py-1">halo bytes</th>
                    <th className="text-right px-2 py-1">rounds</th>
                    <th className="text-right px-2 py-1">µs</th>
                  </tr>
                </thead>
                <tbody>
                  {actions.map((a, i) => (
                    <tr key={i} className="odd:bg-background/30">
                      <td className="px-2 py-1 text-foreground tabular-nums">{a.step}</td>
                      <td className={`px-2 py-1 ${a.repartitioned ? "text-primary" : a.haloSynced ? "text-accent" : "text-muted-foreground"}`}>
                        {a.reason}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">
                        {a.imbalanceBefore.toFixed(2)}
                        {a.repartitioned && <span className="text-primary"> → {a.imbalanceAfter.toFixed(2)}</span>}
                      </td>
                      <td className="px-2 py-1 text-right tabular-nums">{a.migratedTets}</td>
                      <td className="px-2 py-1 text-right text-muted-foreground tabular-nums">{a.haloBytes.toLocaleString()}</td>
                      <td className="px-2 py-1 text-right tabular-nums">{a.haloRounds}</td>
                      <td className="px-2 py-1 text-right text-accent tabular-nums">{a.haloUs.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <div className="rounded-md border border-border bg-background/40 p-4 space-y-2">
        <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          <span>pass history</span>
          <span>priors · {Object.entries(priorTags).map(([k, v]) => `${k}:${v}`).join(" · ") || "none yet"}</span>
        </div>
        {history.length === 0 ? (
          <div className="text-[11px] text-muted-foreground/70 py-4 text-center">
            no passes yet — refinement priors accumulate per scenario.
          </div>
        ) : (
          <div className="overflow-hidden rounded-sm border border-border">
            <table className="w-full text-[10px] font-mono">
              <thead className="bg-muted/30 text-muted-foreground uppercase tracking-[0.14em]">
                <tr>
                  <th className="text-left px-2 py-1">step</th>
                  <th className="text-right px-2 py-1">leaves</th>
                  <th className="text-right px-2 py-1">+Δ</th>
                  <th className="text-right px-2 py-1">mean err</th>
                  <th className="text-right px-2 py-1">p95</th>
                  <th className="text-right px-2 py-1">halo+</th>
                  <th className="text-right px-2 py-1">imb</th>
                  <th className="text-right px-2 py-1">ms</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r, i) => (
                  <tr key={i} className="odd:bg-background/30">
                    <td className="px-2 py-1 text-foreground tabular-nums">{r.step}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.baseLeaves}→{r.refinedLeaves}</td>
                    <td className="px-2 py-1 text-right text-accent tabular-nums">+{r.added}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.meanErr.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.p95Err.toFixed(3)}</td>
                    <td className="px-2 py-1 text-right text-accent tabular-nums">{r.haloAdd}</td>
                    <td className={`px-2 py-1 text-right tabular-nums ${r.repart ? "text-destructive" : ""}`}>
                      {r.imbalance.toFixed(2)}
                    </td>
                    <td className="px-2 py-1 text-right text-muted-foreground tabular-nums">{r.ms}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function FeedbackBadge({ mode, last }: { mode: "physics" | "auto" | "synthetic"; last: AdaptivePassResult | null }) {
  const snap = physicsFeedbackBus.latest();
  const ageMs = snap ? Date.now() - snap.t : Infinity;
  const fresh = ageMs <= 1500;
  const willUsePhysics = (mode === "physics" || mode === "auto") && fresh;
  const lastSrc = last?.fieldSource ?? "—";
  const dotCol = willUsePhysics ? "bg-emerald-500" : mode === "synthetic" ? "bg-amber-500" : "bg-destructive";
  const txtCol = willUsePhysics ? "text-emerald-500" : mode === "synthetic" ? "text-amber-500" : "text-destructive";
  return (
    <span className="flex items-center gap-1.5">
      <span className={`inline-block h-1.5 w-1.5 rounded-full ${dotCol}`} />
      <span className={txtCol}>
        {snap ? `physics · ${snap.N}p · ${(ageMs / 1000).toFixed(1)}s` : "no physics snapshot"}
      </span>
      <span className="text-muted-foreground">last · {lastSrc}</span>
    </span>
  );
}

const CHANNEL_DEFS: Array<{ key: "stress" | "thermal" | "deformation" | "contact"; label: string; color: string; weightKey: keyof import("@/lib/refinement").ErrorWeights }> = [
  { key: "stress",      label: "stress",      color: "#ff5e7e", weightKey: "stress" },
  { key: "thermal",     label: "thermal",     color: "#ffb347", weightKey: "thermal" },
  { key: "deformation", label: "deformation", color: "#9b6bff", weightKey: "deformation" },
  { key: "contact",     label: "contact",     color: "#7ad7ff", weightKey: "contact" },
];

function fieldStats(arr: Float32Array): { mean: number; peak: number; coverage: number } {
  let s = 0, peak = 0, hits = 0;
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    s += v;
    if (v > peak) peak = v;
    if (v > 1e-6) hits++;
  }
  return { mean: arr.length ? s / arr.length : 0, peak, coverage: arr.length ? hits / arr.length : 0 };
}

type ChannelKey = "stress" | "thermal" | "deformation" | "contact";
const HISTORY_LEN = 32;

function Sparkline({ values, color, height = 18 }: { values: number[]; color: string; height?: number }) {
  if (values.length < 2) {
    return <div className="h-[18px] text-[8px] font-mono text-muted-foreground/60 flex items-center">collecting…</div>;
  }
  const max = Math.max(1e-6, ...values);
  const w = 100;
  const step = w / (HISTORY_LEN - 1);
  const pts = values.map((v, i) => {
    const x = (i + (HISTORY_LEN - values.length)) * step;
    const y = height - (v / max) * (height - 2) - 1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const last = values[values.length - 1];
  const lastX = (values.length - 1 + (HISTORY_LEN - values.length)) * step;
  const lastY = height - (last / max) * (height - 2) - 1;
  return (
    <svg viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" className="w-full h-[18px]">
      <polyline fill="none" stroke={color} strokeWidth={1} strokeLinejoin="round" points={pts} opacity={0.9} />
      <circle cx={lastX} cy={lastY} r={1.4} fill={color} />
    </svg>
  );
}

function PhysicsChannelStrip({ last }: { last: AdaptivePassResult }) {
  const realPhysics = last.fieldSource === "physics";
  const ageStr = last.snapshotAgeMs !== undefined ? `${(last.snapshotAgeMs / 1000).toFixed(2)}s` : "—";
  const N = last.snapshot?.N ?? 0;
  const contactCount = last.snapshot?.contacts.length ?? 0;

  // Per-channel ring buffer of peak strength across the last N adaptive passes.
  const [history, setHistory] = useState<Record<ChannelKey, number[]>>({
    stress: [], thermal: [], deformation: [], contact: [],
  });
  const lastRef = useRef<AdaptivePassResult | null>(null);
  useEffect(() => {
    if (lastRef.current === last) return;
    lastRef.current = last;
    setHistory((h) => {
      const next: Record<ChannelKey, number[]> = { ...h };
      for (const c of CHANNEL_DEFS) {
        const peak = fieldStats(last.fields[c.key]).peak;
        next[c.key] = [...h[c.key], peak].slice(-HISTORY_LEN);
      }
      return next;
    });
  }, [last]);

  return (
    <div className="rounded-md border border-border bg-background/40 p-4 space-y-3">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
        <span>physics OS feedback channels</span>
        <span className={realPhysics ? "text-emerald-500" : last.fieldSource === "synthetic" ? "text-amber-500" : "text-primary"}>
          source · {last.fieldSource}
          {realPhysics && <> · {N}p · age {ageStr} · contacts {contactCount}</>}
        </span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {CHANNEL_DEFS.map((c) => {
          const stats = fieldStats(last.fields[c.key]);
          const cover = Math.round(stats.coverage * 100);
          const live = stats.peak > 1e-6;
          const series = history[c.key];
          return (
            <div key={c.key} className="rounded-sm border border-border bg-background/30 p-2 space-y-1">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <span className="uppercase tracking-[0.14em]" style={{ color: c.color }}>{c.label}</span>
                <span className={live ? "text-foreground" : "text-muted-foreground"}>{live ? "live" : "—"}</span>
              </div>
              <div className="h-1.5 rounded-sm bg-muted/40 overflow-hidden">
                <div className="h-full rounded-sm" style={{ width: `${cover}%`, background: c.color }} />
              </div>
              <Sparkline values={series} color={c.color} />
              <div className="flex justify-between text-[9px] font-mono text-muted-foreground tabular-nums">
                <span>cover {cover}%</span>
                <span>peak {stats.peak.toFixed(2)}</span>
              </div>
              <div className="flex justify-between text-[8px] font-mono text-muted-foreground/70 tabular-nums">
                <span>last {series.length}/{HISTORY_LEN}</span>
                <span>max {(Math.max(0, ...series)).toFixed(2)}</span>
              </div>
            </div>
          );
        })}
      </div>
      {!realPhysics && (
        <div className="text-[10px] text-amber-500/80 font-mono">
          ⚠ refinement metric is using {last.fieldSource} fields — start the physics canvas (or set feedback · physics) to drive the metric from real solver state.
        </div>
      )}
    </div>
  );
}
