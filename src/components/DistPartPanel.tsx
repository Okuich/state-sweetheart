import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { buildOctreeMesh, type RefinementSeed } from "@/lib/meshing/octree";
import { buildAdjacency } from "@/lib/meshing/adjacency";
import {
  planDistributed,
  type PartitionAlgorithm,
  type DistPartResult,
} from "@/lib/distpart";

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };
const SEEDS: RefinementSeed[] = [
  { kind: "sharp", center: [0.5, 0.4, 0], radius: 0.2, weight: 1 },
  { kind: "hotspot", center: [-0.4, 0.1, 0.3], radius: 0.25, weight: 0.9 },
  { kind: "contact", center: [0, -0.55, 0], radius: 0.18, weight: 0.7 },
];

const ALGOS: PartitionAlgorithm[] = ["regionGrow", "morton", "spectral", "kway"];
const ALGO_LABEL: Record<PartitionAlgorithm, string> = {
  regionGrow: "BFS region-grow",
  morton: "Z-order strip",
  spectral: "spectral bisect",
  kway: "k-way refine",
};

type CommSort = "none" | "haloOut" | "haloIn";

export function DistPartPanel() {
  const [algo, setAlgo] = useState<PartitionAlgorithm>("kway");
  const [P, setP] = useState(8);
  const [skew, setSkew] = useState(3);
  const [running, setRunning] = useState(false);
  const [last, setLast] = useState<DistPartResult | null>(null);
  const [commSort, setCommSort] = useState<CommSort>("none");
  const [topK, setTopK] = useState(0); // 0 = off, otherwise highlight K hottest edges
  const [drill, setDrill] = useState<{ src: number; dst: number } | null>(null);
  const [history, setHistory] = useState<{ algo: PartitionAlgorithm; P: number; cut: number; imb: number; us: number; bytes: number; rounds: number }[]>([]);

  const setup = useMemo(() => {
    const mesh = buildOctreeMesh(BBOX, SEEDS, { maxDepth: 4, minDepth: 2 });
    const adj = buildAdjacency(mesh);
    return { mesh, adj };
  }, []);

  const run = () => {
    setRunning(true);
    try {
      const T = setup.mesh.tets.length / 4;
      const weights = new Float32Array(T).fill(1);
      // Skew load on partition-0-ish region (left half).
      for (let t = 0; t < T; t++) {
        const i0 = setup.mesh.tets[t * 4] * 3;
        if (setup.mesh.vertices[i0] < 0) weights[t] = skew;
      }
      const r = planDistributed({
        mesh: setup.mesh,
        adj: setup.adj,
        partitionCount: P,
        algorithm: algo,
        rebalance: true,
        weights,
        checkpointStep: history.length,
        traversal: true,
        haloSyncIterations: 16,
        broadphase: true,
        schedule: { batchSize: 4, coalesceCap: 4, deltaCompressionRatio: 0.4 },
      });
      setLast(r);
      setHistory((h) =>
        [
          ...h,
          {
            algo: r.algorithm,
            P: r.partitionCount,
            cut: r.assignment.stats.edgeCut,
            imb: (r.rebalanced ?? r.assignment).stats.imbalance,
            us: r.latency.totalUs,
            bytes: r.halo.syncBytes,
            rounds: r.halo.rounds.length,
          },
        ].slice(-12),
      );
    } finally {
      setRunning(false);
    }
  };

  const exportCsv = () => {
    if (!last) return;
    const P = last.partitionCount;
    const sizes = Array.from((last.rebalanced ?? last.assignment).stats.sizes);
    const cm = last.halo.commMatrix;
    const lines: string[] = [];
    lines.push(`# distpart export · algo=${last.algorithm} · P=${P}`);
    lines.push(`# edgeCut=${last.assignment.stats.edgeCut} · imbalance=${last.assignment.stats.imbalance.toFixed(4)}` +
      (last.rebalanced ? ` → ${last.rebalanced.stats.imbalance.toFixed(4)}` : ""));
    lines.push(`# syncBytes=${last.halo.syncBytes} · rounds=${last.halo.rounds.length} · totalUs=${last.latency.totalUs.toFixed(2)}`);
    lines.push("");
    lines.push("# per-partition stats");
    lines.push("rank,tets,haloOutBytes,haloInBytes");
    for (let r = 0; r < P; r++) {
      let out = 0, inn = 0;
      for (let c = 0; c < P; c++) {
        out += cm[r * P + c];
        inn += cm[c * P + r];
      }
      lines.push(`${r},${sizes[r] ?? 0},${out},${inn}`);
    }
    lines.push("");
    lines.push("# comm matrix · rows=sender, cols=receiver, values=halo bytes");
    lines.push(["sender\\receiver", ...Array.from({ length: P }, (_, c) => `r${c}`)].join(","));
    for (let r = 0; r < P; r++) {
      const row = [`r${r}`];
      for (let c = 0; c < P; c++) row.push(String(cm[r * P + c]));
      lines.push(row.join(","));
    }
    const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `distpart-${last.algorithm}-P${P}-${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const sizes = last ? Array.from((last.rebalanced ?? last.assignment).stats.sizes) : [];
  const maxSize = Math.max(1, ...sizes);
  const commMatrix = last ? last.halo.commMatrix : new Uint32Array(0);
  const commMax = last ? Math.max(1, ...Array.from(commMatrix)) : 1;

  // Permutation of ranks for the comm matrix display. Reorder rows AND
  // cols by total halo OUT (row sum) or halo IN (col sum) so the most
  // chatty ranks cluster top-left and structural hotspots pop out.
  const Plast = last?.partitionCount ?? 0;
  const commPerm = useMemo(() => {
    const perm = new Int32Array(Plast);
    for (let i = 0; i < Plast; i++) perm[i] = i;
    if (Plast === 0 || commSort === "none") return perm;
    const score = new Float64Array(Plast);
    for (let r = 0; r < Plast; r++) {
      let s = 0;
      for (let c = 0; c < Plast; c++) {
        s += commSort === "haloOut"
          ? commMatrix[r * Plast + c]   // bytes sent FROM r
          : commMatrix[c * Plast + r];  // bytes received BY r
      }
      score[r] = s;
    }
    return new Int32Array(
      Array.from(perm).sort((a, b) => score[b] - score[a]),
    );
  }, [commMatrix, commSort, Plast]);

  // Top-K threshold: the K-th largest off-diagonal nonzero edge weight.
  // Cells with v >= threshold are highlighted; others get dimmed.
  const topKThreshold = useMemo(() => {
    if (!Plast || topK <= 0) return 0;
    const vals: number[] = [];
    for (let r = 0; r < Plast; r++) {
      for (let c = 0; c < Plast; c++) {
        if (r === c) continue;
        const v = commMatrix[r * Plast + c];
        if (v > 0) vals.push(v);
      }
    }
    if (!vals.length) return 0;
    vals.sort((a, b) => b - a);
    return vals[Math.min(topK, vals.length) - 1];
  }, [commMatrix, topK, Plast]);
  const topKOptions = [0, 4, 8, 16, 32];


  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.32em] text-muted-foreground">
            Distributed Partitioning Engine
          </div>
          <div className="font-display text-2xl text-foreground">
            Geometry, sliced. <span className="text-primary">Comm, scheduled.</span>
          </div>
        </div>
        <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          tets · <span className="text-primary tabular-nums">{setup.mesh.tets.length / 4}</span>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">algorithm</div>
          <div className="flex flex-wrap gap-1">
            {ALGOS.map((a) => (
              <Button
                key={a}
                variant={a === algo ? "default" : "outline"}
                className="text-[10px] uppercase tracking-[0.16em] h-7"
                onClick={() => setAlgo(a)}
              >
                {ALGO_LABEL[a]}
              </Button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span>partitions (ranks)</span><span className="text-primary tabular-nums">{P}</span>
          </div>
          <Slider value={[P]} min={2} max={16} step={1} onValueChange={([v]) => setP(v)} />
          <div className="flex justify-between text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            <span>load skew (left half)</span><span className="text-primary tabular-nums">{skew}×</span>
          </div>
          <Slider value={[skew]} min={1} max={8} step={0.5} onValueChange={([v]) => setSkew(v)} />
        </div>
        <div className="space-y-2">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">execute</div>
          <Button onClick={run} disabled={running} className="w-full uppercase tracking-[0.18em] text-[10px]">
            {running ? "partitioning…" : "plan distribution"}
          </Button>
          <Button
            onClick={exportCsv}
            disabled={!last}
            variant="outline"
            className="w-full uppercase tracking-[0.18em] text-[10px]"
            title="Download per-partition stats and comm matrix as CSV"
          >
            export csv
          </Button>
          {last && (
            <div className="grid grid-cols-2 gap-2 text-[10px] font-mono pt-1">
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">edge cut</div>
                <div className="text-foreground tabular-nums">{last.assignment.stats.edgeCut}</div>
              </div>
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">imbalance</div>
                <div className="text-foreground tabular-nums">
                  {last.assignment.stats.imbalance.toFixed(2)}
                  {last.rebalanced && (
                    <span className="text-primary ml-1">→ {last.rebalanced.stats.imbalance.toFixed(2)}</span>
                  )}
                </div>
              </div>
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">comm rounds</div>
                <div className="text-accent tabular-nums">{last.halo.rounds.length}</div>
              </div>
              <div>
                <div className="text-muted-foreground uppercase tracking-[0.14em]">sync µs</div>
                <div className="text-accent tabular-nums">{last.latency.totalUs.toFixed(1)}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-md border border-border bg-background/40 p-4 space-y-3">
          <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
            per-rank load · post-rebalance
          </div>
          {sizes.length ? (
            <div className="flex gap-1 items-end h-24">
              {sizes.map((s, i) => (
                <div key={i} className="flex-1 flex flex-col items-center gap-1">
                  <div className="w-full h-20 bg-muted/30 rounded-sm overflow-hidden flex items-end">
                    <div className="w-full bg-primary/70" style={{ height: `${(s / maxSize) * 100}%` }} />
                  </div>
                  <div className="text-[9px] font-mono text-muted-foreground tabular-nums">{s}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-[11px] text-muted-foreground/70">no plan yet.</div>
          )}
          {last?.rebalance && (
            <div className="text-[10px] font-mono text-muted-foreground">
              migrations · <span className="text-foreground">{last.rebalance.moves.length}</span>
              {" · "}migrate bytes · <span className="text-foreground">{last.rebalance.migrationBytes.toLocaleString()}</span>
              {" · "}makespan · <span className="text-foreground">{last.rebalance.beforeMakespan.toFixed(1)} → {last.rebalance.afterMakespan.toFixed(1)}</span>
            </div>
          )}
        </div>

        <div className="rounded-md border border-border bg-background/40 p-4 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              comm matrix · cross-rank halo (NCCL)
            </div>
            <div className="flex gap-1" role="group" aria-label="Sort partitions">
              {([
                ["none", "rank"],
                ["haloOut", "halo↑"],
                ["haloIn", "halo↓"],
              ] as [CommSort, string][]).map(([k, label]) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setCommSort(k)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-[0.14em] border transition-colors ${
                    commSort === k
                      ? "border-primary text-primary bg-primary/10"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                  title={
                    k === "none" ? "Default rank order" :
                    k === "haloOut" ? "Sort by halo bytes sent (row sum) desc" :
                    "Sort by halo bytes received (column sum) desc"
                  }
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-between gap-2">
            <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
              top-K edges
            </div>
            <div className="flex gap-1" role="group" aria-label="Top-K edge highlight">
              {topKOptions.map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setTopK(k)}
                  className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-[0.14em] border transition-colors ${
                    topK === k
                      ? "border-accent text-accent bg-accent/10"
                      : "border-border text-muted-foreground hover:text-foreground"
                  }`}
                  title={k === 0 ? "Show all edges" : `Highlight top ${k} sender→receiver edges`}
                >
                  {k === 0 ? "off" : `K=${k}`}
                </button>
              ))}
            </div>
          </div>
          {last ? (
            <div className="grid gap-[2px]" style={{ gridTemplateColumns: `repeat(${last.partitionCount}, minmax(0, 1fr))` }}>
              {Array.from({ length: Plast * Plast }, (_, idx) => {
                const rDisp = Math.floor(idx / Plast);
                const cDisp = idx % Plast;
                const rOrig = commPerm[rDisp];
                const cOrig = commPerm[cDisp];
                const v = commMatrix[rOrig * Plast + cOrig];
                const intensity = v / commMax;
                const filtered = topK > 0 && (v === 0 || rOrig === cOrig || v < topKThreshold);
                let bg: string;
                if (v === 0) {
                  bg = "hsl(var(--muted) / 0.2)";
                } else if (filtered) {
                  bg = `hsl(var(--muted) / ${0.15 + intensity * 0.2})`;
                } else if (topK > 0) {
                  bg = `hsl(var(--accent) / ${0.35 + intensity * 0.65})`;
                } else {
                  bg = `hsl(var(--primary) / ${0.15 + intensity * 0.85})`;
                }
                return (
                  <div
                    key={idx}
                    className="aspect-square rounded-[1px]"
                    style={{ background: bg, opacity: filtered ? 0.55 : 1 }}
                    title={`rank ${rOrig} → ${cOrig}: ${v}${filtered && topK > 0 ? " (below top-K)" : ""}`}
                  />
                );
              })}
            </div>
          ) : (
            <div className="text-[11px] text-muted-foreground/70">no plan yet.</div>
          )}
          {last && (
            <div className="text-[10px] font-mono text-muted-foreground">
              sync bytes · <span className="text-foreground">{last.halo.syncBytes.toLocaleString()}</span>
              {" · "}rounds · <span className="text-foreground">{last.halo.rounds.length}</span>
              {last.checkpoint && (
                <> · ckpt root · <span className="text-accent">{last.checkpoint.rootDigest.toString(16).padStart(8, "0")}</span></>
              )}
            </div>
          )}
        </div>
      </div>

      {last && (last.traversal || last.haloSync || last.broadphase) && (
        <div className="grid gap-4 md:grid-cols-3">
          {last.traversal && (
            <div className="rounded-md border border-border bg-background/40 p-4 space-y-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                partition-aware traversal
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">super-steps</div>
                  <div className="text-foreground tabular-nums">{last.traversal.steps.length}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">speedup</div>
                  <div className="text-primary tabular-nums">{last.traversal.speedup.toFixed(2)}×</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">visited</div>
                  <div className="text-foreground tabular-nums">{last.traversal.totalVisited}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">est. wall µs</div>
                  <div className="text-accent tabular-nums">{last.traversal.estimatedUs.toFixed(1)}</div>
                </div>
              </div>
              <div className="flex gap-[2px] items-end h-12 pt-1">
                {last.traversal.steps.map((s, i) => {
                  const max = Math.max(1, ...last.traversal!.steps.map((x) => x.parallelWork));
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-primary/70 rounded-[1px]"
                      style={{ height: `${(s.parallelWork / max) * 100}%` }}
                      title={`step ${s.step}: parallel ${s.parallelWork} / serial ${s.serialWork}, halo ${s.haloTets} tets`}
                    />
                  );
                })}
              </div>
              <div className="text-[9px] font-mono text-muted-foreground">
                halo bytes · <span className="text-foreground">{last.traversal.totalHaloBytes.toLocaleString()}</span>
              </div>
              {last.traversal.batched && (
                <div className="text-[9px] font-mono text-muted-foreground border-t border-border/40 pt-1 mt-1 space-y-0.5">
                  <div>
                    batched · <span className="text-primary">{last.traversal.batched.flushes}</span> flushes
                    {" · "}<span className="text-primary">{last.traversal.batched.messages}</span> launches
                    {" · "}<span className="text-foreground">{last.traversal.batched.bytes.toLocaleString()}</span> B
                  </div>
                  <div>
                    sync µs · <span className="text-accent">{last.traversal.batched.us.toFixed(1)}</span>
                    {" · "}max staleness · <span className="text-foreground">{last.traversal.batched.maxStaleness}</span> steps
                  </div>
                </div>
              )}
            </div>
          )}

          {last.haloSync && (
            <div className="rounded-md border border-border bg-background/40 p-4 space-y-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                halo sync · {last.haloSync.iterations.length} iters
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">avg µs</div>
                  <div className="text-foreground tabular-nums">{last.haloSync.avgUs.toFixed(1)}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">total bytes</div>
                  <div className="text-foreground tabular-nums">{last.haloSync.totalBytes.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">eff. GB/s</div>
                  <div className="text-primary tabular-nums">{last.haloSync.effectiveGBs.toFixed(1)}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">staleness</div>
                  <div className="text-accent tabular-nums">{last.haloSync.maxStaleness}</div>
                </div>
              </div>
              <div className="flex gap-[2px] items-end h-12 pt-1">
                {last.haloSync.iterations.map((it, i) => {
                  const max = Math.max(1, ...last.haloSync!.iterations.map((x) => x.iterUs));
                  return (
                    <div
                      key={i}
                      className="flex-1 bg-accent/70 rounded-[1px]"
                      style={{ height: `${(it.iterUs / max) * 100}%` }}
                      title={`iter ${it.iter}: ${it.iterUs.toFixed(1)} µs, ${it.packets} pkts`}
                    />
                  );
                })}
              </div>
            </div>
          )}

          {last.broadphase && (
            <div className="rounded-md border border-border bg-background/40 p-4 space-y-2">
              <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
                partition-aware broadphase
              </div>
              <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">grid</div>
                  <div className="text-foreground tabular-nums">{last.broadphase.gridResolution}³</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">prune rate</div>
                  <div className="text-primary tabular-nums">{(last.broadphase.pruningRate * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">local pairs</div>
                  <div className="text-foreground tabular-nums">{last.broadphase.totalLocalPairs}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">cross pairs</div>
                  <div className="text-accent tabular-nums">{last.broadphase.uniqueCrossPairs}</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">halo cov.</div>
                  <div className="text-primary tabular-nums">{(last.broadphase.haloCoverage * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <div className="text-muted-foreground uppercase tracking-[0.14em]">build ms</div>
                  <div className="text-foreground tabular-nums">{last.broadphase.buildMs}</div>
                </div>
              </div>
              <div className="flex gap-[2px] items-end h-12 pt-1">
                {last.broadphase.perRank.map((r, i) => {
                  const max = Math.max(1, ...last.broadphase!.perRank.map((x) => x.local + x.ghost));
                  return (
                    <div key={i} className="flex-1 flex flex-col-reverse rounded-[1px] overflow-hidden">
                      <div className="bg-primary/70" style={{ height: `${(r.local / max) * 100}%` }} title={`rank ${r.rank} local ${r.local}`} />
                      <div className="bg-accent/70" style={{ height: `${(r.ghost / max) * 100}%` }} title={`rank ${r.rank} ghost ${r.ghost}`} />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {last?.schedule && (
        <div className="rounded-md border border-border bg-background/40 p-4 space-y-3">
          <div className="flex items-baseline justify-between">
            <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
              comm-minimization scheduler · batch {last.schedule.batchSize} · {last.schedule.iterations} iters
            </div>
            <div className="text-[10px] font-mono text-muted-foreground">
              {last.schedule.batches} batches · {last.schedule.rounds.length} rounds
            </div>
          </div>
          <div className="grid grid-cols-3 gap-3 text-[10px] font-mono">
            <div className="rounded-sm border border-border/60 p-2 space-y-1">
              <div className="text-muted-foreground uppercase tracking-[0.14em]">messages</div>
              <div className="flex items-baseline gap-2">
                <span className="text-foreground tabular-nums">{last.schedule.batched.messages.toLocaleString()}</span>
                <span className="text-muted-foreground/70 line-through tabular-nums">{last.schedule.baseline.messages.toLocaleString()}</span>
              </div>
              <div className="text-primary tabular-nums">−{(last.schedule.savings.messages * 100).toFixed(1)}%</div>
            </div>
            <div className="rounded-sm border border-border/60 p-2 space-y-1">
              <div className="text-muted-foreground uppercase tracking-[0.14em]">bytes</div>
              <div className="flex items-baseline gap-2">
                <span className="text-foreground tabular-nums">{last.schedule.batched.bytes.toLocaleString()}</span>
                <span className="text-muted-foreground/70 line-through tabular-nums">{last.schedule.baseline.bytes.toLocaleString()}</span>
              </div>
              <div className="text-primary tabular-nums">−{(last.schedule.savings.bytes * 100).toFixed(1)}%</div>
            </div>
            <div className="rounded-sm border border-border/60 p-2 space-y-1">
              <div className="text-muted-foreground uppercase tracking-[0.14em]">latency µs</div>
              <div className="flex items-baseline gap-2">
                <span className="text-foreground tabular-nums">{last.schedule.batched.totalUs.toFixed(0)}</span>
                <span className="text-muted-foreground/70 line-through tabular-nums">{last.schedule.baseline.totalUs.toFixed(0)}</span>
              </div>
              <div className="text-accent tabular-nums">−{(last.schedule.savings.latency * 100).toFixed(1)}%</div>
            </div>
          </div>
          <div>
            <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground mb-1">
              per-round bytes (parallel max)
            </div>
            <div className="flex gap-[2px] items-end h-14">
              {last.schedule.rounds.map((r, i) => {
                const max = Math.max(1, ...last.schedule!.rounds.map((x) => x.parallelBytes));
                return (
                  <div
                    key={i}
                    className="flex-1 bg-primary/70 rounded-[1px]"
                    style={{ height: `${(r.parallelBytes / max) * 100}%` }}
                    title={`round ${r.index}: ${r.transfers.length} transfers, ${r.parallelBytes.toLocaleString()}B, ${r.roundUs.toFixed(1)}µs`}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}

      <div className="rounded-md border border-border bg-background/40 p-4 space-y-2">
        <div className="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          algorithm history (last 12)
        </div>
        {history.length === 0 ? (
          <div className="text-[11px] text-muted-foreground/70 py-3 text-center">
            run multiple algorithms to compare cut / imbalance / latency.
          </div>
        ) : (
          <div className="overflow-hidden rounded-sm border border-border">
            <table className="w-full text-[10px] font-mono">
              <thead className="bg-muted/30 text-muted-foreground uppercase tracking-[0.14em]">
                <tr>
                  <th className="text-left px-2 py-1">algorithm</th>
                  <th className="text-right px-2 py-1">P</th>
                  <th className="text-right px-2 py-1">cut</th>
                  <th className="text-right px-2 py-1">imb</th>
                  <th className="text-right px-2 py-1">rounds</th>
                  <th className="text-right px-2 py-1">µs</th>
                  <th className="text-right px-2 py-1">bytes</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r, i) => (
                  <tr key={i} className="odd:bg-background/30">
                    <td className="px-2 py-1 text-foreground">{ALGO_LABEL[r.algo]}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.P}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.cut}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.imb.toFixed(2)}</td>
                    <td className="px-2 py-1 text-right text-accent tabular-nums">{r.rounds}</td>
                    <td className="px-2 py-1 text-right tabular-nums">{r.us.toFixed(1)}</td>
                    <td className="px-2 py-1 text-right text-muted-foreground tabular-nums">{r.bytes.toLocaleString()}</td>
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
