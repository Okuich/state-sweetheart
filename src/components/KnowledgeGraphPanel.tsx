// Geometry-Physics Knowledge Graph — UI panel
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  addEdge,
  addNode,
  causalPath,
  emptyGraph,
  knn,
  loadGraph,
  recomputeSimilarity,
  removeNode,
  saveGraph,
  seedIndustrial,
  stats,
  type EdgeKind,
  type Graph,
  type GraphNode,
  type NodeKind,
} from "@/lib/knowledgeGraph";

const KIND_COLOR: Record<NodeKind, string> = {
  geometry: "hsl(var(--primary))",
  material: "hsl(var(--accent))",
  process:  "hsl(40 90% 60%)",
  sim:      "hsl(200 80% 65%)",
  outcome:  "hsl(var(--destructive))",
};

const EDGE_COLOR: Record<EdgeKind, string> = {
  causal:           "hsl(var(--destructive) / 0.8)",
  similar:          "hsl(var(--primary) / 0.55)",
  "optimized-from": "hsl(var(--accent))",
};

export function KnowledgeGraphPanel() {
  const [g, setG] = useState<Graph>(() => emptyGraph());
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [pathEnds, setPathEnds] = useState<{ from?: string; to?: string }>({});

  // hydrate from storage on mount; seed if empty
  useEffect(() => {
    const loaded = loadGraph();
    if (loaded.nodes.length === 0) {
      seedIndustrial(loaded);
      saveGraph(loaded);
    }
    setG({ ...loaded });
  }, []);

  const persist = (next: Graph) => {
    saveGraph(next);
    setG({ ...next });
  };

  const s = useMemo(() => stats(g), [g]);
  const selected = useMemo(
    () => g.nodes.find((n) => n.id === selectedId) ?? null,
    [g, selectedId]
  );
  const neighbors = useMemo(
    () => (selected ? knn(g, selected.id, 5) : []),
    [g, selected]
  );
  const path = useMemo(() => {
    if (pathEnds.from && pathEnds.to) return causalPath(g, pathEnds.from, pathEnds.to);
    return [];
  }, [g, pathEnds]);

  // simple deterministic layout: angle by kind cluster + radius by created order
  const positions = useMemo(() => {
    const groups: Record<NodeKind, GraphNode[]> = {
      geometry: [], material: [], process: [], sim: [], outcome: [],
    };
    for (const n of g.nodes) groups[n.kind].push(n);
    const order: NodeKind[] = ["geometry", "material", "process", "sim", "outcome"];
    const map = new Map<string, { x: number; y: number }>();
    const W = 720, H = 360, cx = W / 2, cy = H / 2;
    order.forEach((k, ki) => {
      const arr = groups[k];
      const angle0 = (ki / order.length) * Math.PI * 2 - Math.PI / 2;
      arr.forEach((n, i) => {
        const r = 80 + (i % 4) * 30;
        const a = angle0 + (i / Math.max(1, arr.length)) * (Math.PI / 2.2) - Math.PI / 4;
        map.set(n.id, { x: cx + Math.cos(a) * r, y: cy + Math.sin(a) * r });
      });
    });
    return map;
  }, [g]);

  // ── actions ────────────────────────────────────────────────
  const handleSeed = () => {
    const fresh = emptyGraph();
    seedIndustrial(fresh);
    persist(fresh);
    setSelectedId(null);
    setPathEnds({});
  };
  const handleClear = () => {
    persist(emptyGraph());
    setSelectedId(null);
    setPathEnds({});
  };
  const handleResim = () => {
    const next = { ...g };
    recomputeSimilarity(next);
    persist(next);
  };
  const handleAddObservation = () => {
    // synthetic new sim + outcome attached to first geometry/material/process
    const next = { ...g };
    const geo = next.nodes.find((n) => n.kind === "geometry");
    const mat = next.nodes.find((n) => n.kind === "material");
    const proc = next.nodes.find((n) => n.kind === "process");
    if (!geo || !mat || !proc) return;
    const stress = Math.round(60 + Math.random() * 320);
    const sim = addNode(next, "sim", `run-${next.nodes.filter((n) => n.kind === "sim").length + 1}`, {
      maxStress: stress,
      energyDrift: +(Math.random() * 0.02).toFixed(4),
    });
    addEdge(next, "causal", geo.id, sim.id, 0.7);
    addEdge(next, "causal", mat.id, sim.id, 0.6);
    const ok = stress < 300;
    const out = addNode(next, "outcome", `${proc.label}/${ok ? "pass" : "fail"}`, {
      pass: ok ? 1 : 0,
      defectRate: +(Math.random() * 0.15).toFixed(3),
    });
    addEdge(next, "causal", proc.id, out.id, 0.8);
    addEdge(next, "causal", sim.id, out.id, 0.55);
    recomputeSimilarity(next);
    persist(next);
  };
  const handleDelete = () => {
    if (!selected) return;
    const next = { ...g };
    removeNode(next, selected.id);
    persist(next);
    setSelectedId(null);
  };
  const setEnd = (which: "from" | "to") => {
    if (!selected) return;
    setPathEnds((p) => ({ ...p, [which]: selected.id }));
  };

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
            module · knowledge-graph
          </div>
          <h2 className="font-display text-2xl md:text-3xl text-glow">
            Geometry-Physics <span className="text-primary">Knowledge</span> Graph
          </h2>
          <p className="text-xs text-muted-foreground max-w-xl mt-1">
            Persistent industrial graph linking motifs · materials · processes ·
            simulations · outcomes via causal, similarity, and optimization edges.
            Stored locally in <code className="text-primary">localStorage</code>.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={handleAddObservation} className="uppercase tracking-[0.16em] text-[10px]">
            + observation
          </Button>
          <Button variant="outline" onClick={handleResim} className="uppercase tracking-[0.16em] text-[10px]">
            recompute · similarity
          </Button>
          <Button variant="outline" onClick={handleSeed} className="uppercase tracking-[0.16em] text-[10px]">
            reseed
          </Button>
          <Button variant="outline" onClick={handleClear} className="uppercase tracking-[0.16em] text-[10px]">
            clear
          </Button>
        </div>
      </header>

      {/* Stats strip */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-2 text-[11px]">
        <Stat label="nodes" value={s.nodes} />
        <Stat label="edges" value={s.edges} />
        <Stat label="components" value={s.components} />
        <Stat label="avg degree" value={s.avgDegree.toFixed(2)} />
        <Stat label="density" value={s.density.toFixed(3)} />
        <Stat label="similar / causal" value={`${s.byEdge.similar} / ${s.byEdge.causal}`} />
      </div>

      {/* Graph svg + sidebar */}
      <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
        <div className="rounded-lg border border-border bg-background/40 p-2">
          <svg viewBox="0 0 720 360" className="w-full h-[360px]">
            {/* edges */}
            {g.edges.map((e) => {
              const a = positions.get(e.from);
              const b = positions.get(e.to);
              if (!a || !b) return null;
              const onPath =
                path.length > 1 &&
                path.includes(e.from) &&
                path.includes(e.to) &&
                Math.abs(path.indexOf(e.from) - path.indexOf(e.to)) === 1;
              return (
                <line
                  key={e.id}
                  x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                  stroke={onPath ? "hsl(var(--accent))" : EDGE_COLOR[e.kind]}
                  strokeWidth={onPath ? 2.2 : 0.8 + e.weight * 1.4}
                  strokeDasharray={e.kind === "similar" ? "3 3" : undefined}
                  opacity={onPath ? 1 : 0.85}
                />
              );
            })}
            {/* nodes */}
            {g.nodes.map((n) => {
              const p = positions.get(n.id);
              if (!p) return null;
              const isSel = n.id === selectedId;
              const isFrom = n.id === pathEnds.from;
              const isTo   = n.id === pathEnds.to;
              return (
                <g
                  key={n.id}
                  transform={`translate(${p.x},${p.y})`}
                  onClick={() => setSelectedId(n.id)}
                  className="cursor-pointer"
                >
                  <circle
                    r={isSel ? 8 : 6}
                    fill={KIND_COLOR[n.kind]}
                    stroke={isFrom ? "hsl(var(--accent))" : isTo ? "hsl(var(--destructive))" : "hsl(var(--background))"}
                    strokeWidth={isFrom || isTo ? 2 : 1}
                  />
                  <text
                    y={-10}
                    textAnchor="middle"
                    fontSize={8}
                    fill="hsl(var(--foreground))"
                    className="font-mono pointer-events-none"
                  >
                    {n.label}
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="flex flex-wrap gap-3 px-2 pt-2 text-[10px] uppercase tracking-[0.18em]">
            {(Object.keys(KIND_COLOR) as NodeKind[]).map((k) => (
              <span key={k} className="flex items-center gap-1.5 text-muted-foreground">
                <span className="h-2 w-2 rounded-full" style={{ background: KIND_COLOR[k] }} />
                {k}
              </span>
            ))}
            <span className="ml-auto text-muted-foreground/70">
              dashed · similarity &nbsp; solid · causal &nbsp; gold · optimized-from
            </span>
          </div>
        </div>

        <aside className="rounded-lg border border-border bg-background/40 p-3 space-y-3">
          <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">selection</div>
          {selected ? (
            <>
              <div className="space-y-1">
                <div className="text-sm font-mono text-foreground/90">{selected.label}</div>
                <div className="text-[10px] uppercase tracking-[0.18em]" style={{ color: KIND_COLOR[selected.kind] }}>
                  {selected.kind}
                </div>
              </div>
              <div className="rounded-md border border-border bg-card/60 p-2">
                <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground mb-1">attrs</div>
                <pre className="text-[10px] font-mono text-foreground/80 whitespace-pre-wrap break-words">
{JSON.stringify(selected.attrs, null, 2)}
                </pre>
              </div>
              <div>
                <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground mb-1">k-nearest</div>
                <ul className="space-y-0.5 text-[10px] font-mono">
                  {neighbors.map((n) => (
                    <li key={n.node.id} className="flex justify-between cursor-pointer hover:text-primary"
                        onClick={() => setSelectedId(n.node.id)}>
                      <span>{n.node.label}</span>
                      <span className="text-muted-foreground">{n.sim.toFixed(2)}</span>
                    </li>
                  ))}
                </ul>
              </div>
              <div className="flex gap-2">
                <Button size="sm" variant="outline" onClick={() => setEnd("from")}
                        className="flex-1 text-[10px] uppercase tracking-[0.14em]">
                  set · from
                </Button>
                <Button size="sm" variant="outline" onClick={() => setEnd("to")}
                        className="flex-1 text-[10px] uppercase tracking-[0.14em]">
                  set · to
                </Button>
              </div>
              <Button size="sm" variant="outline" onClick={handleDelete}
                      className="w-full text-[10px] uppercase tracking-[0.14em] text-destructive">
                delete node
              </Button>
            </>
          ) : (
            <div className="text-[11px] text-muted-foreground">
              Click a node to inspect attributes, find nearest neighbors, or
              build a causal path between two nodes.
            </div>
          )}

          <div className="border-t border-border pt-2">
            <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-1">causal path</div>
            {path.length > 1 ? (
              <ol className="text-[10px] font-mono space-y-0.5">
                {path.map((id, i) => {
                  const n = g.nodes.find((x) => x.id === id);
                  return (
                    <li key={id} className="text-foreground/80">
                      {i + 1}. <span style={{ color: n ? KIND_COLOR[n.kind] : undefined }}>{n?.label ?? id}</span>
                    </li>
                  );
                })}
              </ol>
            ) : (
              <div className="text-[10px] text-muted-foreground">
                {pathEnds.from && pathEnds.to ? "no causal path" : "set a from + to node"}
              </div>
            )}
          </div>
        </aside>
      </div>

      {/* Kind histogram */}
      <div className="grid grid-cols-5 gap-2 text-[10px]">
        {(Object.keys(s.byKind) as NodeKind[]).map((k) => (
          <div key={k} className="rounded-md border border-border bg-card/60 px-2 py-1.5">
            <div className="uppercase tracking-[0.18em] text-muted-foreground">{k}</div>
            <div className="font-mono tabular-nums" style={{ color: KIND_COLOR[k] }}>{s.byKind[k]}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded-md border border-border bg-card/60 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      <div className="font-mono tabular-nums text-foreground/90">{value}</div>
    </div>
  );
}
