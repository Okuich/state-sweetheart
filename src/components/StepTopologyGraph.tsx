/**
 * StepTopologyGraph
 *
 * Interactive SVG viewer for the topology graph emitted by buildTopology().
 * Features:
 *   - Deterministic force-directed layout (seeded by entity ids) — same input
 *     produces the same picture, important for review/screenshots.
 *   - Node coloring: root (no in-edges), orphan (referenced but missing),
 *     leaf (no out-edges), and per-type tinting via a palette hash.
 *   - Hover + click selection: highlights neighbors (in & out) and dims the
 *     rest. Click on the SVG background to clear.
 *   - Pan & zoom (wheel + drag), entity-type filter chips, and a side panel
 *     showing the selected entity plus its incoming/outgoing references.
 *
 * Layout is intentionally simple (Fruchterman–Reingold-ish) and capped at
 * 600 nodes — STEP files larger than that get a sampled view to keep the
 * UI responsive.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { ParseReport, TopoGraph, EntityRef } from "@/lib/stepParser";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const MAX_NODES = 600;

interface Pos { x: number; y: number; }

interface LayoutNode {
  id: EntityRef;
  type: string;
  pos: Pos;
  isRoot: boolean;
  isOrphan: boolean;
  isLeaf: boolean;
}

function hashColor(type: string): string {
  let h = 0;
  for (let i = 0; i < type.length; i++) h = (h * 31 + type.charCodeAt(i)) >>> 0;
  const hue = h % 360;
  return `hsl(${hue} 70% 60%)`;
}

/** Tiny deterministic PRNG so initial positions are reproducible per file. */
function mulberry32(seed: number) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

interface LayoutResult {
  nodes: Map<EntityRef, LayoutNode>;
  edges: { from: EntityRef; to: EntityRef; orphan: boolean }[];
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
  sampled: boolean;
  total: number;
}

function computeLayout(
  report: ParseReport,
  topo: TopoGraph,
  typeFilter: Set<string> | null,
): LayoutResult {
  const allIds = [...topo.nodes.keys()];
  const filtered = typeFilter
    ? allIds.filter((id) => typeFilter.has(topo.nodes.get(id)!.type))
    : allIds;

  const sampled = filtered.length > MAX_NODES;
  const idsArr = sampled ? filtered.slice(0, MAX_NODES) : filtered;
  const ids = new Set(idsArr);

  const rootSet = new Set(topo.roots);
  const orphanSet = new Set(topo.orphans);

  // Always include orphans referenced by visible nodes (so they show up).
  const includedOrphans = new Set<EntityRef>();
  for (const id of ids) {
    for (const r of topo.outEdges.get(id) ?? []) {
      if (orphanSet.has(r)) includedOrphans.add(r);
    }
  }

  const W = 800, H = 500;
  const rand = mulberry32(idsArr.length + 1);
  const nodes = new Map<EntityRef, LayoutNode>();

  for (const id of idsArr) {
    const n = topo.nodes.get(id)!;
    const outs = topo.outEdges.get(id)?.length ?? 0;
    nodes.set(id, {
      id, type: n.type,
      pos: { x: rand() * W, y: rand() * H },
      isRoot: rootSet.has(id),
      isOrphan: false,
      isLeaf: outs === 0,
    });
  }
  for (const id of includedOrphans) {
    nodes.set(id, {
      id, type: "MISSING",
      pos: { x: rand() * W, y: rand() * H },
      isRoot: false, isOrphan: true, isLeaf: true,
    });
  }

  const edges: LayoutResult["edges"] = [];
  for (const id of idsArr) {
    for (const to of topo.outEdges.get(id) ?? []) {
      if (nodes.has(to)) {
        edges.push({ from: id, to, orphan: orphanSet.has(to) });
      }
    }
  }

  // Force-directed iterations (cheap, deterministic).
  const list = [...nodes.values()];
  const k = Math.sqrt((W * H) / Math.max(1, list.length)) * 0.8;
  const iterations = list.length > 200 ? 60 : 120;
  let temp = W / 8;

  for (let it = 0; it < iterations; it++) {
    const disp = new Map<EntityRef, Pos>();
    for (const n of list) disp.set(n.id, { x: 0, y: 0 });

    // Repulsion (sampled neighbors for big graphs)
    const sampleStep = list.length > 250 ? 3 : 1;
    for (let i = 0; i < list.length; i++) {
      const a = list[i];
      for (let j = i + 1; j < list.length; j += sampleStep) {
        const b = list[j];
        const dx = a.pos.x - b.pos.x;
        const dy = a.pos.y - b.pos.y;
        const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
        const f = (k * k) / dist;
        const ux = (dx / dist) * f;
        const uy = (dy / dist) * f;
        const da = disp.get(a.id)!; da.x += ux; da.y += uy;
        const db = disp.get(b.id)!; db.x -= ux; db.y -= uy;
      }
    }
    // Attraction
    for (const e of edges) {
      const a = nodes.get(e.from)!;
      const b = nodes.get(e.to)!;
      const dx = a.pos.x - b.pos.x;
      const dy = a.pos.y - b.pos.y;
      const dist = Math.sqrt(dx * dx + dy * dy) + 0.01;
      const f = (dist * dist) / k;
      const ux = (dx / dist) * f;
      const uy = (dy / dist) * f;
      const da = disp.get(a.id)!; da.x -= ux; da.y -= uy;
      const db = disp.get(b.id)!; db.x += ux; db.y += uy;
    }
    // Apply with temperature & frame bounds
    for (const n of list) {
      const d = disp.get(n.id)!;
      const m = Math.sqrt(d.x * d.x + d.y * d.y) + 0.01;
      n.pos.x += (d.x / m) * Math.min(m, temp);
      n.pos.y += (d.y / m) * Math.min(m, temp);
      n.pos.x = Math.max(10, Math.min(W - 10, n.pos.x));
      n.pos.y = Math.max(10, Math.min(H - 10, n.pos.y));
    }
    temp *= 0.95;
  }

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const n of list) {
    if (n.pos.x < minX) minX = n.pos.x;
    if (n.pos.y < minY) minY = n.pos.y;
    if (n.pos.x > maxX) maxX = n.pos.x;
    if (n.pos.y > maxY) maxY = n.pos.y;
  }
  // Reference report so the unused-arg lint doesn't complain when typeFilter is null.
  void report;
  return {
    nodes, edges,
    bounds: { minX, minY, maxX, maxY },
    sampled, total: filtered.length,
  };
}

export function StepTopologyGraph({
  report, topo,
}: { report: ParseReport; topo: TopoGraph }) {
  const [typeFilter, setTypeFilter] = useState<Set<string> | null>(null);
  const [hoverId, setHoverId] = useState<EntityRef | null>(null);
  const [selectedId, setSelectedId] = useState<EntityRef | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, zoom: 1 });
  const dragRef = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);

  const layout = useMemo(
    () => computeLayout(report, topo, typeFilter),
    [report, topo, typeFilter],
  );

  // Reset selection when filter changes and selected entity is no longer present.
  useEffect(() => {
    if (selectedId !== null && !layout.nodes.has(selectedId)) setSelectedId(null);
  }, [layout, selectedId]);

  const focusId = hoverId ?? selectedId;
  const { neighborsIn, neighborsOut } = useMemo(() => {
    const ins = new Set<EntityRef>();
    const outs = new Set<EntityRef>();
    if (focusId !== null) {
      for (const e of layout.edges) {
        if (e.to === focusId) ins.add(e.from);
        if (e.from === focusId) outs.add(e.to);
      }
    }
    return { neighborsIn: ins, neighborsOut: outs };
  }, [focusId, layout]);

  const topTypes = useMemo(() => {
    return [...report.byType.entries()]
      .map(([t, ids]) => [t, ids.length] as const)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12);
  }, [report]);

  const selectedEntity = selectedId !== null ? report.entities.get(selectedId) : null;

  // ── Pan / zoom ──
  const onWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.87;
    setView((v) => ({ ...v, zoom: Math.max(0.3, Math.min(4, v.zoom * factor)) }));
  };
  const onMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if ((e.target as SVGElement).tagName === "circle") return;
    dragRef.current = { x: e.clientX, y: e.clientY, vx: view.x, vy: view.y };
  };
  const onMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const d = dragRef.current;
    if (!d) return;
    setView((v) => ({ ...v, x: d.vx + (e.clientX - d.x), y: d.vy + (e.clientY - d.y) }));
  };
  const endDrag = () => { dragRef.current = null; };

  const W = 800, H = 500;

  return (
    <div className="rounded-md border border-border bg-background/30 p-3 space-y-3">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            topology
          </span>
          <Badge variant="outline" className="text-[10px]">
            {layout.nodes.size} nodes
          </Badge>
          <Badge variant="outline" className="text-[10px]">
            {layout.edges.length} edges
          </Badge>
          <Badge variant="outline" className="text-[10px] border-primary/60 text-primary">
            {topo.roots.length} roots
          </Badge>
          {topo.orphans.length > 0 && (
            <Badge variant="outline" className="text-[10px] border-destructive/60 text-destructive">
              {topo.orphans.length} orphans
            </Badge>
          )}
          {layout.sampled && (
            <Badge variant="outline" className="text-[10px] border-accent/60 text-accent">
              sampled {MAX_NODES}/{layout.total}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button size="sm" variant="outline" className="h-6 text-[10px]"
            onClick={() => setView({ x: 0, y: 0, zoom: 1 })}>
            reset view
          </Button>
          <Button size="sm" variant="outline" className="h-6 text-[10px]"
            onClick={() => { setSelectedId(null); setTypeFilter(null); }}>
            clear filter
          </Button>
        </div>
      </div>

      {/* Type filter chips */}
      <div className="flex flex-wrap gap-1">
        {topTypes.map(([t, n]) => {
          const active = typeFilter?.has(t) ?? false;
          return (
            <button
              key={t}
              onClick={() => setTypeFilter((cur) => {
                const next = new Set(cur ?? []);
                if (next.has(t)) next.delete(t); else next.add(t);
                return next.size === 0 ? null : next;
              })}
              className={`text-[10px] font-mono px-1.5 py-0.5 rounded border transition ${
                active
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border bg-background/30 text-muted-foreground hover:text-foreground"
              }`}
              style={!active ? { borderLeftColor: hashColor(t), borderLeftWidth: 3 } : undefined}
            >
              {t} <span className="text-muted-foreground">·{n}</span>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-[1fr_240px] gap-3">
        {/* SVG viewport */}
        <div className="rounded border border-border/60 bg-background/50 overflow-hidden relative">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full h-[500px] cursor-grab active:cursor-grabbing select-none"
            onWheel={onWheel}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={endDrag}
            onMouseLeave={endDrag}
            onClick={(e) => {
              if ((e.target as SVGElement).tagName === "svg") setSelectedId(null);
            }}
          >
            <defs>
              <marker id="arrow-default" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="hsl(var(--muted-foreground))" opacity="0.5" />
              </marker>
              <marker id="arrow-hot" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="hsl(var(--primary))" />
              </marker>
              <marker id="arrow-orphan" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                <path d="M 0 0 L 10 5 L 0 10 z" fill="hsl(var(--destructive))" />
              </marker>
            </defs>

            <g transform={`translate(${view.x},${view.y}) scale(${view.zoom})`}>
              {/* Edges */}
              {layout.edges.map((e, i) => {
                const a = layout.nodes.get(e.from)!;
                const b = layout.nodes.get(e.to)!;
                const isFocus =
                  focusId !== null && (e.from === focusId || e.to === focusId);
                const dim = focusId !== null && !isFocus;
                const stroke = e.orphan
                  ? "hsl(var(--destructive))"
                  : isFocus
                    ? "hsl(var(--primary))"
                    : "hsl(var(--muted-foreground))";
                const marker = e.orphan
                  ? "url(#arrow-orphan)"
                  : isFocus
                    ? "url(#arrow-hot)"
                    : "url(#arrow-default)";
                return (
                  <line key={i}
                    x1={a.pos.x} y1={a.pos.y} x2={b.pos.x} y2={b.pos.y}
                    stroke={stroke}
                    strokeOpacity={dim ? 0.08 : e.orphan ? 0.85 : isFocus ? 0.95 : 0.35}
                    strokeWidth={isFocus ? 1.4 : 0.6}
                    markerEnd={marker}
                  />
                );
              })}

              {/* Nodes */}
              {[...layout.nodes.values()].map((n) => {
                const isFocus = n.id === focusId;
                const inNeighbor = neighborsIn.has(n.id);
                const outNeighbor = neighborsOut.has(n.id);
                const isNeighbor = inNeighbor || outNeighbor;
                const dim = focusId !== null && !isFocus && !isNeighbor;
                const fill = n.isOrphan
                  ? "hsl(var(--destructive))"
                  : n.isRoot
                    ? "hsl(var(--primary))"
                    : hashColor(n.type);
                const r = n.isOrphan ? 5 : n.isRoot ? 6 : isFocus ? 7 : 4;
                const stroke = isFocus
                  ? "hsl(var(--foreground))"
                  : isNeighbor
                    ? "hsl(var(--primary))"
                    : "hsl(var(--background))";
                return (
                  <circle key={n.id}
                    cx={n.pos.x} cy={n.pos.y} r={r}
                    fill={fill}
                    fillOpacity={dim ? 0.15 : 1}
                    stroke={stroke}
                    strokeWidth={isFocus ? 2 : isNeighbor ? 1.5 : 0.5}
                    style={{ cursor: "pointer" }}
                    onMouseEnter={() => setHoverId(n.id)}
                    onMouseLeave={() => setHoverId(null)}
                    onClick={(e) => { e.stopPropagation(); setSelectedId(n.id); }}
                  >
                    <title>{`#${n.id} · ${n.type}${n.isRoot ? " (root)" : ""}${n.isOrphan ? " (orphan ref)" : ""}`}</title>
                  </circle>
                );
              })}
            </g>
          </svg>

          {/* Legend */}
          <div className="absolute bottom-2 left-2 flex flex-wrap gap-2 text-[9px] font-mono bg-background/80 backdrop-blur px-2 py-1 rounded border border-border/60">
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-primary" /> root
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full bg-destructive" /> orphan
            </span>
            <span className="flex items-center gap-1">
              <span className="inline-block h-2 w-2 rounded-full"
                style={{ background: "hsl(var(--muted-foreground))" }} /> entity
            </span>
            <span className="text-muted-foreground">scroll · zoom</span>
            <span className="text-muted-foreground">drag · pan</span>
          </div>
        </div>

        {/* Inspector */}
        <div className="rounded border border-border/60 bg-background/50 p-2 text-[10px] font-mono">
          {selectedEntity ? (
            <>
              <div className="flex items-center justify-between mb-1">
                <span className="text-foreground">#{selectedEntity.id}</span>
                <span className="text-primary uppercase tracking-[0.18em] text-[9px]">
                  {selectedEntity.type}
                </span>
              </div>
              <pre className="text-[9px] text-foreground/75 whitespace-pre-wrap break-all max-h-24 overflow-auto mb-2">
                {selectedEntity.raw}
              </pre>
              <div className="text-muted-foreground uppercase tracking-[0.16em] text-[9px] mb-0.5">
                out · {neighborsOut.size}
              </div>
              <div className="flex flex-wrap gap-1 mb-2 max-h-16 overflow-auto">
                {[...neighborsOut].slice(0, 50).map((id) => (
                  <button key={id}
                    onClick={() => setSelectedId(id)}
                    className="px-1 rounded bg-muted/40 hover:bg-primary/20">
                    #{id}
                  </button>
                ))}
                {neighborsOut.size === 0 && (
                  <span className="text-muted-foreground/60">leaf</span>
                )}
              </div>
              <div className="text-muted-foreground uppercase tracking-[0.16em] text-[9px] mb-0.5">
                in · {neighborsIn.size}
              </div>
              <div className="flex flex-wrap gap-1 max-h-16 overflow-auto">
                {[...neighborsIn].slice(0, 50).map((id) => (
                  <button key={id}
                    onClick={() => setSelectedId(id)}
                    className="px-1 rounded bg-muted/40 hover:bg-primary/20">
                    #{id}
                  </button>
                ))}
                {neighborsIn.size === 0 && (
                  <span className="text-primary">root</span>
                )}
              </div>
            </>
          ) : selectedId !== null ? (
            <div className="text-destructive">#{selectedId} · orphan reference (no entity body)</div>
          ) : (
            <div className="text-muted-foreground/70">
              hover or click a node to inspect entity, references, and dependents.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
