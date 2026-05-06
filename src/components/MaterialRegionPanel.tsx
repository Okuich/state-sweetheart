import { useCallback, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  buildCubeMesh, DEFAULT_PALETTE, histogram,
  paintByBoxRegion, paintBySphereRegion, restore, snapshot, tetCentroid,
  type TetMesh,
} from "@/lib/matIdAssignment";

type BrushShape = "sphere" | "box";

export function MaterialRegionPanel() {
  // Mesh resolution
  const [res, setRes] = useState(6);
  const meshRef = useRef<TetMesh>(buildCubeMesh(6, 6, 6));
  const [, force] = useState(0);
  const repaint = () => force((n) => n + 1);

  const [activeMat, setActiveMat] = useState(1);
  const [brush, setBrush] = useState<BrushShape>("sphere");
  const [brushRadius, setBrushRadius] = useState(0.18);
  const [brushPos, setBrushPos] = useState<[number, number, number]>([0.5, 0.5, 0.5]);
  const undoRef = useRef<Uint32Array[]>([]);

  const rebuild = (n: number) => {
    setRes(n);
    meshRef.current = buildCubeMesh(n, n, n);
    undoRef.current = [];
    repaint();
  };

  const apply = useCallback(() => {
    const m = meshRef.current;
    undoRef.current.push(snapshot(m));
    if (undoRef.current.length > 30) undoRef.current.shift();
    if (brush === "sphere") {
      paintBySphereRegion(m, activeMat, brushPos, brushRadius);
    } else {
      const r = brushRadius;
      paintByBoxRegion(m, activeMat,
        [brushPos[0] - r, brushPos[1] - r, brushPos[2] - r],
        [brushPos[0] + r, brushPos[1] + r, brushPos[2] + r]);
    }
    repaint();
  }, [activeMat, brush, brushPos, brushRadius]);

  const undo = () => {
    const snap = undoRef.current.pop();
    if (snap) { restore(meshRef.current, snap); repaint(); }
  };

  const clearAll = () => {
    undoRef.current.push(snapshot(meshRef.current));
    meshRef.current.matId.fill(0);
    repaint();
  };

  const hist = useMemo(
    () => histogram(meshRef.current, DEFAULT_PALETTE),
    // depend on force counter via res/brush-apply triggering rerenders
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [meshRef.current.matId, res],
  );

  // ── Visualization: orthographic projection of every tet's centroid ──
  // Color = palette[matId]. Sized by 1/res so they form a recognisable solid.
  const view = useMemo(() => {
    const W = 360, H = 360, pad = 8;
    const m = meshRef.current;
    type Dot = { x: number; y: number; depth: number; color: string; r: number };
    const dots: Dot[] = [];
    // Isometric projection
    const a = Math.PI / 6;
    const cosA = Math.cos(a), sinA = Math.sin(a);
    const proj = (x: number, y: number, z: number) => ({
      px: (x - z) * cosA,
      py: (x + z) * sinA - y,
      depth: x + z - y,
    });
    const dotR = Math.max(2, (W - 2 * pad) / (res * 2.6));
    for (let i = 0; i < m.tets.length; i++) {
      const c = tetCentroid(m, i);
      const { px, py, depth } = proj(c[0], c[1], c[2]);
      const color = DEFAULT_PALETTE[m.matId[i]]?.color ?? "#444";
      dots.push({ x: px, y: py, depth, color, r: dotR });
    }
    // Project brush sphere centroid for cursor.
    const brushP = proj(...brushPos);
    // Auto-fit
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const d of dots) {
      if (d.x < minX) minX = d.x; if (d.x > maxX) maxX = d.x;
      if (d.y < minY) minY = d.y; if (d.y > maxY) maxY = d.y;
    }
    const w = maxX - minX || 1, h = maxY - minY || 1;
    const s = Math.min((W - 2 * pad) / w, (H - 2 * pad) / h);
    const tx = (px: number) => pad + (px - minX) * s;
    const ty = (py: number) => pad + (py - minY) * s;
    dots.sort((a, b) => a.depth - b.depth);
    return {
      W, H,
      dots: dots.map((d) => ({ ...d, x: tx(d.x), y: ty(d.y) })),
      brush: { x: tx(brushP.px), y: ty(brushP.py), r: brushRadius * s * 1.4 },
    };
  }, [res, brushPos, brushRadius, hist]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Material regions (per-element MatID)
        </h3>
        <div className="flex gap-1">
          <Button variant="ghost" size="sm" onClick={undo} className="h-7 px-2 text-xs">
            undo
          </Button>
          <Button variant="ghost" size="sm" onClick={clearAll} className="h-7 px-2 text-xs">
            clear
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[auto_1fr]">
        {/* Visualization */}
        <div className="rounded border border-border/60 bg-background/40 p-2">
          <svg viewBox={`0 0 ${view.W} ${view.H}`} className="aspect-square w-full max-w-[360px]">
            {view.dots.map((d, i) => (
              <circle key={i} cx={d.x} cy={d.y} r={d.r} fill={d.color}
                      stroke="hsl(var(--background))" strokeWidth={0.4} />
            ))}
            <circle cx={view.brush.x} cy={view.brush.y} r={view.brush.r}
                    fill="none" stroke="hsl(var(--primary))"
                    strokeWidth={1.2} strokeDasharray="3 3" />
          </svg>
          <div className="mt-1 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            {meshRef.current.tets.length} tets · isometric view
          </div>
        </div>

        {/* Controls */}
        <div className="space-y-3">
          {/* Mesh resolution */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">mesh resolution</span>
              <span className="tabular-nums">{res}³</span>
            </div>
            <Slider value={[res]} min={2} max={10} step={1}
                    onValueChange={(v) => rebuild(v[0])} />
          </div>

          {/* Active material */}
          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">active material</div>
            <Select value={String(activeMat)} onValueChange={(v) => setActiveMat(Number(v))}>
              <SelectTrigger className="h-9 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {DEFAULT_PALETTE.map((p) => (
                  <SelectItem key={p.id} value={String(p.id)} className="text-xs">
                    <span className="inline-block h-3 w-3 align-middle rounded-sm mr-2"
                          style={{ background: p.color }} />
                    {p.name} <span className="text-muted-foreground ml-1">({p.kind})</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Brush shape */}
          <div className="flex gap-1">
            {(["sphere", "box"] as BrushShape[]).map((b) => (
              <Button key={b} variant={brush === b ? "default" : "outline"}
                      size="sm" onClick={() => setBrush(b)}
                      className="h-7 px-3 text-xs flex-1">
                {b}
              </Button>
            ))}
          </div>

          {/* Brush radius */}
          <div className="space-y-1">
            <div className="flex justify-between text-[11px]">
              <span className="text-muted-foreground">brush radius</span>
              <span className="tabular-nums">{brushRadius.toFixed(2)}</span>
            </div>
            <Slider value={[brushRadius]} min={0.05} max={0.6} step={0.01}
                    onValueChange={(v) => setBrushRadius(v[0])} />
          </div>

          {/* Brush position */}
          {(["x", "y", "z"] as const).map((axis, i) => (
            <div key={axis} className="space-y-1">
              <div className="flex justify-between text-[11px]">
                <span className="text-muted-foreground">brush {axis}</span>
                <span className="tabular-nums">{brushPos[i].toFixed(2)}</span>
              </div>
              <Slider value={[brushPos[i]]} min={0} max={1} step={0.02}
                      onValueChange={(v) => {
                        const nb: [number, number, number] = [...brushPos];
                        nb[i] = v[0];
                        setBrushPos(nb);
                      }} />
            </div>
          ))}

          <Button onClick={apply} className="w-full h-9 text-xs">paint with brush</Button>
        </div>
      </div>

      {/* Legend / histogram */}
      <div className="rounded border border-border/60 bg-background/40 p-2">
        <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          assignment histogram
        </div>
        <div className="space-y-1.5">
          {DEFAULT_PALETTE.map((p) => {
            const count = hist.get(p.id) ?? 0;
            const pct = (count / Math.max(1, meshRef.current.tets.length)) * 100;
            return (
              <div key={p.id} className="space-y-0.5">
                <div className="flex justify-between text-[11px]">
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block h-3 w-3 rounded-sm"
                          style={{ background: p.color }} />
                    {p.name}
                  </span>
                  <span className="tabular-nums text-muted-foreground">
                    {count} · {pct.toFixed(1)}%
                  </span>
                </div>
                <div className="h-1 w-full overflow-hidden rounded bg-border/60">
                  <div className="h-full" style={{ width: `${pct}%`, background: p.color }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
