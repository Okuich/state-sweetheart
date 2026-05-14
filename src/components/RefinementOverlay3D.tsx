/**
 * Live 3D overlay for the adaptive refinement pass.
 *
 * Shows the *base* octree leaf cells colored by:
 *   - "mask"   – which cells were split / coarsened / unchanged this pass
 *   - "field"  – dominant physics field that drove the per-leaf error
 *   - "error"  – heatmap of the combined error indicator
 *
 * Pure Canvas2D painter's-algorithm projection, same shape as MeshViewer3D
 * so it stays performant on 30k+ leaves and avoids a three.js dep.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { OctreeMesh } from "@/lib/meshing/octree";
import type { AdaptivePassResult } from "@/lib/refinement";

type OverlayMode = "mask" | "field" | "error";

const FIELD_LABELS = ["stress", "thermal", "contact", "deform", "curve", "resid"];
const FIELD_COLORS = ["#ff5e7e", "#ffb347", "#7ad7ff", "#9b6bff", "#5cffb8", "#ffd34d"];

const CUBE_EDGES: [number, number][] = [
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

function hexToRgb(h: string): [number, number, number] {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function rgba(c: [number, number, number], a: number): string {
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

interface Props {
  baseMesh: OctreeMesh;
  result: AdaptivePassResult;
  height?: number;
}

export function RefinementOverlay3D({ baseMesh, result, height = 320 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [mode, setMode] = useState<OverlayMode>("mask");
  const [yaw, setYaw] = useState(0.6);
  const [pitch, setPitch] = useState(-0.45);
  const [zoom, setZoom] = useState(1);
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);

  const geo = useMemo(() => {
    const { bbox } = baseMesh;
    const ext = Math.max(
      bbox.max[0] - bbox.min[0],
      bbox.max[1] - bbox.min[1],
      bbox.max[2] - bbox.min[2],
    ) || 1;
    const cx = (bbox.min[0] + bbox.max[0]) / 2;
    const cy = (bbox.min[1] + bbox.max[1]) / 2;
    const cz = (bbox.min[2] + bbox.max[2]) / 2;
    const norm = (p: number, c: number) => (p - c) / ext;

    const L = baseMesh.leaves.length;
    const corners = new Float32Array(L * 8 * 3);
    const centers = new Float32Array(L * 3);
    for (let l = 0; l < L; l++) {
      const node = baseMesh.nodes[baseMesh.leaves[l]];
      const b = node.bbox;
      const xs = [b.min[0], b.max[0]];
      const ys = [b.min[1], b.max[1]];
      const zs = [b.min[2], b.max[2]];
      let i = 0;
      for (let zi = 0; zi < 2; zi++)
        for (let yi = 0; yi < 2; yi++)
          for (let xi = 0; xi < 2; xi++) {
            corners[(l * 8 + i) * 3 + 0] = norm(xs[xi], cx);
            corners[(l * 8 + i) * 3 + 1] = norm(ys[yi], cy);
            corners[(l * 8 + i) * 3 + 2] = norm(zs[zi], cz);
            i++;
          }
      centers[l * 3 + 0] = norm((b.min[0] + b.max[0]) / 2, cx);
      centers[l * 3 + 1] = norm((b.min[1] + b.max[1]) / 2, cy);
      centers[l * 3 + 2] = norm((b.min[2] + b.max[2]) / 2, cz);
    }
    return { corners, centers };
  }, [baseMesh]);

  const masks = useMemo(() => {
    const L = baseMesh.leaves.length;
    const refined = new Uint8Array(L);
    const coarsened = new Uint8Array(L);
    for (const li of result.pass.plan.splitLeaves) if (li < L) refined[li] = 1;
    for (const li of result.pass.plan.coarsenLeaves) if (li < L) coarsened[li] = 1;
    return { refined, coarsened };
  }, [baseMesh, result]);

  const errMax = useMemo(() => {
    let m = 0;
    for (let i = 0; i < result.error.combined.length; i++) {
      if (result.error.combined[i] > m) m = result.error.combined[i];
    }
    return m || 1;
  }, [result]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const focal = Math.min(cssW, cssH) * 0.9 * zoom;
    const camDist = 2.4;

    const project = (x: number, y: number, z: number): [number, number, number] => {
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      const y2 = y * cosP - z1 * sinP;
      const z2 = y * sinP + z1 * cosP;
      const zc = z2 + camDist;
      const s = focal / Math.max(0.1, zc);
      return [cssW / 2 + x1 * s, cssH / 2 - y2 * s, zc];
    };

    const { corners, centers } = geo;
    const L = corners.length / 24;
    const projCorners = new Float32Array(L * 8 * 2);
    const projCenters = new Float32Array(L);
    for (let l = 0; l < L; l++) {
      for (let c = 0; c < 8; c++) {
        const i = (l * 8 + c) * 3;
        const p = project(corners[i], corners[i + 1], corners[i + 2]);
        projCorners[(l * 8 + c) * 2 + 0] = p[0];
        projCorners[(l * 8 + c) * 2 + 1] = p[1];
      }
      const cz = project(centers[l * 3], centers[l * 3 + 1], centers[l * 3 + 2])[2];
      projCenters[l] = cz;
    }
    const order = Array.from({ length: L }, (_, i) => i).sort(
      (a, b) => projCenters[b] - projCenters[a],
    );

    const colorOf = (l: number): { stroke: [number, number, number]; alpha: number; fill: number } => {
      if (mode === "mask") {
        if (masks.refined[l]) return { stroke: hexToRgb("#34d399"), alpha: 0.95, fill: 0.22 };
        if (masks.coarsened[l]) return { stroke: hexToRgb("#fbbf24"), alpha: 0.85, fill: 0.16 };
        return { stroke: [70, 80, 100], alpha: 0.35, fill: 0.02 };
      }
      if (mode === "field") {
        const d = result.error.dominant[l] ?? 0;
        const c = hexToRgb(FIELD_COLORS[d % FIELD_COLORS.length]);
        const e = (result.error.combined[l] ?? 0) / errMax;
        return { stroke: c, alpha: 0.4 + 0.55 * e, fill: 0.05 + 0.18 * e };
      }
      // error heatmap — viridis-ish ramp
      const t = Math.min(1, (result.error.combined[l] ?? 0) / errMax);
      const r = Math.round(34 + (250 - 34) * t);
      const g = Math.round(80 + (180 - 80) * (1 - Math.abs(0.5 - t) * 2));
      const b = Math.round(150 + (60 - 150) * t);
      return { stroke: [r, g, b], alpha: 0.3 + 0.65 * t, fill: 0.03 + 0.2 * t };
    };

    ctx.lineWidth = 0.6;
    for (const l of order) {
      const { stroke, alpha, fill } = colorOf(l);
      const z = projCenters[l];
      const fade = Math.max(0.2, Math.min(1, 1.6 / z));
      ctx.strokeStyle = rgba(stroke, alpha * fade);
      ctx.fillStyle = rgba(stroke, fill * fade);
      // Simple front-face fill via min/max rect of 4 front corners (cheap glow)
      if (fill > 0.05) {
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (let c = 0; c < 8; c++) {
          const x = projCorners[(l * 8 + c) * 2 + 0];
          const y = projCorners[(l * 8 + c) * 2 + 1];
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
        ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
      }
      ctx.beginPath();
      for (const [a, b] of CUBE_EDGES) {
        const ax = projCorners[(l * 8 + a) * 2 + 0];
        const ay = projCorners[(l * 8 + a) * 2 + 1];
        const bx = projCorners[(l * 8 + b) * 2 + 0];
        const by = projCorners[(l * 8 + b) * 2 + 1];
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }

    // Axis gizmo
    const ax0 = 28, ay0 = cssH - 24, axLen = 18;
    const axes: [string, [number, number, number]][] = [
      ["x", [1, 0, 0]], ["y", [0, 1, 0]], ["z", [0, 0, 1]],
    ];
    const axCols = ["#f87171", "#4ade80", "#60a5fa"];
    axes.forEach(([label, v], i) => {
      const x1 = v[0] * cosY + v[2] * sinY;
      const z1 = -v[0] * sinY + v[2] * cosY;
      const y2 = v[1] * cosP - z1 * sinP;
      ctx.strokeStyle = axCols[i];
      ctx.fillStyle = axCols[i];
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.moveTo(ax0, ay0);
      ctx.lineTo(ax0 + x1 * axLen, ay0 - y2 * axLen);
      ctx.stroke();
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillText(label, ax0 + x1 * axLen + 2, ay0 - y2 * axLen + 3);
    });
  }, [geo, masks, errMax, mode, yaw, pitch, zoom, result]);

  const onPointerDown = (e: React.PointerEvent) => {
    (e.target as Element).setPointerCapture(e.pointerId);
    drag.current = { x: e.clientX, y: e.clientY, yaw, pitch };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    setYaw(drag.current.yaw + dx * 0.008);
    setPitch(Math.max(-1.4, Math.min(1.4, drag.current.pitch - dy * 0.008)));
  };
  const onPointerUp = () => { drag.current = null; };
  const onWheel = (e: React.WheelEvent) => {
    setZoom((z) => Math.max(0.4, Math.min(4, z * (e.deltaY > 0 ? 0.92 : 1.08))));
  };

  const refinedCount = result.pass.plan.splitLeaves.length;
  const coarsenedCount = result.pass.plan.coarsenLeaves.length;
  const unchangedCount = baseMesh.leaves.length - refinedCount - coarsenedCount;

  return (
    <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          refinement overlay · drag to orbit · scroll to zoom
        </div>
        <div className="flex gap-1 text-[10px]">
          {(["mask", "field", "error"] as OverlayMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-2 py-1 rounded uppercase tracking-[0.18em] border ${
                mode === m
                  ? "border-primary bg-primary/15 text-foreground"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onWheel={onWheel}
        style={{ width: "100%", height, touchAction: "none", cursor: drag.current ? "grabbing" : "grab" }}
        className="block rounded border border-border/50 bg-[radial-gradient(circle_at_50%_40%,rgba(255,255,255,0.04),transparent_70%)]"
      />
      <div className="flex items-center justify-between text-[10px] font-mono text-muted-foreground gap-3 flex-wrap">
        {mode === "mask" ? (
          <div className="flex gap-3">
            <span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rounded-sm bg-emerald-400" /> refined · <span className="text-foreground tabular-nums">{refinedCount}</span></span>
            <span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rounded-sm bg-amber-400" /> coarsened · <span className="text-foreground tabular-nums">{coarsenedCount}</span></span>
            <span className="flex items-center gap-1.5"><span className="inline-block w-2 h-2 rounded-sm bg-muted-foreground/50" /> stable · <span className="text-foreground tabular-nums">{unchangedCount}</span></span>
          </div>
        ) : mode === "field" ? (
          <div className="flex gap-2 flex-wrap">
            {FIELD_LABELS.map((l, i) => (
              <span key={l} className="flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-sm" style={{ background: FIELD_COLORS[i] }} />
                {l}
              </span>
            ))}
          </div>
        ) : (
          <span>combined error · 0 → <span className="text-foreground tabular-nums">{errMax.toFixed(3)}</span></span>
        )}
        <span>{baseMesh.leaves.length.toLocaleString()} base leaves</span>
      </div>
    </div>
  );
}
