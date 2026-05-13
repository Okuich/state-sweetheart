/**
 * Lightweight Canvas2D 3D preview for the generated mesh.
 *
 * Renders octree leaf cells (cube wireframes) and optionally tetrahedral
 * edges, colored by partition / refinement depth / boundary flag. Pure
 * software perspective projection + painter's algorithm — no three.js
 * dependency, fast enough for ~30k leaves.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { MeshingResult } from "@/lib/meshing";

type ColorMode = "partition" | "depth" | "boundary";

interface Props {
  result: MeshingResult;
  height?: number;
}

const PALETTE = [
  "#22d3ee", "#a78bfa", "#f472b6", "#fb923c",
  "#34d399", "#facc15", "#60a5fa", "#f87171",
  "#c084fc", "#4ade80", "#fbbf24", "#38bdf8",
  "#fb7185", "#2dd4bf", "#e879f9", "#84cc16",
];

function hexToRgb(h: string): [number, number, number] {
  const v = parseInt(h.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function rgba(c: [number, number, number], a: number): string {
  return `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

const CUBE_EDGES: [number, number][] = [
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

export function MeshViewer3D({ result, height = 320 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [colorMode, setColorMode] = useState<ColorMode>("partition");
  const [showTets, setShowTets] = useState(false);
  const [showCells, setShowCells] = useState(true);
  const [yaw, setYaw] = useState(0.6);
  const [pitch, setPitch] = useState(-0.45);
  const [zoom, setZoom] = useState(1);
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);

  const { mesh, partition } = result;

  // Pre-compute leaf cube corners + per-leaf metadata in normalized [-0.5,0.5]
  const leafGeo = useMemo(() => {
    const { bbox } = mesh;
    const ext = Math.max(
      bbox.max[0] - bbox.min[0],
      bbox.max[1] - bbox.min[1],
      bbox.max[2] - bbox.min[2],
    ) || 1;
    const cx = (bbox.min[0] + bbox.max[0]) / 2;
    const cy = (bbox.min[1] + bbox.max[1]) / 2;
    const cz = (bbox.min[2] + bbox.max[2]) / 2;
    const norm = (p: number, c: number) => (p - c) / ext;

    const L = mesh.leaves.length;
    // 8 corners × 3 coords per leaf.
    const corners = new Float32Array(L * 8 * 3);
    const centers = new Float32Array(L * 3);
    const depths = new Uint8Array(L);
    const boundary = mesh.boundaryLeaf;

    // partition per leaf (majority of its 6 tets — but each leaf's tets all
    // start in the same partition under our partitioner most of the time;
    // we just take the first tet's partition for speed).
    const leafPart = new Uint16Array(L);
    for (let l = 0; l < L; l++) leafPart[l] = partition.tetPart[l * 6] ?? 0;

    for (let l = 0; l < L; l++) {
      const node = mesh.nodes[mesh.leaves[l]];
      const b = node.bbox;
      depths[l] = node.depth;
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

    let maxDepth = 0;
    for (const d of depths) if (d > maxDepth) maxDepth = d;

    return { corners, centers, depths, boundary, leafPart, maxDepth };
  }, [mesh, partition]);

  // Tet edges (deduplicated) in normalized space — only when showTets.
  const tetEdges = useMemo(() => {
    if (!showTets) return null;
    const { bbox, vertices, tets } = mesh;
    const ext = Math.max(
      bbox.max[0] - bbox.min[0],
      bbox.max[1] - bbox.min[1],
      bbox.max[2] - bbox.min[2],
    ) || 1;
    const cx = (bbox.min[0] + bbox.max[0]) / 2;
    const cy = (bbox.min[1] + bbox.max[1]) / 2;
    const cz = (bbox.min[2] + bbox.max[2]) / 2;

    const V = vertices.length / 3;
    const nv = new Float32Array(V * 3);
    for (let i = 0; i < V; i++) {
      nv[i * 3 + 0] = (vertices[i * 3 + 0] - cx) / ext;
      nv[i * 3 + 1] = (vertices[i * 3 + 1] - cy) / ext;
      nv[i * 3 + 2] = (vertices[i * 3 + 2] - cz) / ext;
    }

    const T = tets.length / 4;
    const seen = new Set<number>();
    const edges: number[] = [];
    const TET_E: [number, number][] = [
      [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
    ];
    // Cap to avoid pathological draws.
    const cap = Math.min(T, 12000);
    const stride = Math.max(1, Math.floor(T / cap));
    for (let t = 0; t < T; t += stride) {
      for (const [a, b] of TET_E) {
        const va = tets[t * 4 + a];
        const vb = tets[t * 4 + b];
        const lo = va < vb ? va : vb;
        const hi = va < vb ? vb : va;
        const key = lo * V + hi;
        if (seen.has(key)) continue;
        seen.add(key);
        edges.push(va, vb);
      }
    }
    return { nv, edges: new Uint32Array(edges) };
  }, [mesh, showTets]);

  // Render
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

    // Background subtle grid bg
    ctx.fillStyle = "rgba(255,255,255,0.015)";
    ctx.fillRect(0, 0, cssW, cssH);

    const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
    const cosP = Math.cos(pitch), sinP = Math.sin(pitch);
    const focal = Math.min(cssW, cssH) * 0.9 * zoom;
    const camDist = 2.4;

    function project(x: number, y: number, z: number): [number, number, number] {
      // Yaw around Y, pitch around X.
      const x1 = x * cosY + z * sinY;
      const z1 = -x * sinY + z * cosY;
      const y2 = y * cosP - z1 * sinP;
      const z2 = y * sinP + z1 * cosP;
      const zc = z2 + camDist;
      const s = focal / Math.max(0.1, zc);
      return [cssW / 2 + x1 * s, cssH / 2 - y2 * s, zc];
    }

    const { corners, centers, depths, boundary, leafPart, maxDepth } = leafGeo;
    const L = corners.length / 24;

    function leafColor(l: number): [number, number, number] {
      if (colorMode === "partition") {
        return hexToRgb(PALETTE[leafPart[l] % PALETTE.length]);
      }
      if (colorMode === "depth") {
        const t = maxDepth > 0 ? depths[l] / maxDepth : 0;
        // cyan → magenta gradient
        return [
          Math.round(34 + (244 - 34) * t),
          Math.round(211 + (114 - 211) * t),
          Math.round(238 + (182 - 238) * t),
        ];
      }
      // boundary
      return boundary[l]
        ? hexToRgb("#fb7185")
        : [80, 90, 110];
    }

    // Project leaf centers for depth sort + corners.
    const projCorners = new Float32Array(L * 8 * 2);
    const projCenters = new Float32Array(L);
    for (let l = 0; l < L; l++) {
      let zSum = 0;
      for (let c = 0; c < 8; c++) {
        const i = (l * 8 + c) * 3;
        const p = project(corners[i], corners[i + 1], corners[i + 2]);
        projCorners[(l * 8 + c) * 2 + 0] = p[0];
        projCorners[(l * 8 + c) * 2 + 1] = p[1];
        zSum += p[2];
      }
      // Use center z for sort.
      const cz = project(centers[l * 3], centers[l * 3 + 1], centers[l * 3 + 2])[2];
      projCenters[l] = cz === cz ? cz : zSum / 8;
    }

    const order = new Uint32Array(L);
    for (let i = 0; i < L; i++) order[i] = i;
    // Painter's: far first.
    const arr = Array.from(order);
    arr.sort((a, b) => projCenters[b] - projCenters[a]);

    if (showCells) {
      ctx.lineWidth = 0.6;
      for (const l of arr) {
        const col = leafColor(l);
        // fade by depth so back cells recede
        const z = projCenters[l];
        const fade = Math.max(0.15, Math.min(1, 1.6 / z));
        ctx.strokeStyle = rgba(col, 0.55 * fade);
        ctx.fillStyle = rgba(col, 0.06 * fade);
        ctx.beginPath();
        // Draw the 12 edges
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
    }

    if (tetEdges) {
      ctx.strokeStyle = "rgba(167,139,250,0.28)";
      ctx.lineWidth = 0.4;
      ctx.beginPath();
      const { nv, edges } = tetEdges;
      for (let i = 0; i < edges.length; i += 2) {
        const a = edges[i], b = edges[i + 1];
        const pa = project(nv[a * 3], nv[a * 3 + 1], nv[a * 3 + 2]);
        const pb = project(nv[b * 3], nv[b * 3 + 1], nv[b * 3 + 2]);
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
      }
      ctx.stroke();
    }

    // Axis gizmo (bottom-left)
    const ax0 = 32, ay0 = cssH - 32, axLen = 22;
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
  }, [leafGeo, tetEdges, yaw, pitch, zoom, colorMode, showCells]);

  // Pointer handlers
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

  return (
    <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
          3d preview · drag to orbit · scroll to zoom
        </div>
        <div className="flex gap-1 text-[10px]">
          {(["partition", "depth", "boundary"] as ColorMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setColorMode(m)}
              className={`px-2 py-1 rounded uppercase tracking-[0.18em] border ${
                colorMode === m
                  ? "border-primary bg-primary/15 text-foreground"
                  : "border-border text-muted-foreground hover:text-foreground"
              }`}
            >
              {m}
            </button>
          ))}
          <button
            onClick={() => setShowCells((v) => !v)}
            className={`px-2 py-1 rounded uppercase tracking-[0.18em] border ${
              showCells ? "border-primary bg-primary/15 text-foreground" : "border-border text-muted-foreground"
            }`}
          >
            cells
          </button>
          <button
            onClick={() => setShowTets((v) => !v)}
            className={`px-2 py-1 rounded uppercase tracking-[0.18em] border ${
              showTets ? "border-primary bg-primary/15 text-foreground" : "border-border text-muted-foreground"
            }`}
          >
            tets
          </button>
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
      <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
        <span>{mesh.leaves.length.toLocaleString()} leaves · {(mesh.tets.length / 4).toLocaleString()} tets</span>
        <span>yaw {yaw.toFixed(2)} · pitch {pitch.toFixed(2)} · zoom {zoom.toFixed(2)}</span>
      </div>
    </div>
  );
}
