/**
 * Thermal Field Panel — runs the Phase-2 thermal engine on an in-panel
 * mesh and visualizes the resulting temperature field T, hotspot map,
 * and per-tet heat-flux vectors over a Canvas2D projection.
 *
 * Self-contained: generates a bar mesh via Geometry OS, pins Dirichlet
 * temperatures on the ±X end-caps, calls `solveThermal`, and renders
 * vertex colors (viridis ramp) plus optional flux arrows.
 */
import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { generateMesh, type MeshingResult } from "@/lib/meshing";
import {
  solveThermal, type ThermalSolution,
  inverseDesignKappa,
  differentiateThermal,
  targetTemperatureLoss,
} from "@/lib/pde";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type FieldMode = "temperature" | "hotspot";

type FaceKey = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
type HotMode = "dirichlet" | "neumann" | "insulated";

interface NeumannBC {
  /** Stable id for list editing. */
  id: string;
  face: FaceKey;
  /** Inward heat flux q_N (W/m²). Positive = heat entering the body. */
  flux: number;
}

interface Params {
  length: number;
  width: number;
  height: number;
  hotT: number;
  coldT: number;
  hotMode: HotMode;
  /** Inward flux on the +x face when hotMode = "neumann" (W/m²). */
  hotFlux: number;
  kappa: number;
  minDepth: number;
  maxDepth: number;
  /** Extra Neumann patches on side faces. */
  neumann: NeumannBC[];
}

const DEFAULTS: Params = {
  length: 2, width: 0.4, height: 0.4,
  hotT: 400, coldT: 300,
  hotMode: "dirichlet", hotFlux: 50_000,
  kappa: 45,
  minDepth: 2, maxDepth: 3,
  neumann: [],
};

// Viridis-ish 5-stop ramp.
const RAMP: Array<[number, [number, number, number]]> = [
  [0.0, [ 68,   1,  84]],
  [0.25,[ 59,  82, 139]],
  [0.5, [ 33, 145, 140]],
  [0.75,[ 94, 201,  98]],
  [1.0, [253, 231,  37]],
];
function ramp(t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (x <= RAMP[i][0]) {
      const a = RAMP[i - 1], b = RAMP[i];
      const f = (x - a[0]) / (b[0] - a[0]);
      return [
        Math.round(a[1][0] + (b[1][0] - a[1][0]) * f),
        Math.round(a[1][1] + (b[1][1] - a[1][1]) * f),
        Math.round(a[1][2] + (b[1][2] - a[1][2]) * f),
      ];
    }
  }
  return RAMP[RAMP.length - 1][1];
}

interface SolveOutput {
  mesh: MeshingResult;
  thermal: ThermalSolution;
  kappa: Float64Array;
  dirichletCount: { hot: number; cold: number };
  neumannSummary: Array<{ face: FaceKey; flux: number; area: number; nodes: number; power: number }>;
  totalNeumannPower: number;
  elapsedMs: number;
  Tmin: number; Tmax: number;
  fluxMax: number;
  dirichletSet: Set<number>;
  loads: Float64Array;
}

/** Min/max bbox extent along each axis for a face key. */
function faceTest(
  face: FaceKey, bb: { min: ReadonlyArray<number>; max: ReadonlyArray<number> }, tol: number,
): (x: number, y: number, z: number) => boolean {
  switch (face) {
    case "-x": return (x) => x <= bb.min[0] + tol;
    case "+x": return (x) => x >= bb.max[0] - tol;
    case "-y": return (_x, y) => y <= bb.min[1] + tol;
    case "+y": return (_x, y) => y >= bb.max[1] - tol;
    case "-z": return (_x, _y, z) => z <= bb.min[2] + tol;
    case "+z": return (_x, _y, z) => z >= bb.max[2] - tol;
  }
}

/** Surface triangles (faces shared by exactly one tet). */
function extractSurfaceTriangles(tets: Uint32Array): Array<[number, number, number]> {
  const map = new Map<string, { tri: [number, number, number]; count: number }>();
  const add = (a: number, b: number, c: number) => {
    const sorted = [a, b, c].sort((x, y) => x - y) as [number, number, number];
    const k = sorted.join(",");
    const ex = map.get(k);
    if (ex) ex.count++; else map.set(k, { tri: [a, b, c], count: 1 });
  };
  const n = tets.length / 4;
  for (let t = 0; t < n; t++) {
    const v0 = tets[t * 4], v1 = tets[t * 4 + 1];
    const v2 = tets[t * 4 + 2], v3 = tets[t * 4 + 3];
    add(v0, v1, v2); add(v0, v1, v3); add(v0, v2, v3); add(v1, v2, v3);
  }
  const out: Array<[number, number, number]> = [];
  map.forEach((v) => { if (v.count === 1) out.push(v.tri); });
  return out;
}

function triArea(
  v: Float32Array, a: number, b: number, c: number,
): number {
  const ax = v[a * 3], ay = v[a * 3 + 1], az = v[a * 3 + 2];
  const bx = v[b * 3] - ax, by = v[b * 3 + 1] - ay, bz = v[b * 3 + 2] - az;
  const cx = v[c * 3] - ax, cy = v[c * 3 + 1] - ay, cz = v[c * 3 + 2] - az;
  const nx = by * cz - bz * cy;
  const ny = bz * cx - bx * cz;
  const nz = bx * cy - by * cx;
  return 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
}

function runSolve(params: Params, kappaOverride?: Float64Array): SolveOutput {
  const t0 = performance.now();
  const { length, width, height } = params;
  const mesh = generateMesh({
    bbox: { min: [0, 0, 0], max: [length, width, height] },
    seeds: [],
    octree: { minDepth: params.minDepth, maxDepth: params.maxDepth },
    partitionCount: 1,
  });
  const verts = mesh.mesh.vertices;
  const tets = mesh.mesh.tets;
  const bb = mesh.mesh.bbox;
  const nV = verts.length / 3;
  const nTets = tets.length / 4;
  const ext = Math.max(length, width, height);
  const tol = ext * 1e-4;

  // Dirichlet on -x (cold) always; +x conditional.
  const onMinusX = faceTest("-x", bb, tol);
  const onPlusX = faceTest("+x", bb, tol);
  const dirichlet: { index: number; value: number }[] = [];
  let coldCount = 0, hotCount = 0;
  const dirichletSet = new Set<number>();
  for (let i = 0; i < nV; i++) {
    const x = verts[i * 3], y = verts[i * 3 + 1], z = verts[i * 3 + 2];
    if (onMinusX(x, y, z)) {
      dirichlet.push({ index: i, value: params.coldT });
      dirichletSet.add(i);
      coldCount++;
    } else if (params.hotMode === "dirichlet" && onPlusX(x, y, z)) {
      dirichlet.push({ index: i, value: params.hotT });
      dirichletSet.add(i);
      hotCount++;
    }
  }

  // Build combined Neumann list (hot-face Neumann + user side patches).
  const neumannBCs: Array<{ face: FaceKey; flux: number }> = [];
  if (params.hotMode === "neumann") {
    neumannBCs.push({ face: "+x", flux: params.hotFlux });
  }
  for (const bc of params.neumann) {
    if (bc.flux !== 0) neumannBCs.push({ face: bc.face, flux: bc.flux });
  }

  // Integrate Neumann fluxes into nodal loads via surface triangles.
  // For each tri whose 3 vertices all lie on the face plane, distribute
  // (flux · area) equally to its 3 vertices. Dirichlet nodes still receive
  // the contribution but the Dirichlet pin overrides them in the solver.
  const surface = extractSurfaceTriangles(tets);
  const loads = new Float64Array(nV);
  const summary: SolveOutput["neumannSummary"] = [];
  for (const bc of neumannBCs) {
    const test = faceTest(bc.face, bb, tol);
    let area = 0;
    const nodes = new Set<number>();
    for (const [a, b, c] of surface) {
      const ax = verts[a * 3], ay = verts[a * 3 + 1], az = verts[a * 3 + 2];
      const bx = verts[b * 3], by = verts[b * 3 + 1], bz = verts[b * 3 + 2];
      const cx = verts[c * 3], cy = verts[c * 3 + 1], cz = verts[c * 3 + 2];
      if (test(ax, ay, az) && test(bx, by, bz) && test(cx, cy, cz)) {
        const A = triArea(verts as Float32Array, a, b, c);
        const share = (bc.flux * A) / 3;
        loads[a] += share; loads[b] += share; loads[c] += share;
        area += A;
        nodes.add(a); nodes.add(b); nodes.add(c);
      }
    }
    summary.push({
      face: bc.face, flux: bc.flux, area,
      nodes: nodes.size, power: bc.flux * area,
    });
  }
  const totalNeumannPower = summary.reduce((s, x) => s + x.power, 0);

  const kappa = new Float64Array(nTets);
  if (kappaOverride && kappaOverride.length === nTets) {
    kappa.set(kappaOverride);
  } else {
    for (let t = 0; t < nTets; t++) kappa[t] = params.kappa;
  }

  const thermal = solveThermal({
    mesh: { vertices: verts, tets },
    kappa,
    dirichlet,
    neumannLoads: loads,
    referenceTemperature: (params.hotT + params.coldT) / 2,
  });

  let Tmin = Infinity, Tmax = -Infinity;
  for (let i = 0; i < thermal.T.length; i++) {
    if (thermal.T[i] < Tmin) Tmin = thermal.T[i];
    if (thermal.T[i] > Tmax) Tmax = thermal.T[i];
  }
  let fluxMax = 0;
  for (let i = 0; i < thermal.fluxMagnitude.length; i++) {
    if (thermal.fluxMagnitude[i] > fluxMax) fluxMax = thermal.fluxMagnitude[i];
  }
  return {
    mesh, thermal, kappa,
    dirichletCount: { hot: hotCount, cold: coldCount },
    neumannSummary: summary,
    totalNeumannPower,
    elapsedMs: performance.now() - t0,
    Tmin, Tmax, fluxMax,
    dirichletSet, loads,
  };
}

interface Probe {
  id: string;
  index: number;
  target: number;
  weight: number;
}

interface ViewerProps {
  out: SolveOutput;
  mode: FieldMode;
  showFlux: boolean;
  height?: number;
  probes?: Probe[];
  pickArmed?: boolean;
  onPick?: (vertexIndex: number) => void;
  overrideField?: {
    values: Float64Array;
    min: number;
    max: number;
    label: string;
    /** Optional unit string for the legend numbers. */
    unit?: string;
    /** If true, normalize per-vertex by the symmetric max |v| (signed → diverging mapping). */
    diverging?: boolean;
  } | null;
}

function ThermalViewer({
  out, mode, showFlux, height = 360,
  probes = [], pickArmed = false, onPick,
  overrideField = null,
}: ViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [yaw, setYaw] = useState(0.7);
  const [pitch, setPitch] = useState(-0.35);
  const [zoom, setZoom] = useState(1);
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number; moved: boolean } | null>(null);
  const projectedRef = useRef<{ px: Float32Array; py: Float32Array; pz: Float32Array } | null>(null);

  const geo = useMemo(() => {
    const verts = out.mesh.mesh.vertices;
    const tets = out.mesh.mesh.tets;
    const bb = out.mesh.mesh.bbox;
    const ext = Math.max(
      bb.max[0] - bb.min[0],
      bb.max[1] - bb.min[1],
      bb.max[2] - bb.min[2],
    ) || 1;
    const cx = (bb.min[0] + bb.max[0]) / 2;
    const cy = (bb.min[1] + bb.max[1]) / 2;
    const cz = (bb.min[2] + bb.max[2]) / 2;
    const nV = verts.length / 3;
    const norm = new Float32Array(nV * 3);
    for (let i = 0; i < nV; i++) {
      norm[i * 3]     = (verts[i * 3]     - cx) / ext;
      norm[i * 3 + 1] = (verts[i * 3 + 1] - cy) / ext;
      norm[i * 3 + 2] = (verts[i * 3 + 2] - cz) / ext;
    }
    const nTets = tets.length / 4;
    const tetCentroid = new Float32Array(nTets * 3);
    for (let t = 0; t < nTets; t++) {
      let xs = 0, ys = 0, zs = 0;
      for (let k = 0; k < 4; k++) {
        const v = tets[t * 4 + k];
        xs += norm[v * 3];
        ys += norm[v * 3 + 1];
        zs += norm[v * 3 + 2];
      }
      tetCentroid[t * 3]     = xs / 4;
      tetCentroid[t * 3 + 1] = ys / 4;
      tetCentroid[t * 3 + 2] = zs / 4;
    }
    // Surface triangles: faces shared by only one tet.
    const faceMap = new Map<string, { a: number; b: number; c: number; count: number }>();
    const addFace = (a: number, b: number, c: number) => {
      const sorted = [a, b, c].sort((x, y) => x - y);
      const key = sorted.join(",");
      const ex = faceMap.get(key);
      if (ex) ex.count++;
      else faceMap.set(key, { a, b, c, count: 1 });
    };
    for (let t = 0; t < nTets; t++) {
      const v0 = tets[t * 4], v1 = tets[t * 4 + 1];
      const v2 = tets[t * 4 + 2], v3 = tets[t * 4 + 3];
      addFace(v0, v1, v2); addFace(v0, v1, v3);
      addFace(v0, v2, v3); addFace(v1, v2, v3);
    }
    const surface: Array<[number, number, number]> = [];
    faceMap.forEach((f) => { if (f.count === 1) surface.push([f.a, f.b, f.c]); });
    return { norm, tetCentroid, surface, nV, nTets };
  }, [out]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.clientWidth * devicePixelRatio;
    const H = c.clientHeight * devicePixelRatio;
    c.width = W; c.height = H;
    ctx.fillStyle = "#0a0a0a";
    ctx.fillRect(0, 0, W, H);

    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const project = (x: number, y: number, z: number): [number, number, number] => {
      const xr = x * cy - z * sy;
      const zr = x * sy + z * cy;
      const yr = y * cp - zr * sp;
      const zr2 = y * sp + zr * cp;
      const dist = 2.5;
      const f = (W * 0.45 * zoom) / (dist + zr2);
      return [W / 2 + xr * f, H / 2 - yr * f, zr2];
    };

    const T = out.thermal.T;
    const hot = out.thermal.hotspot;
    const Tspan = Math.max(1e-12, out.Tmax - out.Tmin);
    let fieldNorm: (i: number) => number;
    if (overrideField) {
      const { values, min, max, diverging } = overrideField;
      if (diverging) {
        const mAbs = Math.max(Math.abs(min), Math.abs(max), 1e-30);
        fieldNorm = (i) => 0.5 + 0.5 * Math.max(-1, Math.min(1, values[i] / mAbs));
      } else {
        const span = Math.max(1e-30, max - min);
        fieldNorm = (i) => (values[i] - min) / span;
      }
    } else if (mode === "temperature") {
      fieldNorm = (i) => (T[i] - out.Tmin) / Tspan;
    } else {
      fieldNorm = (i) => hot[i];
    }
    void hot;

    // Project all verts.
    const px = new Float32Array(geo.nV);
    const py = new Float32Array(geo.nV);
    const pz = new Float32Array(geo.nV);
    for (let i = 0; i < geo.nV; i++) {
      const [x, y, z] = project(
        geo.norm[i * 3], geo.norm[i * 3 + 1], geo.norm[i * 3 + 2],
      );
      px[i] = x; py[i] = y; pz[i] = z;
    }
    projectedRef.current = { px, py, pz };

    // Sort surface triangles back-to-front by avg depth.
    const tris = geo.surface.map((tri) => {
      const [a, b, c] = tri;
      return { a, b, c, z: (pz[a] + pz[b] + pz[c]) / 3 };
    }).sort((u, v) => u.z - v.z);

    for (const tri of tris) {
      const { a, b, c } = tri;
      // Backface cull.
      const ux = px[b] - px[a], uy = py[b] - py[a];
      const vx = px[c] - px[a], vy = py[c] - py[a];
      const cross = ux * vy - uy * vx;
      if (cross <= 0) continue;
      const fa = fieldNorm(a), fb = fieldNorm(b), fc = fieldNorm(c);
      const f = (fa + fb + fc) / 3;
      const [r, g, bl] = ramp(f);
      ctx.beginPath();
      ctx.moveTo(px[a], py[a]);
      ctx.lineTo(px[b], py[b]);
      ctx.lineTo(px[c], py[c]);
      ctx.closePath();
      ctx.fillStyle = `rgb(${r},${g},${bl})`;
      ctx.fill();
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.stroke();
    }

    if (showFlux && out.fluxMax > 0) {
      const flux = out.thermal.fluxPerTet;
      const fmag = out.thermal.fluxMagnitude;
      const scale = 0.08 / Math.max(1e-12, out.fluxMax);
      ctx.lineWidth = 1;
      // Subsample for clarity (cap at 600 arrows).
      const step = Math.max(1, Math.floor(geo.nTets / 600));
      for (let t = 0; t < geo.nTets; t += step) {
        const m = fmag[t];
        if (m === 0) continue;
        const cx0 = geo.tetCentroid[t * 3];
        const cy0 = geo.tetCentroid[t * 3 + 1];
        const cz0 = geo.tetCentroid[t * 3 + 2];
        const fxw = flux[t * 3] * scale;
        const fyw = flux[t * 3 + 1] * scale;
        const fzw = flux[t * 3 + 2] * scale;
        const [x0, y0] = project(cx0, cy0, cz0);
        const [x1, y1] = project(cx0 + fxw, cy0 + fyw, cz0 + fzw);
        ctx.strokeStyle = `rgba(248, 250, 252, ${0.35 + 0.55 * (m / out.fluxMax)})`;
        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }

    // Legend.
    const lw = 160 * devicePixelRatio, lh = 10 * devicePixelRatio;
    const lx = 16 * devicePixelRatio, ly = H - 28 * devicePixelRatio;
    const grad = ctx.createLinearGradient(lx, 0, lx + lw, 0);
    for (let i = 0; i <= 10; i++) {
      const [r, g, bl] = ramp(i / 10);
      grad.addColorStop(i / 10, `rgb(${r},${g},${bl})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(lx, ly, lw, lh);
    ctx.fillStyle = "rgba(229,231,235,0.9)";
    ctx.font = `${10 * devicePixelRatio}px ui-sans-serif, system-ui`;
    let lo: string, hi: string, midLabel: string;
    if (overrideField) {
      const { min, max, label, unit, diverging } = overrideField;
      const u = unit ? ` ${unit}` : "";
      if (diverging) {
        const mAbs = Math.max(Math.abs(min), Math.abs(max));
        lo = `${(-mAbs).toExponential(1)}${u}`;
        hi = `${(+mAbs).toExponential(1)}${u}`;
      } else {
        lo = `${min.toExponential(1)}${u}`;
        hi = `${max.toExponential(1)}${u}`;
      }
      midLabel = label;
    } else if (mode === "temperature") {
      lo = `${out.Tmin.toFixed(1)} K`;
      hi = `${out.Tmax.toFixed(1)} K`;
      midLabel = "T";
    } else {
      lo = "0"; hi = "1"; midLabel = "hotspot";
    }
    ctx.fillText(lo, lx, ly - 4 * devicePixelRatio);
    const hiW = ctx.measureText(hi).width;
    ctx.fillText(hi, lx + lw - hiW, ly - 4 * devicePixelRatio);
    ctx.fillText(midLabel, lx, ly + lh + 12 * devicePixelRatio);
    // Probe markers (drawn on top).
    if (probes.length > 0) {
      const T = out.thermal.T;
      for (const p of probes) {
        if (p.index < 0 || p.index >= geo.nV) continue;
        const x = px[p.index], y = py[p.index];
        const cur = T[p.index];
        const err = cur - p.target;
        const ok = Math.abs(err) < 0.5;
        ctx.beginPath();
        ctx.arc(x, y, 6 * devicePixelRatio, 0, Math.PI * 2);
        ctx.fillStyle = ok ? "rgba(34,197,94,0.95)" : "rgba(239,68,68,0.95)";
        ctx.fill();
        ctx.lineWidth = 1.5 * devicePixelRatio;
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.stroke();
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.font = `${10 * devicePixelRatio}px ui-monospace, monospace`;
        ctx.fillText(
          `#${p.index} → ${p.target.toFixed(0)}K (${cur.toFixed(0)})`,
          x + 9 * devicePixelRatio, y - 6 * devicePixelRatio,
        );
      }
    }
  }, [out, geo, yaw, pitch, zoom, mode, showFlux, probes, overrideField]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!pickArmed || !onPick) return;
    const proj = projectedRef.current;
    const c = canvasRef.current;
    if (!proj || !c) return;
    const rect = c.getBoundingClientRect();
    const mx = (e.clientX - rect.left) * devicePixelRatio;
    const my = (e.clientY - rect.top) * devicePixelRatio;
    let best = -1, bestD = Infinity;
    for (let i = 0; i < geo.nV; i++) {
      const dx = proj.px[i] - mx, dy = proj.py[i] - my;
      const d = dx * dx + dy * dy;
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0 && bestD < (24 * devicePixelRatio) * (24 * devicePixelRatio)) {
      onPick(best);
    }
  }, [pickArmed, onPick, geo]);

  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-border bg-black"
      style={{ height }}
      onMouseDown={(e) => {
        drag.current = { x: e.clientX, y: e.clientY, yaw, pitch, moved: false };
      }}
      onMouseMove={(e) => {
        if (!drag.current) return;
        const dx = e.clientX - drag.current.x;
        const dy = e.clientY - drag.current.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.current.moved = true;
        setYaw(drag.current.yaw + dx * 0.01);
        setPitch(drag.current.pitch + dy * 0.01);
      }}
      onMouseUp={(e) => {
        const wasDrag = drag.current?.moved;
        drag.current = null;
        if (!wasDrag) handleClick(e);
      }}
      onMouseLeave={() => { drag.current = null; }}
      onWheel={(e) => {
        setZoom((z) => Math.max(0.3, Math.min(4, z * (e.deltaY < 0 ? 1.1 : 0.9))));
      }}
    >
      <canvas
        ref={canvasRef}
        className={`h-full w-full ${pickArmed ? "cursor-crosshair" : "cursor-grab active:cursor-grabbing"}`}
      />
      {pickArmed && (
        <div className="absolute top-2 left-2 rounded bg-primary/90 text-primary-foreground text-[11px] px-2 py-1 font-mono">
          Click a vertex to add probe
        </div>
      )}
    </div>
  );
}

type PanelMode = "forward" | "optimize";

interface OptimizeOptions {
  steps: number;
  learningRate: number;
  kappaMin: number;
  kappaMax: number;
  regularization: number;
}

const OPT_DEFAULTS: OptimizeOptions = {
  steps: 25,
  learningRate: 0.08,
  kappaMin: 0.1,
  kappaMax: 500,
  regularization: 0.0,
};

interface OptimizeResult {
  history: Array<{ step: number; loss: number; gradNorm: number }>;
  kappa: Float64Array;
  elapsedMs: number;
  kappaMin: number;
  kappaMax: number;
}

export function ThermalFieldPanel() {
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [out, setOut] = useState<SolveOutput | null>(null);
  const [mode, setMode] = useState<FieldMode>("temperature");
  const [showFlux, setShowFlux] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [panelMode, setPanelMode] = useState<PanelMode>("forward");
  const [probes, setProbes] = useState<Probe[]>([]);
  const [pickArmed, setPickArmed] = useState(false);
  const [optOpts, setOptOpts] = useState<OptimizeOptions>(OPT_DEFAULTS);
  const [optResult, setOptResult] = useState<OptimizeResult | null>(null);
  const [kappaField, setKappaField] = useState<Float64Array | null>(null);

  interface GradDiag {
    /** Per-vertex |dL/dκ| spread from incident tets (for surface heatmap). */
    perVertex: Float64Array;
    /** Per-tet ∂L/∂κ. */
    perTet: Float64Array;
    /** Per-tet |∂L/∂κ|·κ (log-space gradient). */
    perTetLog: Float64Array;
    minTet: number; maxTet: number;
    minVtx: number; maxVtx: number;
    loss: number;
    gradNormKappa: number;
    gradNormLogKappa: number;
    gradNormSource: number;
    gradNormLoads: number;
  }
  const [gradDiag, setGradDiag] = useState<GradDiag | null>(null);
  const [showGradOverlay, setShowGradOverlay] = useState(false);

  const run = (kappaOverride?: Float64Array) => {
    setBusy(true); setErr(null);
    setTimeout(() => {
      try {
        setOut(runSolve(params, kappaOverride));
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    }, 0);
  };

  useEffect(() => { run(); /* initial */ // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reset probes/optimization when geometry changes (probe vertex indices
  // are mesh-dependent and become invalid on remesh).
  useEffect(() => {
    setProbes([]);
    setOptResult(null);
    setKappaField(null);
    setPickArmed(false);
    setGradDiag(null);
    setShowGradOverlay(false);
  }, [params.length, params.width, params.height, params.minDepth, params.maxDepth]);

  const set = <K extends keyof Params>(k: K, v: Params[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

  const addProbe = useCallback((vertexIndex: number) => {
    setProbes((prev) => {
      if (prev.some((p) => p.index === vertexIndex)) return prev;
      const currentT = out?.thermal.T[vertexIndex] ?? params.coldT;
      return [...prev, {
        id: `pr_${Date.now()}_${Math.random().toString(36).slice(2, 5)}`,
        index: vertexIndex,
        target: Math.round(currentT),
        weight: 1,
      }];
    });
    setPickArmed(false);
  }, [out, params.coldT]);

  const runOptimize = () => {
    if (!out) { setErr("Run forward solver first."); return; }
    if (probes.length === 0) { setErr("Add at least one probe target."); return; }
    setBusy(true); setErr(null);
    setTimeout(() => {
      try {
        const t0 = performance.now();
        const dirichletArr = Array.from(out.dirichletSet).map((index) => ({
          index,
          value: out.thermal.T[index],
        }));
        const history: OptimizeResult["history"] = [];
        const k0 = kappaField ?? out.kappa;
        const result = inverseDesignKappa(
          {
            mesh: { vertices: out.mesh.mesh.vertices, tets: out.mesh.mesh.tets },
            kappa: new Float64Array(k0),
            neumannLoads: out.loads,
            dirichlet: dirichletArr,
          },
          probes.map((p) => ({ index: p.index, target: p.target, weight: p.weight })),
          {
            steps: optOpts.steps,
            learningRate: optOpts.learningRate,
            kappaMin: optOpts.kappaMin,
            kappaMax: optOpts.kappaMax,
            regularization: optOpts.regularization,
            onStep: (step, loss, _kappa) => {
              history.push({ step, loss, gradNorm: 0 });
            },
          },
        );
        // Replace gradNorm placeholders with the precise history from the solver.
        const final: OptimizeResult = {
          history: result.history.length ? result.history : history,
          kappa: result.kappa,
          elapsedMs: performance.now() - t0,
          kappaMin: Math.min(...Array.from(result.kappa)),
          kappaMax: Math.max(...Array.from(result.kappa)),
        };
        setOptResult(final);
        setKappaField(result.kappa);
        setGradDiag(null);
        // Re-run forward visualization with optimized κ.
        setOut(runSolve(params, result.kappa));
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    }, 0);
  };

  const resetKappa = () => {
    setKappaField(null);
    setOptResult(null);
    setGradDiag(null);
    setShowGradOverlay(false);
    run();
  };

  const computeGradient = () => {
    if (!out) { setErr("Run forward solver first."); return; }
    if (probes.length === 0) { setErr("Add at least one probe to define a loss."); return; }
    setBusy(true); setErr(null);
    setTimeout(() => {
      try {
        const dirichletArr = Array.from(out.dirichletSet).map((index) => ({
          index, value: out.thermal.T[index],
        }));
        const kappaForGrad = kappaField ?? out.kappa;
        const { loss, dLdT } = targetTemperatureLoss(
          out.thermal.T,
          probes.map((p) => ({ index: p.index, target: p.target, weight: p.weight })),
        );
        const grads = differentiateThermal(
          {
            mesh: { vertices: out.mesh.mesh.vertices, tets: out.mesh.mesh.tets },
            kappa: new Float64Array(kappaForGrad),
            neumannLoads: out.loads,
            dirichlet: dirichletArr,
          },
          { dLdT },
          out.thermal,
        );

        const tets = out.mesh.mesh.tets;
        const nV = out.thermal.T.length;
        const nT = grads.dLdKappa.length;
        const perTetLog = new Float64Array(nT);
        let minTet = Infinity, maxTet = -Infinity;
        let gNorm2 = 0, gNorm2Log = 0;
        for (let t = 0; t < nT; t++) {
          const g = grads.dLdKappa[t];
          const gLog = g * kappaForGrad[t];
          perTetLog[t] = gLog;
          if (g < minTet) minTet = g;
          if (g > maxTet) maxTet = g;
          gNorm2 += g * g;
          gNorm2Log += gLog * gLog;
        }

        // Spread |dL/dκ| to vertices by averaging |g_t| over incident tets
        // → produces a surface heatmap proxy of where the gradient lives.
        const accum = new Float64Array(nV);
        const count = new Uint32Array(nV);
        for (let t = 0; t < nT; t++) {
          const a = Math.abs(grads.dLdKappa[t]);
          for (let k = 0; k < 4; k++) {
            const vid = tets[t * 4 + k];
            accum[vid] += a;
            count[vid] += 1;
          }
        }
        let minVtx = Infinity, maxVtx = -Infinity;
        const perVertex = new Float64Array(nV);
        for (let i = 0; i < nV; i++) {
          const v = count[i] > 0 ? accum[i] / count[i] : 0;
          perVertex[i] = v;
          if (v < minVtx) minVtx = v;
          if (v > maxVtx) maxVtx = v;
        }

        let gNormSrc2 = 0, gNormLd2 = 0;
        for (let i = 0; i < nV; i++) gNormSrc2 += grads.dLdSource[i] * grads.dLdSource[i];
        for (let i = 0; i < nV; i++) gNormLd2  += grads.dLdLoads[i]  * grads.dLdLoads[i];

        setGradDiag({
          perVertex, perTet: grads.dLdKappa, perTetLog,
          minTet, maxTet, minVtx, maxVtx,
          loss,
          gradNormKappa:    Math.sqrt(gNorm2),
          gradNormLogKappa: Math.sqrt(gNorm2Log),
          gradNormSource:   Math.sqrt(gNormSrc2),
          gradNormLoads:    Math.sqrt(gNormLd2),
        });
        setShowGradOverlay(true);
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    }, 0);
  };


  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <CardTitle>Thermal Field Engine</CardTitle>
            <div className="text-xs text-muted-foreground mt-1">
              steady-state −∇·(κ∇T) = 0 on Geometry OS mesh · FEM P1 tets · PCG solver
            </div>
          </div>
          <div className="flex items-center gap-2">
            {out && (
              <>
                <Badge variant="outline">{out.mesh.summary.tets.count.toLocaleString()} tets</Badge>
                <Badge variant="outline">{out.thermal.solve.result.iterations} iters</Badge>
                <Badge variant={out.thermal.solve.result.converged ? "default" : "destructive"}>
                  {out.thermal.solve.result.converged ? "converged" : "no conv"}
                </Badge>
                <Badge variant="outline">{out.elapsedMs.toFixed(0)} ms</Badge>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <NumField label="Length" value={params.length} step={0.1} onChange={(v) => set("length", v)} />
          <NumField label="Width" value={params.width} step={0.05} onChange={(v) => set("width", v)} />
          <NumField label="Height" value={params.height} step={0.05} onChange={(v) => set("height", v)} />
          <NumField label="κ (W/m·K)" value={params.kappa} step={1} onChange={(v) => set("kappa", v)} />
          <NumField label="Cold face T (K)" value={params.coldT} step={5} onChange={(v) => set("coldT", v)} />
          <div className="space-y-1">
            <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">Hot face (+x) BC</Label>
            <Select value={params.hotMode} onValueChange={(v) => set("hotMode", v as HotMode)}>
              <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="dirichlet">Dirichlet (fixed T)</SelectItem>
                <SelectItem value="neumann">Neumann (heat flux)</SelectItem>
                <SelectItem value="insulated">Insulated (q=0)</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {params.hotMode === "dirichlet" && (
            <NumField label="Hot face T (K)" value={params.hotT} step={5} onChange={(v) => set("hotT", v)} />
          )}
          {params.hotMode === "neumann" && (
            <NumField label="Hot face flux (W/m²)" value={params.hotFlux} step={1000} onChange={(v) => set("hotFlux", v)} />
          )}
          <NumField label="min depth" value={params.minDepth} step={1} onChange={(v) => set("minDepth", Math.max(1, Math.round(v)))} />
          <NumField label="max depth" value={params.maxDepth} step={1} onChange={(v) => set("maxDepth", Math.max(params.minDepth, Math.round(v)))} />
        </div>

        <NeumannEditor
          value={params.neumann}
          onChange={(n) => set("neumann", n)}
        />

        <div className="flex items-center gap-2 flex-wrap">
          <Button onClick={() => run(kappaField ?? undefined)} disabled={busy} size="sm">
            {busy ? "Solving…" : "Run solver"}
          </Button>
          <div className="inline-flex rounded-md border border-border overflow-hidden">
            <button
              type="button"
              onClick={() => setPanelMode("forward")}
              className={`px-3 py-1.5 text-xs ${panelMode === "forward" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground"}`}
            >Forward</button>
            <button
              type="button"
              onClick={() => setPanelMode("optimize")}
              className={`px-3 py-1.5 text-xs ${panelMode === "optimize" ? "bg-primary text-primary-foreground" : "bg-background text-muted-foreground"}`}
            >Optimize κ</button>
          </div>
          {kappaField && (
            <Badge variant="secondary" className="font-mono">
              κ range {Math.min(...Array.from(kappaField)).toFixed(2)} – {Math.max(...Array.from(kappaField)).toFixed(2)}
            </Badge>
          )}
          {kappaField && (
            <Button onClick={resetKappa} size="sm" variant="ghost">Reset κ</Button>
          )}
          <Select value={mode} onValueChange={(v) => setMode(v as FieldMode)}>
            <SelectTrigger className="w-[180px] h-9"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="temperature">Temperature</SelectItem>
              <SelectItem value="hotspot">Hotspot map</SelectItem>
            </SelectContent>
          </Select>
          <Button
            variant={showFlux ? "default" : "outline"}
            size="sm"
            onClick={() => setShowFlux((s) => !s)}
          >
            Heat-flux vectors {showFlux ? "on" : "off"}
          </Button>
        </div>

        {err && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {err}
          </div>
        )}

        {out && (
          <>
            <ThermalViewer
              out={out}
              mode={mode}
              showFlux={showFlux && !(showGradOverlay && gradDiag)}
              probes={probes}
              pickArmed={panelMode === "optimize" && pickArmed}
              onPick={addProbe}
              overrideField={showGradOverlay && gradDiag ? {
                values: gradDiag.perVertex,
                min: gradDiag.minVtx,
                max: gradDiag.maxVtx,
                label: "|∂L/∂κ| (vtx-avg)",
                unit: "",
              } : null}
            />
            {panelMode === "optimize" && (
              <OptimizePanel
                probes={probes}
                onProbesChange={setProbes}
                pickArmed={pickArmed}
                onTogglePick={() => setPickArmed((p) => !p)}
                opts={optOpts}
                onOptsChange={setOptOpts}
                onRun={runOptimize}
                busy={busy}
                result={optResult}
                currentT={out.thermal.T}
                onComputeGradient={computeGradient}
                gradDiag={gradDiag}
                showGradOverlay={showGradOverlay}
                onToggleGradOverlay={() => setShowGradOverlay((s) => !s)}
              />
            )}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
              <Stat label="T min" value={`${out.Tmin.toFixed(2)} K`} />
              <Stat label="T max" value={`${out.Tmax.toFixed(2)} K`} />
              <Stat label="ΔT" value={`${(out.Tmax - out.Tmin).toFixed(2)} K`} />
              <Stat label="|q| max" value={out.fluxMax.toExponential(2)} />
              <Stat label="Dirichlet" value={`hot ${out.dirichletCount.hot} · cold ${out.dirichletCount.cold}`} />
              <Stat label="vertices" value={out.thermal.T.length.toLocaleString()} />
              <Stat label="residual" value={out.thermal.solve.result.residual.toExponential(2)} />
              <Stat label="solver" value={`${out.thermal.solve.result.iterations} PCG iters`} />
              <Stat label="Σ Neumann power" value={`${out.totalNeumannPower.toFixed(1)} W`} />
            </div>
            {out.neumannSummary.length > 0 && (
              <div className="rounded-md border border-border bg-muted/20 p-3">
                <div className="text-[11px] uppercase tracking-wide text-muted-foreground mb-2">
                  Neumann patches (integrated)
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs font-mono">
                  {out.neumannSummary.map((s, i) => (
                    <div key={i} className="flex items-center justify-between rounded border border-border/60 bg-background/40 px-2 py-1">
                      <span>face <span className="text-foreground">{s.face}</span></span>
                      <span>q={s.flux.toFixed(0)} W/m²</span>
                      <span>A={s.area.toFixed(3)} m²</span>
                      <span>{s.nodes} nodes</span>
                      <span className={s.power >= 0 ? "text-emerald-400" : "text-amber-400"}>
                        {s.power >= 0 ? "+" : ""}{s.power.toFixed(1)} W
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}

function NumField({
  label, value, step, onChange,
}: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <div className="space-y-1">
      <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</Label>
      <Input
        type="number"
        value={value}
        step={step}
        className="h-9"
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (Number.isFinite(v)) onChange(v);
        }}
      />
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border bg-muted/30 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="font-mono text-sm text-foreground">{value}</div>
    </div>
  );
}

const FACE_OPTIONS: FaceKey[] = ["+x", "-x", "+y", "-y", "+z", "-z"];

function NeumannEditor({
  value, onChange,
}: { value: NeumannBC[]; onChange: (v: NeumannBC[]) => void }) {
  const add = () => {
    const used = new Set(value.map((b) => b.face));
    const next = FACE_OPTIONS.find((f) => !used.has(f)) ?? "+y";
    onChange([...value, { id: `bc_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`, face: next, flux: 10_000 }]);
  };
  const update = (id: string, patch: Partial<NeumannBC>) =>
    onChange(value.map((b) => (b.id === id ? { ...b, ...patch } : b)));
  const remove = (id: string) => onChange(value.filter((b) => b.id !== id));

  return (
    <div className="rounded-md border border-border bg-muted/20 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-muted-foreground">Neumann heat-flux patches</div>
          <div className="text-[11px] text-muted-foreground">Positive flux = heat entering the body (W/m²). Side faces default to insulated.</div>
        </div>
        <Button size="sm" variant="outline" onClick={add}>+ Add patch</Button>
      </div>
      {value.length === 0 && (
        <div className="text-xs text-muted-foreground italic">No side-face patches. Add one to inject or extract heat.</div>
      )}
      {value.map((bc) => (
        <div key={bc.id} className="grid grid-cols-[110px_1fr_auto] gap-2 items-end">
          <div className="space-y-1">
            <Label className="text-[10px] uppercase tracking-wide text-muted-foreground">Face</Label>
            <Select value={bc.face} onValueChange={(v) => update(bc.id, { face: v as FaceKey })}>
              <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
              <SelectContent>
                {FACE_OPTIONS.map((f) => (
                  <SelectItem key={f} value={f}>{f}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <NumField
            label="Flux q (W/m²)"
            value={bc.flux}
            step={1000}
            onChange={(v) => update(bc.id, { flux: v })}
          />
          <Button size="sm" variant="ghost" onClick={() => remove(bc.id)}>Remove</Button>
        </div>
      ))}
    </div>
  );
}

function OptimizePanel({
  probes, onProbesChange, pickArmed, onTogglePick,
  opts, onOptsChange, onRun, busy, result, currentT,
}: {
  probes: Probe[];
  onProbesChange: (p: Probe[]) => void;
  pickArmed: boolean;
  onTogglePick: () => void;
  opts: OptimizeOptions;
  onOptsChange: (o: OptimizeOptions) => void;
  onRun: () => void;
  busy: boolean;
  result: OptimizeResult | null;
  currentT: Float64Array;
}) {
  const updateProbe = (id: string, patch: Partial<Probe>) =>
    onProbesChange(probes.map((p) => (p.id === id ? { ...p, ...patch } : p)));
  const removeProbe = (id: string) => onProbesChange(probes.filter((p) => p.id !== id));

  const finalLoss = result?.history.length
    ? result.history[result.history.length - 1].loss
    : null;
  const initialLoss = result?.history.length ? result.history[0].loss : null;

  return (
    <div className="rounded-md border border-primary/40 bg-primary/5 p-3 space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-primary">Inverse design — Optimize κ</div>
          <div className="text-[11px] text-muted-foreground">
            Adjoint gradient on log(κ) minimizes ½·Σwᵢ·(Tᵢ−T*ᵢ)² at probe vertices.
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant={pickArmed ? "default" : "outline"} onClick={onTogglePick}>
            {pickArmed ? "Cancel pick" : "+ Pick probe on mesh"}
          </Button>
          <Button size="sm" onClick={onRun} disabled={busy || probes.length === 0}>
            {busy ? "Optimizing…" : `Run optimization (${opts.steps} steps)`}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
        <NumField label="steps" value={opts.steps} step={5}
          onChange={(v) => onOptsChange({ ...opts, steps: Math.max(1, Math.round(v)) })} />
        <NumField label="learning rate" value={opts.learningRate} step={0.01}
          onChange={(v) => onOptsChange({ ...opts, learningRate: Math.max(1e-4, v) })} />
        <NumField label="κ min" value={opts.kappaMin} step={0.05}
          onChange={(v) => onOptsChange({ ...opts, kappaMin: Math.max(1e-6, v) })} />
        <NumField label="κ max" value={opts.kappaMax} step={10}
          onChange={(v) => onOptsChange({ ...opts, kappaMax: Math.max(opts.kappaMin * 2, v) })} />
        <NumField label="reg (log κ)" value={opts.regularization} step={0.01}
          onChange={(v) => onOptsChange({ ...opts, regularization: Math.max(0, v) })} />
      </div>

      {probes.length === 0 ? (
        <div className="text-xs italic text-muted-foreground">
          No probes yet. Click <span className="font-mono">+ Pick probe on mesh</span>, then click a vertex on the viewport above.
        </div>
      ) : (
        <div className="space-y-1">
          {probes.map((p) => {
            const cur = currentT[p.index] ?? NaN;
            const err = cur - p.target;
            return (
              <div key={p.id} className="grid grid-cols-[80px_1fr_1fr_120px_auto] gap-2 items-end text-xs font-mono">
                <div className="text-foreground">#{p.index}</div>
                <NumField label="target T (K)" value={p.target} step={5}
                  onChange={(v) => updateProbe(p.id, { target: v })} />
                <NumField label="weight" value={p.weight} step={0.1}
                  onChange={(v) => updateProbe(p.id, { weight: Math.max(0, v) })} />
                <div className={`px-2 py-1 rounded border border-border/60 bg-background/40 ${Math.abs(err) < 1 ? "text-emerald-400" : "text-amber-400"}`}>
                  cur {cur.toFixed(1)} · Δ{err >= 0 ? "+" : ""}{err.toFixed(1)}
                </div>
                <Button size="sm" variant="ghost" onClick={() => removeProbe(p.id)}>Remove</Button>
              </div>
            );
          })}
        </div>
      )}

      {result && (
        <div className="space-y-2">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <Stat label="initial loss" value={initialLoss?.toExponential(3) ?? "—"} />
            <Stat label="final loss" value={finalLoss?.toExponential(3) ?? "—"} />
            <Stat label="κ range" value={`${result.kappaMin.toFixed(2)} – ${result.kappaMax.toFixed(2)}`} />
            <Stat label="elapsed" value={`${result.elapsedMs.toFixed(0)} ms`} />
          </div>
          <LossChart history={result.history} />
        </div>
      )}
    </div>
  );
}

function LossChart({ history }: { history: Array<{ step: number; loss: number; gradNorm: number }> }) {
  if (history.length < 2) {
    return <div className="text-xs text-muted-foreground italic">Need ≥2 steps to chart loss.</div>;
  }
  const W = 600, H = 140, pad = 28;
  const losses = history.map((h) => Math.max(h.loss, 1e-30));
  const lMin = Math.min(...losses);
  const lMax = Math.max(...losses);
  const useLog = lMax / Math.max(lMin, 1e-30) > 50;
  const toY = (v: number) => {
    const a = useLog ? Math.log10(v) : v;
    const a0 = useLog ? Math.log10(lMin) : lMin;
    const a1 = useLog ? Math.log10(lMax) : lMax;
    const span = Math.max(a1 - a0, 1e-12);
    return H - pad - ((a - a0) / span) * (H - 2 * pad);
  };
  const toX = (i: number) => pad + (i / (history.length - 1)) * (W - 2 * pad);
  const path = history.map((h, i) => `${i === 0 ? "M" : "L"}${toX(i).toFixed(1)},${toY(losses[i]).toFixed(1)}`).join(" ");
  return (
    <div className="rounded-md border border-border bg-background/40 p-2">
      <div className="flex items-center justify-between text-[11px] text-muted-foreground mb-1">
        <span>Loss vs step {useLog ? "(log scale)" : ""}</span>
        <span className="font-mono">{lMin.toExponential(2)} → {lMax.toExponential(2)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[140px]">
        <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad}
          fill="none" stroke="hsl(var(--border))" strokeDasharray="2 3" />
        <path d={path} fill="none" stroke="hsl(var(--primary))" strokeWidth={2} />
        {history.map((h, i) => (
          <circle key={i} cx={toX(i)} cy={toY(losses[i])} r={2} fill="hsl(var(--primary))" />
        ))}
        <text x={pad} y={H - 8} fontSize="10" fill="currentColor" className="text-muted-foreground">step 0</text>
        <text x={W - pad - 24} y={H - 8} fontSize="10" fill="currentColor" className="text-muted-foreground">step {history.length - 1}</text>
      </svg>
    </div>
  );
}
