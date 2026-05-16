/**
 * Thermal Field Panel — runs the Phase-2 thermal engine on an in-panel
 * mesh and visualizes the resulting temperature field T, hotspot map,
 * and per-tet heat-flux vectors over a Canvas2D projection.
 *
 * Self-contained: generates a bar mesh via Geometry OS, pins Dirichlet
 * temperatures on the ±X end-caps, calls `solveThermal`, and renders
 * vertex colors (viridis ramp) plus optional flux arrows.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { generateMesh, type MeshingResult } from "@/lib/meshing";
import { solveThermal, type ThermalSolution } from "@/lib/pde";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type FieldMode = "temperature" | "hotspot";

interface Params {
  length: number;
  width: number;
  height: number;
  hotT: number;
  coldT: number;
  kappa: number;
  minDepth: number;
  maxDepth: number;
}

const DEFAULTS: Params = {
  length: 2, width: 0.4, height: 0.4,
  hotT: 400, coldT: 300, kappa: 45,
  minDepth: 2, maxDepth: 3,
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
  dirichletCount: { hot: number; cold: number };
  elapsedMs: number;
  Tmin: number; Tmax: number;
  fluxMax: number;
}

function runSolve(params: Params): SolveOutput {
  const t0 = performance.now();
  const { length, width, height } = params;
  const mesh = generateMesh({
    bbox: { min: [0, 0, 0], max: [length, width, height] },
    seeds: [],
    octree: { minDepth: params.minDepth, maxDepth: params.maxDepth },
    partitionCount: 1,
  });
  const verts = mesh.mesh.vertices;
  const nV = verts.length / 3;
  const nTets = mesh.mesh.tets.length / 4;
  // Identify Dirichlet vertices at x≈0 (cold) and x≈length (hot).
  const tol = length * 1e-4;
  const dirichlet: { index: number; value: number }[] = [];
  let hot = 0, cold = 0;
  for (let i = 0; i < nV; i++) {
    const x = verts[i * 3];
    if (x <= tol) { dirichlet.push({ index: i, value: params.coldT }); cold++; }
    else if (x >= length - tol) { dirichlet.push({ index: i, value: params.hotT }); hot++; }
  }
  const kappa = new Float64Array(nTets);
  for (let t = 0; t < nTets; t++) kappa[t] = params.kappa;

  const thermal = solveThermal({
    mesh: {
      vertices: verts,
      tets: mesh.mesh.tets,
    },
    kappa,
    dirichlet,
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
    mesh, thermal,
    dirichletCount: { hot, cold },
    elapsedMs: performance.now() - t0,
    Tmin, Tmax, fluxMax,
  };
}

interface ViewerProps {
  out: SolveOutput;
  mode: FieldMode;
  showFlux: boolean;
  height?: number;
}

function ThermalViewer({ out, mode, showFlux, height = 360 }: ViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [yaw, setYaw] = useState(0.7);
  const [pitch, setPitch] = useState(-0.35);
  const [zoom, setZoom] = useState(1);
  const drag = useRef<{ x: number; y: number; yaw: number; pitch: number } | null>(null);

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
    const field = mode === "temperature" ? T : hot;
    const Tspan = Math.max(1e-12, out.Tmax - out.Tmin);
    const fieldNorm = (i: number) => mode === "temperature"
      ? (T[i] - out.Tmin) / Tspan
      : hot[i];

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
    const lo = mode === "temperature" ? `${out.Tmin.toFixed(1)} K` : "0";
    const hi = mode === "temperature" ? `${out.Tmax.toFixed(1)} K` : "1";
    ctx.fillText(lo, lx, ly - 4 * devicePixelRatio);
    const hiW = ctx.measureText(hi).width;
    ctx.fillText(hi, lx + lw - hiW, ly - 4 * devicePixelRatio);
    ctx.fillText(mode === "temperature" ? "T" : "hotspot", lx, ly + lh + 12 * devicePixelRatio);
  }, [out, geo, yaw, pitch, zoom, mode, showFlux]);

  return (
    <div
      className="relative w-full overflow-hidden rounded-md border border-border bg-black"
      style={{ height }}
      onMouseDown={(e) => { drag.current = { x: e.clientX, y: e.clientY, yaw, pitch }; }}
      onMouseMove={(e) => {
        if (!drag.current) return;
        const dx = e.clientX - drag.current.x;
        const dy = e.clientY - drag.current.y;
        setYaw(drag.current.yaw + dx * 0.01);
        setPitch(drag.current.pitch + dy * 0.01);
      }}
      onMouseUp={() => { drag.current = null; }}
      onMouseLeave={() => { drag.current = null; }}
      onWheel={(e) => {
        setZoom((z) => Math.max(0.3, Math.min(4, z * (e.deltaY < 0 ? 1.1 : 0.9))));
      }}
    >
      <canvas ref={canvasRef} className="h-full w-full cursor-grab active:cursor-grabbing" />
    </div>
  );
}

export function ThermalFieldPanel() {
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [out, setOut] = useState<SolveOutput | null>(null);
  const [mode, setMode] = useState<FieldMode>("temperature");
  const [showFlux, setShowFlux] = useState(true);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = () => {
    setBusy(true); setErr(null);
    // Defer to next tick so the busy state paints.
    setTimeout(() => {
      try {
        setOut(runSolve(params));
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    }, 0);
  };

  useEffect(() => { run(); /* initial */ // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const set = <K extends keyof Params>(k: K, v: Params[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

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
                <Badge variant="outline">{out.thermal.solve.iterations} iters</Badge>
                <Badge variant={out.thermal.solve.converged ? "default" : "destructive"}>
                  {out.thermal.solve.converged ? "converged" : "no conv"}
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
          <NumField label="Hot face T (K)" value={params.hotT} step={5} onChange={(v) => set("hotT", v)} />
          <NumField label="Cold face T (K)" value={params.coldT} step={5} onChange={(v) => set("coldT", v)} />
          <NumField label="min depth" value={params.minDepth} step={1} onChange={(v) => set("minDepth", Math.max(1, Math.round(v)))} />
          <NumField label="max depth" value={params.maxDepth} step={1} onChange={(v) => set("maxDepth", Math.max(params.minDepth, Math.round(v)))} />
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <Button onClick={run} disabled={busy} size="sm">
            {busy ? "Solving…" : "Run solver"}
          </Button>
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
            <ThermalViewer out={out} mode={mode} showFlux={showFlux} />
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
              <Stat label="T min" value={`${out.Tmin.toFixed(2)} K`} />
              <Stat label="T max" value={`${out.Tmax.toFixed(2)} K`} />
              <Stat label="ΔT" value={`${(out.Tmax - out.Tmin).toFixed(2)} K`} />
              <Stat label="|q| max" value={out.fluxMax.toExponential(2)} />
              <Stat label="Dirichlet" value={`hot ${out.dirichletCount.hot} · cold ${out.dirichletCount.cold}`} />
              <Stat label="vertices" value={out.thermal.T.length.toLocaleString()} />
              <Stat label="residual" value={out.thermal.solve.residual.toExponential(2)} />
              <Stat label="solver" value={`${out.thermal.solve.iterations} PCG iters`} />
            </div>
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
