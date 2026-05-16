/**
 * Electrostatic Field Panel — Phase-3 visualization.
 *
 * Drives `solveElectrostatic` on a Geometry OS bar mesh modelled as a
 * parallel-plate capacitor (electrodes on ±x faces, side faces floating)
 * and renders the resulting electric potential V or field-intensity |E|
 * over a Canvas2D surface projection. Optionally overlays per-tet E-field
 * vectors and RK4 streamlines from the +x electrode toward −x.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { generateMesh, type MeshingResult } from "@/lib/meshing";
import {
  solveElectrostatic, traceFieldLine,
  type ElectrostaticSolution, type SampleE,
} from "@/lib/pde";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type FieldMode = "potential" | "intensity";

interface Params {
  length: number;
  width: number;
  height: number;
  vHigh: number;
  vLow: number;
  epsilonR: number;
  chargeDensity: number;
  minDepth: number;
  maxDepth: number;
}

const EPS0 = 8.854187817e-12;

const DEFAULTS: Params = {
  length: 2, width: 0.4, height: 0.4,
  vHigh: 10, vLow: 0,
  epsilonR: 1, chargeDensity: 0,
  minDepth: 2, maxDepth: 3,
};

// Plasma-like 5-stop ramp (distinct from viridis so the two panels read
// differently at a glance).
const RAMP: Array<[number, [number, number, number]]> = [
  [0.0,  [ 13,   8, 135]],
  [0.25, [126,   3, 168]],
  [0.5,  [204,  71, 120]],
  [0.75, [248, 149,  64]],
  [1.0,  [240, 249,  33]],
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
  result: ElectrostaticSolution;
  electrodeCount: { high: number; low: number };
  elapsedMs: number;
  Vmin: number; Vmax: number;
  Emax: number;
  capacitanceEstimate: number;
}

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
  const tets = mesh.mesh.tets;
  const nV = verts.length / 3;
  const nTets = tets.length / 4;
  const tol = length * 1e-4;

  const dirichlet: { index: number; value: number }[] = [];
  let high = 0, low = 0;
  for (let i = 0; i < nV; i++) {
    const x = verts[i * 3];
    if (x <= tol) { dirichlet.push({ index: i, value: params.vLow }); low++; }
    else if (x >= length - tol) { dirichlet.push({ index: i, value: params.vHigh }); high++; }
  }
  const epsilon = new Float64Array(nTets);
  const epsAbs = params.epsilonR * EPS0;
  for (let t = 0; t < nTets; t++) epsilon[t] = epsAbs;

  const chargeDensity = params.chargeDensity !== 0
    ? new Float64Array(nV).fill(params.chargeDensity)
    : undefined;

  const result = solveElectrostatic({
    mesh: { vertices: verts, tets },
    epsilon,
    chargeDensity,
    dirichlet,
  });

  let Vmin = Infinity, Vmax = -Infinity;
  for (let i = 0; i < result.V.length; i++) {
    if (result.V[i] < Vmin) Vmin = result.V[i];
    if (result.V[i] > Vmax) Vmax = result.V[i];
  }
  let Emax = 0;
  for (let i = 0; i < result.fieldMagnitudeTet.length; i++) {
    if (result.fieldMagnitudeTet[i] > Emax) Emax = result.fieldMagnitudeTet[i];
  }

  // Coarse parallel-plate capacitance estimate C ≈ ε·A/d (sanity).
  const plateArea = width * height;
  const capacitanceEstimate = epsAbs * plateArea / length;

  return {
    mesh, result,
    electrodeCount: { high, low },
    elapsedMs: performance.now() - t0,
    Vmin, Vmax, Emax,
    capacitanceEstimate,
  };
}

interface ViewerProps {
  out: SolveOutput;
  mode: FieldMode;
  showField: boolean;
  showLines: boolean;
  height?: number;
}

function ElectroViewer({ out, mode, showField, showLines, height = 360 }: ViewerProps) {
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
      bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2],
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
    const tetCentroidWorld = new Float64Array(nTets * 3);
    for (let t = 0; t < nTets; t++) {
      let xs = 0, ys = 0, zs = 0, xsw = 0, ysw = 0, zsw = 0;
      for (let k = 0; k < 4; k++) {
        const v = tets[t * 4 + k];
        xs += norm[v * 3]; ys += norm[v * 3 + 1]; zs += norm[v * 3 + 2];
        xsw += verts[v * 3]; ysw += verts[v * 3 + 1]; zsw += verts[v * 3 + 2];
      }
      tetCentroid[t * 3] = xs / 4; tetCentroid[t * 3 + 1] = ys / 4; tetCentroid[t * 3 + 2] = zs / 4;
      tetCentroidWorld[t * 3] = xsw / 4; tetCentroidWorld[t * 3 + 1] = ysw / 4; tetCentroidWorld[t * 3 + 2] = zsw / 4;
    }
    const surface = extractSurfaceTriangles(tets);
    // Normalize world → mesh transform for streamlines.
    const toNorm = (x: number, y: number, z: number): [number, number, number] =>
      [(x - cx) / ext, (y - cy) / ext, (z - cz) / ext];
    return { norm, tetCentroid, tetCentroidWorld, surface, nV, nTets, ext, cx, cy, cz, toNorm };
  }, [out]);

  // Streamlines (world space), computed via a brute-force nearest-tet
  // sampler — bounded to a small seed count so it stays interactive.
  const streamlines = useMemo(() => {
    if (!showLines) return [];
    const bb = out.mesh.mesh.bbox;
    const E = out.result.EPerTet;
    const Emag = out.result.fieldMagnitudeTet;
    const cw = geo.tetCentroidWorld;
    const nT = geo.nTets;
    const sample: SampleE = (x, y, z) => {
      // Find nearest tet centroid.
      let best = -1, bd = Infinity;
      for (let t = 0; t < nT; t++) {
        const dx = cw[t * 3] - x, dy = cw[t * 3 + 1] - y, dz = cw[t * 3 + 2] - z;
        const d = dx * dx + dy * dy + dz * dz;
        if (d < bd) { bd = d; best = t; }
      }
      if (best < 0) return null;
      const m = Emag[best];
      if (m < 1e-20) return null;
      // Return unit-length direction so RK4 stepSize is in world units.
      return [E[best * 3] / m, E[best * 3 + 1] / m, E[best * 3 + 2] / m] as const;
    };
    const ext = Math.max(
      bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2],
    );
    const step = ext * 0.02;
    const maxSteps = 200;
    const seeds: Array<[number, number, number]> = [];
    // Seed a 3×3 grid on the +x electrode face.
    const px = bb.max[0] - ext * 1e-3;
    for (let j = 1; j <= 3; j++) {
      for (let k = 1; k <= 3; k++) {
        seeds.push([
          px,
          bb.min[1] + (j / 4) * (bb.max[1] - bb.min[1]),
          bb.min[2] + (k / 4) * (bb.max[2] - bb.min[2]),
        ]);
      }
    }
    const lines: Float64Array[] = [];
    for (const s of seeds) {
      // direction = -1 means follow −E (high → low V is +E direction; we
      // want to travel high → low, which is along +E for our convention).
      const pts = traceFieldLine(s, sample, {
        stepSize: step, maxSteps, direction: 1,
      });
      lines.push(pts);
    }
    return lines;
  }, [out, geo, showLines]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.clientWidth * devicePixelRatio;
    const H = c.clientHeight * devicePixelRatio;
    c.width = W; c.height = H;
    ctx.fillStyle = "#050505";
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

    const V = out.result.V;
    const Emag = out.result.fieldMagnitude;
    const Vspan = Math.max(1e-12, out.Vmax - out.Vmin);
    const Espan = Math.max(1e-12, (() => {
      let m = 0; for (let i = 0; i < Emag.length; i++) if (Emag[i] > m) m = Emag[i]; return m;
    })());
    const fieldNorm = (i: number) => mode === "potential"
      ? (V[i] - out.Vmin) / Vspan
      : Emag[i] / Espan;

    const px = new Float32Array(geo.nV);
    const py = new Float32Array(geo.nV);
    const pz = new Float32Array(geo.nV);
    for (let i = 0; i < geo.nV; i++) {
      const [x, y, z] = project(geo.norm[i * 3], geo.norm[i * 3 + 1], geo.norm[i * 3 + 2]);
      px[i] = x; py[i] = y; pz[i] = z;
    }
    const tris = geo.surface.map((tri) => {
      const [a, b, c] = tri;
      return { a, b, c, z: (pz[a] + pz[b] + pz[c]) / 3 };
    }).sort((u, v) => u.z - v.z);
    for (const tri of tris) {
      const { a, b, c } = tri;
      const ux = px[b] - px[a], uy = py[b] - py[a];
      const vx = px[c] - px[a], vy = py[c] - py[a];
      if (ux * vy - uy * vx <= 0) continue;
      const fa = fieldNorm(a), fb = fieldNorm(b), fc = fieldNorm(c);
      const f = (fa + fb + fc) / 3;
      const [r, g, bl] = ramp(f);
      ctx.beginPath();
      ctx.moveTo(px[a], py[a]); ctx.lineTo(px[b], py[b]); ctx.lineTo(px[c], py[c]);
      ctx.closePath();
      ctx.fillStyle = `rgb(${r},${g},${bl})`;
      ctx.fill();
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = "rgba(0,0,0,0.25)";
      ctx.stroke();
    }

    if (showField && out.Emax > 0) {
      const E = out.result.EPerTet;
      const Et = out.result.fieldMagnitudeTet;
      const scale = 0.08 / out.Emax;
      ctx.lineWidth = 1;
      const step = Math.max(1, Math.floor(geo.nTets / 600));
      for (let t = 0; t < geo.nTets; t += step) {
        const m = Et[t];
        if (m === 0) continue;
        const cx0 = geo.tetCentroid[t * 3];
        const cy0 = geo.tetCentroid[t * 3 + 1];
        const cz0 = geo.tetCentroid[t * 3 + 2];
        const exw = E[t * 3] * scale / geo.ext;
        const eyw = E[t * 3 + 1] * scale / geo.ext;
        const ezw = E[t * 3 + 2] * scale / geo.ext;
        const [x0, y0] = project(cx0, cy0, cz0);
        const [x1, y1] = project(cx0 + exw, cy0 + eyw, cz0 + ezw);
        ctx.strokeStyle = `rgba(248, 250, 252, ${0.35 + 0.55 * (m / out.Emax)})`;
        ctx.beginPath();
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }

    if (showLines) {
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "rgba(125, 211, 252, 0.85)";
      for (const line of streamlines) {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < line.length; i += 3) {
          const [nx, ny, nz] = geo.toNorm(line[i], line[i + 1], line[i + 2]);
          const [sx, sy2] = project(nx, ny, nz);
          if (!started) { ctx.moveTo(sx, sy2); started = true; }
          else ctx.lineTo(sx, sy2);
        }
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
    const lo = mode === "potential" ? `${out.Vmin.toFixed(2)} V` : "0";
    const hi = mode === "potential" ? `${out.Vmax.toFixed(2)} V` : out.Emax.toExponential(1);
    ctx.fillText(lo, lx, ly - 4 * devicePixelRatio);
    const hiW = ctx.measureText(hi).width;
    ctx.fillText(hi, lx + lw - hiW, ly - 4 * devicePixelRatio);
    ctx.fillText(mode === "potential" ? "V" : "|E| (V/m)", lx, ly + lh + 12 * devicePixelRatio);
  }, [out, geo, yaw, pitch, zoom, mode, showField, showLines, streamlines]);

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

export function ElectrostaticFieldPanel() {
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [out, setOut] = useState<SolveOutput | null>(null);
  const [mode, setMode] = useState<FieldMode>("potential");
  const [showField, setShowField] = useState(true);
  const [showLines, setShowLines] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const run = () => {
    setBusy(true); setErr(null);
    setTimeout(() => {
      try { setOut(runSolve(params)); }
      catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
      finally { setBusy(false); }
    }, 0);
  };

  useEffect(() => { run(); // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const set = <K extends keyof Params>(k: K, v: Params[K]) =>
    setParams((p) => ({ ...p, [k]: v }));

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between gap-3 flex-wrap">
          <div>
            <CardTitle>Electrostatic Field Engine</CardTitle>
            <div className="text-xs text-muted-foreground mt-1">
              −∇·(ε∇V) = ρ on Geometry OS mesh · FEM P1 tets · PCG · RK4 streamlines
            </div>
          </div>
          <div className="flex items-center gap-2">
            {out && (
              <>
                <Badge variant="outline">{out.mesh.summary.tets.count.toLocaleString()} tets</Badge>
                <Badge variant="outline">{out.result.solve.result.iterations} iters</Badge>
                <Badge variant={out.result.solve.result.converged ? "default" : "destructive"}>
                  {out.result.solve.result.converged ? "converged" : "no conv"}
                </Badge>
                <Badge variant="outline">{out.elapsedMs.toFixed(0)} ms</Badge>
              </>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <NumField label="Length (m)" value={params.length} step={0.1} onChange={(v) => set("length", v)} />
          <NumField label="Width (m)" value={params.width} step={0.05} onChange={(v) => set("width", v)} />
          <NumField label="Height (m)" value={params.height} step={0.05} onChange={(v) => set("height", v)} />
          <NumField label="εr (rel. permittivity)" value={params.epsilonR} step={0.5} onChange={(v) => set("epsilonR", v)} />
          <NumField label="V high (V)" value={params.vHigh} step={1} onChange={(v) => set("vHigh", v)} />
          <NumField label="V low (V)" value={params.vLow} step={1} onChange={(v) => set("vLow", v)} />
          <NumField label="ρ (C/m³)" value={params.chargeDensity} step={1e-6} onChange={(v) => set("chargeDensity", v)} />
          <NumField label="max depth" value={params.maxDepth} step={1} onChange={(v) => set("maxDepth", Math.max(params.minDepth, Math.round(v)))} />
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <Button onClick={run} disabled={busy} size="sm">
            {busy ? "Solving…" : "Run solver"}
          </Button>
          <Select value={mode} onValueChange={(v) => setMode(v as FieldMode)}>
            <SelectTrigger className="w-[200px] h-9"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="potential">Potential V</SelectItem>
              <SelectItem value="intensity">Field intensity |E|</SelectItem>
            </SelectContent>
          </Select>
          <Button variant={showField ? "default" : "outline"} size="sm" onClick={() => setShowField((s) => !s)}>
            E-field arrows {showField ? "on" : "off"}
          </Button>
          <Button variant={showLines ? "default" : "outline"} size="sm" onClick={() => setShowLines((s) => !s)}>
            Streamlines {showLines ? "on" : "off"}
          </Button>
        </div>

        {err && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {err}
          </div>
        )}

        {out && (
          <>
            <ElectroViewer out={out} mode={mode} showField={showField} showLines={showLines} />
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
              <Stat label="V min" value={`${out.Vmin.toFixed(3)} V`} />
              <Stat label="V max" value={`${out.Vmax.toFixed(3)} V`} />
              <Stat label="ΔV" value={`${(out.Vmax - out.Vmin).toFixed(3)} V`} />
              <Stat label="|E| max" value={`${out.Emax.toExponential(2)} V/m`} />
              <Stat label="Electrodes" value={`+ ${out.electrodeCount.high} · − ${out.electrodeCount.low}`} />
              <Stat label="vertices" value={out.result.V.length.toLocaleString()} />
              <Stat label="residual" value={out.result.solve.result.residual.toExponential(2)} />
              <Stat label="C ≈ εA/d" value={`${out.capacitanceEstimate.toExponential(2)} F`} />
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
