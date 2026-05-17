/**
 * Potential-Flow Panel — Phase-4 visualization.
 *
 * Solves ∇²φ = 0 on a Geometry OS bar mesh with prescribed inlet/outlet
 * potentials and renders the velocity potential φ or speed |v| over the
 * surface, with optional per-tet velocity arrows and RK4 streamlines
 * seeded on the inlet face. Uses `makeVelocitySampler` to feed the same
 * `traceFieldLine` RK4 integrator that drives the electrostatic engine.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { generateMesh, type MeshingResult } from "@/lib/meshing";
import {
  solvePotentialFlow, makeVelocitySampler, traceFieldLine,
  type PotentialFlowSolution,
} from "@/lib/pde";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

type FieldMode = "potential" | "speed" | "cp" | "pressure";
type FaceKey = "+x" | "-x" | "+y" | "-y" | "+z" | "-z";
type FaceMode = "dirichlet" | "neumann" | "wall";

interface FaceBC {
  mode: FaceMode;
  /** φ value (m²/s) when mode = "dirichlet". */
  phi: number;
  /** Normal velocity v·n_out (m/s) when mode = "neumann". Positive = outflow. */
  vN: number;
}

interface Params {
  length: number;
  width: number;
  height: number;
  density: number;        // ρ (kg/m³) — Bernoulli
  p0: number;             // stagnation / reference pressure (Pa)
  minDepth: number;
  maxDepth: number;
  seedsPerSide: number;
  rk4Steps: number;
  faces: Record<FaceKey, FaceBC>;
  /** Auto-pin a gauge node when no Dirichlet face is selected. */
  pinGauge: boolean;
}

const FACE_KEYS: FaceKey[] = ["-x", "+x", "-y", "+y", "-z", "+z"];

const DEFAULTS: Params = {
  length: 2, width: 0.5, height: 0.5,
  density: 1.225, p0: 101325,
  minDepth: 2, maxDepth: 3,
  seedsPerSide: 4,
  rk4Steps: 240,
  faces: {
    "-x": { mode: "dirichlet", phi: 0, vN: -1 },   // inlet (gauge)
    "+x": { mode: "dirichlet", phi: 2, vN:  1 },   // outlet
    "-y": { mode: "wall",      phi: 0, vN:  0 },
    "+y": { mode: "wall",      phi: 0, vN:  0 },
    "-z": { mode: "wall",      phi: 0, vN:  0 },
    "+z": { mode: "wall",      phi: 0, vN:  0 },
  },
  pinGauge: true,
};

// Viridis-like ramp (distinct from thermal's inferno and electro's plasma).
const RAMP: Array<[number, [number, number, number]]> = [
  [0.0,  [ 68,   1,  84]],
  [0.25, [ 59,  82, 139]],
  [0.5,  [ 33, 144, 141]],
  [0.75, [ 93, 201,  99]],
  [1.0,  [253, 231,  37]],
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

interface FaceSummary {
  face: FaceKey;
  mode: FaceMode;
  area: number;
  nodes: number;
  /** Dirichlet: φ value applied. */
  phi?: number;
  /** Neumann: prescribed v_N (m/s). */
  vN?: number;
  /** Neumann: integrated volumetric flow ∫ v_N dA (m³/s). */
  flow?: number;
}

interface SolveOutput {
  mesh: MeshingResult;
  result: PotentialFlowSolution;
  elapsedMs: number;
  phiMin: number; phiMax: number;
  speedMin: number; speedMax: number;
  cpMin: number; cpMax: number;
  /** Bernoulli pressure p = p₀ − ½ρ|v|² per vertex (Pa). */
  pressure: Float64Array;
  pMin: number; pMax: number;
  density: number; p0: number;
  dirichletCount: number;
  faceSummary: FaceSummary[];
  netFlux: number;
  volumetricFlow: number;
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

function triArea(
  v: Float32Array | Float64Array, a: number, b: number, c: number,
): number {
  const ax = v[a * 3], ay = v[a * 3 + 1], az = v[a * 3 + 2];
  const bx = v[b * 3] - ax, by = v[b * 3 + 1] - ay, bz = v[b * 3 + 2] - az;
  const cx = v[c * 3] - ax, cy = v[c * 3 + 1] - ay, cz = v[c * 3 + 2] - az;
  const nx = by * cz - bz * cy;
  const ny = bz * cx - bx * cz;
  const nz = bx * cy - by * cx;
  return 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
}

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
  const bb = mesh.mesh.bbox;
  const ext = Math.max(
    bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2],
  );
  const tol = ext * 1e-4;

  // Classify each vertex by which face(s) it touches.
  const faceVerts: Record<FaceKey, number[]> = {
    "-x": [], "+x": [], "-y": [], "+y": [], "-z": [], "+z": [],
  };
  for (const f of FACE_KEYS) {
    const test = faceTest(f, bb, tol);
    for (let i = 0; i < nV; i++) {
      if (test(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2])) faceVerts[f].push(i);
    }
  }

  // Dirichlet — collect from face config; later face overrides earlier on shared edge nodes.
  const dirichletMap = new Map<number, number>();
  for (const f of FACE_KEYS) {
    if (params.faces[f].mode !== "dirichlet") continue;
    const v = params.faces[f].phi;
    for (const i of faceVerts[f]) dirichletMap.set(i, v);
  }
  const dirichlet = Array.from(dirichletMap, ([index, value]) => ({ index, value }));

  // Neumann — integrate v_N × area over surface triangles whose 3 verts
  // all lie on a Neumann face. Distribute (v_N · A) equally to 3 nodes.
  const surface = extractSurfaceTriangles(tets);
  const loads = new Float64Array(nV);
  const faceArea: Record<FaceKey, number> = {
    "-x": 0, "+x": 0, "-y": 0, "+y": 0, "-z": 0, "+z": 0,
  };
  let anyNeumann = false;
  for (const f of FACE_KEYS) {
    if (params.faces[f].mode !== "neumann") continue;
    anyNeumann = true;
    const test = faceTest(f, bb, tol);
    const vN = params.faces[f].vN;
    for (const [a, b, c] of surface) {
      const pass = (i: number) =>
        test(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
      if (pass(a) && pass(b) && pass(c)) {
        const A = triArea(verts as Float32Array, a, b, c);
        faceArea[f] += A;
        const share = (vN * A) / 3;
        loads[a] += share; loads[b] += share; loads[c] += share;
      }
    }
  }

  const result = solvePotentialFlow({
    mesh: { vertices: verts, tets },
    dirichlet: dirichlet.length > 0 ? dirichlet : undefined,
    neumannLoads: anyNeumann ? loads : undefined,
    pinGauge: dirichlet.length === 0 && params.pinGauge,
  });

  let phiMin = Infinity, phiMax = -Infinity;
  for (let i = 0; i < result.phi.length; i++) {
    if (result.phi[i] < phiMin) phiMin = result.phi[i];
    if (result.phi[i] > phiMax) phiMax = result.phi[i];
  }
  let speedMin = Infinity, speedMax = 0;
  for (let i = 0; i < result.speed.length; i++) {
    const s = result.speed[i];
    if (s < speedMin) speedMin = s;
    if (s > speedMax) speedMax = s;
  }
  let cpMin = Infinity, cpMax = -Infinity;
  for (let i = 0; i < result.cp.length; i++) {
    if (result.cp[i] < cpMin) cpMin = result.cp[i];
    if (result.cp[i] > cpMax) cpMax = result.cp[i];
  }

  // Domain-mean axial velocity × cross-section (simple incompressibility check).
  let uxSum = 0, uxN = 0;
  for (let t = 0; t < result.velocityPerTet.length / 3; t++) {
    uxSum += result.velocityPerTet[t * 3]; uxN++;
  }
  const meanUx = uxN > 0 ? uxSum / uxN : 0;
  const volumetricFlow = meanUx * width * height;

  // Bernoulli (steady, incompressible, irrotational): p = p₀ − ½ρ|v|².
  const nVerts = result.speed.length;
  const pressure = new Float64Array(nVerts);
  const half = 0.5 * params.density;
  let pMin = Infinity, pMax = -Infinity;
  for (let i = 0; i < nVerts; i++) {
    const p = params.p0 - half * result.speed[i] * result.speed[i];
    pressure[i] = p;
    if (p < pMin) pMin = p;
    if (p > pMax) pMax = p;
  }

  // Per-face area for Dirichlet faces too (for summary).
  for (const f of FACE_KEYS) {
    if (faceArea[f] > 0 || params.faces[f].mode === "wall") continue;
    const test = faceTest(f, bb, tol);
    for (const [a, b, c] of surface) {
      const pass = (i: number) =>
        test(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
      if (pass(a) && pass(b) && pass(c)) {
        faceArea[f] += triArea(verts as Float32Array, a, b, c);
      }
    }
  }

  const faceSummary: FaceSummary[] = FACE_KEYS.map((f) => {
    const bc = params.faces[f];
    const s: FaceSummary = {
      face: f, mode: bc.mode, area: faceArea[f], nodes: faceVerts[f].length,
    };
    if (bc.mode === "dirichlet") s.phi = bc.phi;
    if (bc.mode === "neumann") { s.vN = bc.vN; s.flow = bc.vN * faceArea[f]; }
    return s;
  });
  const netFlux = faceSummary.reduce((s, x) => s + (x.flow ?? 0), 0);

  return {
    mesh, result,
    elapsedMs: performance.now() - t0,
    phiMin, phiMax, speedMin, speedMax, cpMin, cpMax,
    pressure, pMin, pMax, density: params.density, p0: params.p0,
    dirichletCount: dirichlet.length,
    faceSummary, netFlux, volumetricFlow,
  };
}

interface ViewerProps {
  out: SolveOutput;
  mode: FieldMode;
  showVectors: boolean;
  showStreamlines: boolean;
  seedsPerSide: number;
  rk4Steps: number;
  height?: number;
}

function FlowViewer({
  out, mode, showVectors, showStreamlines, seedsPerSide, rk4Steps, height = 380,
}: ViewerProps) {
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
    const tetCentroidNorm = new Float32Array(nTets * 3);
    for (let t = 0; t < nTets; t++) {
      let xs = 0, ys = 0, zs = 0;
      for (let k = 0; k < 4; k++) {
        const v = tets[t * 4 + k];
        xs += norm[v * 3]; ys += norm[v * 3 + 1]; zs += norm[v * 3 + 2];
      }
      tetCentroidNorm[t * 3] = xs / 4;
      tetCentroidNorm[t * 3 + 1] = ys / 4;
      tetCentroidNorm[t * 3 + 2] = zs / 4;
    }
    const surface = extractSurfaceTriangles(tets);
    const toNorm = (x: number, y: number, z: number): [number, number, number] =>
      [(x - cx) / ext, (y - cy) / ext, (z - cz) / ext];
    return { norm, tetCentroidNorm, surface, nV, nTets, ext, bb, toNorm };
  }, [out]);

  // RK4 streamlines through the mesh-backed velocity sampler.
  const streamlines = useMemo(() => {
    if (!showStreamlines) return [] as Float64Array[];
    const meshIn = { vertices: out.mesh.mesh.vertices, tets: out.mesh.mesh.tets };
    const sampleV = makeVelocitySampler(meshIn, out.result.velocityPerTet);
    // Normalize sampled velocity → unit direction so stepSize stays in world units.
    const sample = (x: number, y: number, z: number) => {
      const v = sampleV(x, y, z);
      if (!v) return null;
      const m = Math.hypot(v[0], v[1], v[2]);
      if (m < 1e-20) return null;
      return [v[0] / m, v[1] / m, v[2] / m] as const;
    };
    const bb = out.mesh.mesh.bbox;
    const ext = Math.max(
      bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2],
    );
    const step = ext * 0.015;
    const seeds: Array<[number, number, number]> = [];
    const inletX = bb.min[0] + ext * 1e-3;
    const n = Math.max(1, seedsPerSide);
    for (let j = 1; j <= n; j++) {
      for (let k = 1; k <= n; k++) {
        seeds.push([
          inletX,
          bb.min[1] + (j / (n + 1)) * (bb.max[1] - bb.min[1]),
          bb.min[2] + (k / (n + 1)) * (bb.max[2] - bb.min[2]),
        ]);
      }
    }
    const lines: Float64Array[] = [];
    for (const s of seeds) {
      const pts = traceFieldLine(s, sample, {
        stepSize: step, maxSteps: rk4Steps, direction: 1,
      });
      lines.push(pts);
    }
    return lines;
  }, [out, showStreamlines, seedsPerSide, rk4Steps]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const W = c.clientWidth * devicePixelRatio;
    const H = c.clientHeight * devicePixelRatio;
    c.width = W; c.height = H;
    ctx.fillStyle = "#040608";
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

    const phi = out.result.phi;
    const sp2 = out.result.speed;
    const cpv = out.result.cp;
    const prs = out.pressure;
    const phiSpan = Math.max(1e-12, out.phiMax - out.phiMin);
    const spSpan = Math.max(1e-12, out.speedMax - out.speedMin);
    const cpSpan = Math.max(1e-12, out.cpMax - out.cpMin);
    const pSpan = Math.max(1e-12, out.pMax - out.pMin);
    const fieldNorm = (i: number) => {
      if (mode === "potential") return (phi[i] - out.phiMin) / phiSpan;
      if (mode === "speed")     return (sp2[i] - out.speedMin) / spSpan;
      if (mode === "pressure")  return (prs[i] - out.pMin) / pSpan;
      return (cpv[i] - out.cpMin) / cpSpan;
    };

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
      ctx.fillStyle = `rgba(${r},${g},${bl},0.92)`;
      ctx.fill();
      ctx.lineWidth = 0.5;
      ctx.strokeStyle = "rgba(0,0,0,0.3)";
      ctx.stroke();
    }

    if (showVectors && out.speedMax > 0) {
      const V = out.result.velocityPerTet;
      const Sp = out.result.speedTet;
      const scale = 0.09 / out.speedMax;
      ctx.lineWidth = 1;
      const step = Math.max(1, Math.floor(geo.nTets / 600));
      for (let t = 0; t < geo.nTets; t += step) {
        const m = Sp[t];
        if (m === 0) continue;
        const cx0 = geo.tetCentroidNorm[t * 3];
        const cy0 = geo.tetCentroidNorm[t * 3 + 1];
        const cz0 = geo.tetCentroidNorm[t * 3 + 2];
        const vxw = V[t * 3] * scale / geo.ext;
        const vyw = V[t * 3 + 1] * scale / geo.ext;
        const vzw = V[t * 3 + 2] * scale / geo.ext;
        const [x0, y0] = project(cx0, cy0, cz0);
        const [x1, y1] = project(cx0 + vxw, cy0 + vyw, cz0 + vzw);
        const alpha = 0.35 + 0.55 * (m / out.speedMax);
        ctx.strokeStyle = `rgba(226, 232, 240, ${alpha})`;
        ctx.beginPath();
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        ctx.stroke();
      }
    }

    if (showStreamlines) {
      ctx.lineWidth = 1.6;
      for (const line of streamlines) {
        ctx.beginPath();
        let started = false;
        for (let i = 0; i < line.length; i += 3) {
          const [nx, ny, nz] = geo.toNorm(line[i], line[i + 1], line[i + 2]);
          const [sx, sy2] = project(nx, ny, nz);
          if (!started) { ctx.moveTo(sx, sy2); started = true; }
          else ctx.lineTo(sx, sy2);
        }
        ctx.strokeStyle = "rgba(165, 243, 252, 0.92)";
        ctx.stroke();
        // Seed dot.
        if (line.length >= 3) {
          const [nx, ny, nz] = geo.toNorm(line[0], line[1], line[2]);
          const [sx, sy2] = project(nx, ny, nz);
          ctx.fillStyle = "rgba(125, 211, 252, 1)";
          ctx.beginPath();
          ctx.arc(sx, sy2, 2.5 * devicePixelRatio, 0, Math.PI * 2);
          ctx.fill();
        }
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
    let lo = "", hi = "", unit = "";
    if (mode === "potential") {
      lo = out.phiMin.toFixed(3); hi = out.phiMax.toFixed(3); unit = "φ (m²/s)";
    } else if (mode === "speed") {
      lo = out.speedMin.toFixed(3); hi = out.speedMax.toFixed(3); unit = "|v| (m/s)";
    } else if (mode === "pressure") {
      lo = out.pMin.toExponential(2); hi = out.pMax.toExponential(2); unit = "p (Pa)";
    } else {
      lo = out.cpMin.toFixed(3); hi = out.cpMax.toFixed(3); unit = "Cp";
    }
    ctx.fillText(lo, lx, ly - 4 * devicePixelRatio);
    const hiW = ctx.measureText(hi).width;
    ctx.fillText(hi, lx + lw - hiW, ly - 4 * devicePixelRatio);
    ctx.fillText(unit, lx, ly + lh + 12 * devicePixelRatio);
  }, [out, geo, yaw, pitch, zoom, mode, showVectors, showStreamlines, streamlines]);

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

export function PotentialFlowPanel() {
  const [params, setParams] = useState<Params>(DEFAULTS);
  const [out, setOut] = useState<SolveOutput | null>(null);
  const [mode, setMode] = useState<FieldMode>("speed");
  const [showVectors, setShowVectors] = useState(false);
  const [showStreamlines, setShowStreamlines] = useState(true);
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
            <CardTitle>Potential-Flow Engine</CardTitle>
            <div className="text-xs text-muted-foreground mt-1">
              ∇²φ = 0 · v = ∇φ on Geometry OS mesh · FEM P1 tets · RK4 streamlines
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
          <NumField label="max depth" value={params.maxDepth} step={1} onChange={(v) => set("maxDepth", Math.max(params.minDepth, Math.round(v)))} />
          <NumField label="ρ (kg/m³)" value={params.density} step={0.1} onChange={(v) => set("density", Math.max(1e-9, v))} />
          <NumField label="p₀ (Pa)" value={params.p0} step={100} onChange={(v) => set("p0", v)} />
          <NumField label="seeds / side" value={params.seedsPerSide} step={1} onChange={(v) => set("seedsPerSide", Math.max(1, Math.min(8, Math.round(v))))} />
          <NumField label="RK4 steps" value={params.rk4Steps} step={20} onChange={(v) => set("rk4Steps", Math.max(20, Math.round(v)))} />
        </div>

        <BoundaryEditor
          faces={params.faces}
          pinGauge={params.pinGauge}
          onFaceChange={(face, patch) => setParams((p) => ({
            ...p,
            faces: { ...p.faces, [face]: { ...p.faces[face], ...patch } },
          }))}
          onPinChange={(v) => set("pinGauge", v)}
        />

        <div className="flex items-center gap-2 flex-wrap">
          <Button onClick={run} disabled={busy} size="sm">
            {busy ? "Solving…" : "Run solver"}
          </Button>
          <Select value={mode} onValueChange={(v) => setMode(v as FieldMode)}>
            <SelectTrigger className="w-[220px] h-9"><SelectValue /></SelectTrigger>
            <SelectContent>
              <SelectItem value="potential">Velocity potential φ</SelectItem>
              <SelectItem value="speed">Speed |v|</SelectItem>
              <SelectItem value="cp">Pressure coefficient Cp</SelectItem>
              <SelectItem value="pressure">Bernoulli pressure p</SelectItem>
            </SelectContent>
          </Select>
          <Button variant={showVectors ? "default" : "outline"} size="sm" onClick={() => setShowVectors((s) => !s)}>
            v arrows {showVectors ? "on" : "off"}
          </Button>
          <Button variant={showStreamlines ? "default" : "outline"} size="sm" onClick={() => setShowStreamlines((s) => !s)}>
            Streamlines {showStreamlines ? "on" : "off"}
          </Button>
        </div>

        {err && (
          <div className="rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {err}
          </div>
        )}

        {out && (
          <>
            <FlowViewer
              out={out} mode={mode}
              showVectors={showVectors}
              showStreamlines={showStreamlines}
              seedsPerSide={params.seedsPerSide}
              rk4Steps={params.rk4Steps}
            />
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
              <Stat label="φ min" value={out.phiMin.toFixed(3)} />
              <Stat label="φ max" value={out.phiMax.toFixed(3)} />
              <Stat label="|v| max" value={`${out.speedMax.toFixed(3)} m/s`} />
              <Stat label="Cp range" value={`${out.cpMin.toFixed(2)} … ${out.cpMax.toFixed(2)}`} />
              <Stat label="V_ref" value={`${out.result.referenceSpeed.toFixed(3)} m/s`} />
              <Stat label="Dirichlet nodes" value={`${out.dirichletCount}`} />
              <Stat label="vertices" value={out.result.phi.length.toLocaleString()} />
              <Stat label="residual" value={out.result.solve.result.residual.toExponential(2)} />
              <Stat label="Q ≈ ūx·A" value={`${out.volumetricFlow.toExponential(2)} m³/s`} />
              <Stat label="p min" value={`${out.pMin.toExponential(3)} Pa`} />
              <Stat label="p max" value={`${out.pMax.toExponential(3)} Pa`} />
              <Stat label="Δp = ½ρ|v|²max" value={`${(0.5 * out.density * out.speedMax * out.speedMax).toExponential(2)} Pa`} />
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
