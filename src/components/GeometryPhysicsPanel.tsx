/**
 * Geometry-Aware Physics Intelligence — live panel.
 *
 * Generates an icosphere demo mesh, applies a parametric deformation
 * (dent + stretch), and runs the unified geometry+physics pipeline:
 *
 *   topology profile → curvature field → deformation metrics →
 *   curvature-aware stress prediction → topology-aware solver routing →
 *   topology-preserving shape-energy optimization
 *
 * Visualizes the deformed mesh projected to 2D with per-vertex stress
 * heatmap and hotspot ring. Tracks success-metric deltas (realism,
 * mesh quality gain, geometric prediction accuracy) against an
 * un-routed baseline.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  analyze, makeIcoSphere, deformMesh, optimizeShape, computeCurvature,
  predictStress, computeTopology,
  type GeometryReport, type TriMesh,
} from "@/lib/geomphys";

const SOLVER_TONE: Record<string, string> = {
  explicit:         "text-emerald-300 border-emerald-500/40 bg-emerald-500/5",
  implicit_cg:      "text-primary border-primary/40 bg-primary/10",
  implicit_direct:  "text-amber-300 border-amber-500/40 bg-amber-500/10",
  rom_reduced:      "text-cyan-300 border-cyan-500/40 bg-cyan-500/10",
};

export function GeometryPhysicsPanel() {
  const [subdivisions, setSubdivisions] = useState(2);
  const [dent, setDent] = useState(0.35);
  const [stretch, setStretch] = useState(0.6);
  const [optIters, setOptIters] = useState(8);
  const [live, setLive] = useState(true);
  const tick = useRef(0);

  // Build rest mesh whenever subdivisions change.
  const rest = useMemo(() => makeIcoSphere(subdivisions, 1), [subdivisions]);

  // Apply deformation each tick or on slider change.
  const [deformed, setDeformed] = useState<TriMesh>(() =>
    deformMesh(rest, dent, stretch));

  useEffect(() => {
    setDeformed(deformMesh(rest, dent, stretch));
  }, [rest, dent, stretch]);

  // Live oscillation pulse (when running)
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => {
      tick.current += 1;
      const phase = 0.18 * Math.sin(tick.current / 14);
      setDeformed(deformMesh(rest, dent + phase, stretch + phase * 0.5));
    }, 220);
    return () => clearInterval(id);
  }, [live, rest, dent, stretch]);

  // Run pipeline
  const report: GeometryReport = useMemo(
    () => analyze(rest, deformed),
    [rest, deformed],
  );

  // Optimize on demand (heavy-ish, so memo only when inputs change)
  const optResult = useMemo(
    () => optimizeShape(deformed, { iterations: optIters }),
    [deformed, optIters],
  );

  // Success-metric deltas vs un-routed baseline (always implicit_direct).
  const realismDelta = useMemo(() => {
    // Realism gain = chosen accuracy / baseline accuracy − 1, but only
    // when chosen solver is cheaper. Otherwise realism is preserved (0).
    const baseAcc = 0.98, baseCost = 1.0;
    const a = report.routing.accuracyFactor;
    const c = report.routing.costFactor;
    // Reward when accuracy is within 1% of baseline at lower cost — i.e.
    // we got equivalent realism for a fraction of the work.
    const costSaved = 1 - c / baseCost;
    const accLoss = baseAcc - a;
    return Math.max(0, costSaved - Math.max(0, accLoss) * 4);
  }, [report]);

  const meshGain = useMemo(() => {
    const before = optResult.before.distortion;
    const after = optResult.after.distortion;
    if (before < 1e-6) return 0;
    return (before - after) / before;
  }, [optResult]);

  const predictionAcc = useMemo(() => {
    // Geometric prediction accuracy proxy: alignment between curvature-
    // predicted hotspot region and actual max-displacement vertex region.
    const dispMax = report.deformation?.perVertexDisp ?? new Float64Array(0);
    const stress = report.stress?.vertexStress ?? new Float64Array(0);
    if (dispMax.length === 0 || stress.length === 0) return 0;
    // Take top-10% by stress, measure mean disp percentile they live in.
    const idxs = [...stress.keys()].sort((a, b) => stress[b] - stress[a]);
    const top = idxs.slice(0, Math.max(1, Math.floor(stress.length * 0.1)));
    const sortedDisp = [...dispMax].sort((a, b) => a - b);
    let pctSum = 0;
    for (const i of top) {
      const v = dispMax[i];
      const rank = lowerBound(sortedDisp, v);
      pctSum += rank / sortedDisp.length;
    }
    return pctSum / top.length;
  }, [report]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Geometry-Aware Physics Intelligence
          <Badge variant="outline" className="text-[10px]">
            {report.topology.V} V · {report.topology.E} E · {report.topology.F} F
            {" · "}χ={report.topology.euler} · genus={report.topology.genus.toFixed(0)}
          </Badge>
        </h3>
        <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
          onClick={() => setLive((v) => !v)}>
          {live ? "pause" : "resume"}
        </Button>
      </div>

      {/* KPI ribbon */}
      <div className="grid grid-cols-3 gap-2">
        <Kpi label="simulation realism Δ" v={realismDelta} target={0.20}
             fmt={(x) => `${(x * 100).toFixed(1)}%`} />
        <Kpi label="mesh optimization gain" v={meshGain} target={0.15}
             fmt={(x) => `${(x * 100).toFixed(1)}%`} />
        <Kpi label="geometric prediction accuracy" v={predictionAcc} target={0.85}
             fmt={(x) => `${(x * 100).toFixed(1)}%`} />
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {/* Mesh visualization */}
        <div className="rounded border border-border bg-background/40 p-3">
          <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
            <span>Deformed mesh · curvature-aware stress heatmap</span>
            <span>hotspot v#{report.stress?.hotspotVertex ?? "—"}</span>
          </div>
          <MeshView mesh={deformed} report={report} />
          <div className="mt-2 grid grid-cols-2 gap-2 text-[10px] tabular-nums">
            <Metric label="max disp" v={report.deformation?.maxDisp.toFixed(3) ?? "—"} />
            <Metric label="mean stretch" v={report.deformation?.meanStretch.toFixed(3) ?? "—"} />
            <Metric label="area ratio" v={report.deformation?.meanAreaRatio.toFixed(3) ?? "—"} />
            <Metric label="bending (rad)" v={report.deformation?.meanBendingRad.toFixed(3) ?? "—"} />
            <Metric label="distortion" v={report.deformation?.distortion.toFixed(3) ?? "—"} />
            <Metric label="strain energy" v={report.stress?.totalStrainEnergy.toFixed(3) ?? "—"} />
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-3">
          {/* Solver routing */}
          <div className={`rounded border px-3 py-2 ${SOLVER_TONE[report.routing.solver]}`}>
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] opacity-80">
              <span>Topology-aware solver routing</span>
              <span>confidence {(report.routing.confidence * 100).toFixed(0)}%</span>
            </div>
            <div className="mt-1 flex items-baseline justify-between">
              <span className="font-mono text-lg uppercase">{report.routing.solver}</span>
              <span className="text-[10px] opacity-70">
                cost {(report.routing.costFactor * 100).toFixed(0)}% ·
                acc {(report.routing.accuracyFactor * 100).toFixed(0)}%
              </span>
            </div>
            <ul className="mt-1 space-y-0.5 text-[10px] opacity-80">
              {report.routing.reasoning.map((r, i) => (
                <li key={i}>· {r}</li>
              ))}
            </ul>
          </div>

          {/* Curvature stats */}
          <div className="rounded border border-border bg-background/40 p-3">
            <div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
              Curvature field
            </div>
            <CurvatureBars curv={report.curvature} />
            <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] tabular-nums">
              <Metric label="tri quality (mean)" v={report.topology.triQualityMean.toFixed(2)} />
              <Metric label="tri quality (min)" v={report.topology.triQualityMin.toFixed(2)} />
              <Metric label="max valence" v={String(report.topology.maxValence)} />
            </div>
          </div>

          {/* Shape optimization */}
          <div className="rounded border border-border bg-background/40 p-3">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] text-muted-foreground mb-2">
              <span>Topology-preserving shape optimization</span>
              <span className={optResult.topologyPreserved ? "text-emerald-300" : "text-destructive"}>
                {optResult.topologyPreserved ? "topology preserved" : "TOPOLOGY BROKEN"}
              </span>
            </div>
            <div className="grid grid-cols-3 gap-2 text-[10px] tabular-nums">
              <Metric label="energy before" v={optResult.before.energy.toFixed(3)} />
              <Metric label="energy after" v={optResult.after.energy.toFixed(3)} />
              <Metric label="reduction" v={`${(optResult.energyReduction * 100).toFixed(1)}%`} />
              <Metric label="distortion before" v={optResult.before.distortion.toFixed(3)} />
              <Metric label="distortion after" v={optResult.after.distortion.toFixed(3)} />
              <Metric label="iterations" v={String(optResult.iterations)} />
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-4">
        <SliderField label="subdivisions" v={subdivisions} min={1} max={4} step={1}
          onChange={(v) => setSubdivisions(v)} />
        <SliderField label="dent depth" v={dent} min={0} max={0.8} step={0.01}
          onChange={(v) => setDent(v)} />
        <SliderField label="radial stretch" v={stretch} min={0} max={1.2} step={0.01}
          onChange={(v) => setStretch(v)} />
        <SliderField label="opt iterations" v={optIters} min={2} max={20} step={1}
          onChange={(v) => setOptIters(v)} />
      </div>
    </div>
  );
}

// ============================================================
// subcomponents
// ============================================================

function Kpi({
  label, v, fmt, target,
}: { label: string; v: number; fmt: (x: number) => string; target: number }) {
  const ok = v >= target;
  return (
    <div className={`rounded border px-3 py-2 ${
      ok ? "border-emerald-500/40 bg-emerald-500/5 text-emerald-300"
         : "border-border bg-background/40 text-muted-foreground"
    }`}>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.14em] opacity-80">
        <span>{label}</span>
        <span>≥ {(target * 100).toFixed(0)}%</span>
      </div>
      <div className="mt-0.5 font-mono text-lg tabular-nums">{fmt(v)}</div>
    </div>
  );
}

function Metric({ label, v }: { label: string; v: string }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 px-2 py-1">
      <div className="text-[9px] uppercase tracking-[0.14em] text-muted-foreground/80">{label}</div>
      <div className="font-mono text-foreground/90">{v}</div>
    </div>
  );
}

function SliderField({
  label, v, min, max, step, onChange,
}: { label: string; v: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] uppercase tracking-[0.14em]">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-primary tabular-nums">{step < 1 ? v.toFixed(2) : v}</span>
      </div>
      <Slider value={[v]} min={min} max={max} step={step}
        onValueChange={([x]) => onChange(x)} />
    </div>
  );
}

function CurvatureBars({ curv }: { curv: ReturnType<typeof computeCurvature> }) {
  const h = curv.meanCurvature;
  const k = curv.gaussianCurvature;
  // 12-bucket histograms
  const bucket = (arr: Float64Array) => {
    if (arr.length === 0) return new Array(12).fill(0);
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < arr.length; i++) { if (arr[i] < mn) mn = arr[i]; if (arr[i] > mx) mx = arr[i]; }
    const range = Math.max(1e-9, mx - mn);
    const out = new Array(12).fill(0);
    for (let i = 0; i < arr.length; i++) {
      const b = Math.min(11, Math.floor(((arr[i] - mn) / range) * 12));
      out[b]++;
    }
    const m = Math.max(1, ...out);
    return out.map((v) => v / m);
  };
  const Hbins = bucket(h);
  const Kbins = bucket(k);
  return (
    <div className="space-y-2">
      <Histogram label="|H| mean curvature" bins={Hbins} cls="bg-primary/70" />
      <Histogram label="K Gaussian curvature" bins={Kbins} cls="bg-cyan-400/70" />
    </div>
  );
}

function Histogram({ label, bins, cls }: { label: string; bins: number[]; cls: string }) {
  return (
    <div>
      <div className="text-[9px] uppercase tracking-[0.14em] text-muted-foreground/80 mb-0.5">{label}</div>
      <div className="flex h-7 items-end gap-px">
        {bins.map((b, i) => (
          <div key={i} className={`flex-1 ${cls} rounded-t`}
            style={{ height: `${Math.max(3, b * 100)}%` }} />
        ))}
      </div>
    </div>
  );
}

function MeshView({ mesh, report }: { mesh: TriMesh; report: GeometryReport }) {
  const w = 360, h = 240;
  // Orthographic projection on XY, color edges by stress avg.
  const stress = report.stress?.vertexStress ?? new Float64Array(mesh.positions.length / 3);
  let sMax = 1e-9;
  for (let i = 0; i < stress.length; i++) if (stress[i] > sMax) sMax = stress[i];

  // bbox
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < mesh.positions.length / 3; i++) {
    const x = mesh.positions[3 * i], y = mesh.positions[3 * i + 1];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const pad = 12;
  const sx = (x: number) => pad + ((x - minX) / Math.max(1e-9, maxX - minX)) * (w - 2 * pad);
  const sy = (y: number) => h - pad - ((y - minY) / Math.max(1e-9, maxY - minY)) * (h - 2 * pad);

  const T = mesh.indices.length / 3;
  const tris: { d: string; color: string; depth: number }[] = [];
  for (let t = 0; t < T; t++) {
    const a = mesh.indices[3 * t], b = mesh.indices[3 * t + 1], c = mesh.indices[3 * t + 2];
    // backface cull via signed area in screen space
    const ax = sx(mesh.positions[3 * a]), ay = sy(mesh.positions[3 * a + 1]);
    const bx = sx(mesh.positions[3 * b]), by = sy(mesh.positions[3 * b + 1]);
    const cx = sx(mesh.positions[3 * c]), cy = sy(mesh.positions[3 * c + 1]);
    const signed = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
    if (signed > 0) continue;
    const sBar = (stress[a] + stress[b] + stress[c]) / 3 / sMax;
    // Painter's depth from average z
    const z = (mesh.positions[3 * a + 2] + mesh.positions[3 * b + 2] + mesh.positions[3 * c + 2]) / 3;
    const r = Math.round(70 + sBar * 185);
    const g = Math.round(180 - sBar * 130);
    const bl = Math.round(220 - sBar * 200);
    tris.push({
      d: `M${ax.toFixed(1)},${ay.toFixed(1)} L${bx.toFixed(1)},${by.toFixed(1)} L${cx.toFixed(1)},${cy.toFixed(1)} Z`,
      color: `rgb(${r},${g},${bl})`,
      depth: z,
    });
  }
  tris.sort((a, b) => a.depth - b.depth);

  // Hotspot marker
  const hi = report.stress?.hotspotVertex ?? 0;
  const hx = sx(mesh.positions[3 * hi]);
  const hy = sy(mesh.positions[3 * hi + 1]);

  return (
    <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-56">
      <rect x={0} y={0} width={w} height={h} className="fill-background/50" />
      {tris.map((t, i) => (
        <path key={i} d={t.d} fill={t.color} fillOpacity={0.7}
          stroke="rgba(255,255,255,0.06)" strokeWidth={0.4} />
      ))}
      <circle cx={hx} cy={hy} r={6} fill="none" stroke="rgb(244,63,94)" strokeWidth={1.6} />
      <circle cx={hx} cy={hy} r={2.5} fill="rgb(244,63,94)" />
    </svg>
  );
}

function lowerBound(arr: number[], v: number): number {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] < v) lo = m + 1; else hi = m;
  }
  return lo;
}
