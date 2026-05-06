import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  parseStep, buildTopology, describe, SAMPLE_STEP,
} from "@/lib/stepParser";
import {
  analyzeGeometry, type FeatureIntelligence, type Severity,
} from "@/lib/geometryFeatures";

const VARIANTS: { label: string; src: string }[] = [
  { label: "cube · canonical", src: SAMPLE_STEP },
  {
    label: "bracket · holes + fillets",
    src: SAMPLE_STEP
      .replace(/MANIFOLD_SOLID_BREP\('cube'/, "MANIFOLD_SOLID_BREP('bracket'")
      + `#90 = CYLINDRICAL_SURFACE('h0',#10,0.1);
#91 = CYLINDRICAL_SURFACE('h1',#11,0.1);
#92 = CYLINDRICAL_SURFACE('h2',#12,0.1);
#93 = TOROIDAL_SURFACE('fil0',#10,0.05,0.005);
#94 = TOROIDAL_SURFACE('fil1',#11,0.05,0.005);
#95 = CONICAL_SURFACE('cham0',#12,0.02,15.);
`,
  },
  {
    label: "freeform · molded shell",
    src: SAMPLE_STEP + `#100 = B_SPLINE_SURFACE_WITH_KNOTS('s0',3,3,(),.UNSPECIFIED.,.F.,.F.,.F.);
#101 = B_SPLINE_SURFACE_WITH_KNOTS('s1',3,3,(),.UNSPECIFIED.,.F.,.F.,.F.);
#102 = B_SPLINE_SURFACE_WITH_KNOTS('s2',3,3,(),.UNSPECIFIED.,.F.,.F.,.F.);
#103 = B_SPLINE_SURFACE_WITH_KNOTS('s3',3,3,(),.UNSPECIFIED.,.F.,.F.,.F.);
#104 = B_SPLINE_SURFACE_WITH_KNOTS('s4',3,3,(),.UNSPECIFIED.,.F.,.F.,.F.);
#105 = TOROIDAL_SURFACE('blend',#10,0.04,0.004);
`,
  },
  {
    label: "thin shaft · slender",
    src: SAMPLE_STEP
      .replace("CARTESIAN_POINT('p1',(1.,0.,0.))", "CARTESIAN_POINT('p1',(20.,0.,0.))")
      .replace("CARTESIAN_POINT('p2',(1.,1.,0.))", "CARTESIAN_POINT('p2',(20.,1.,0.))")
      .replace("CARTESIAN_POINT('p5',(1.,0.,1.))", "CARTESIAN_POINT('p5',(20.,0.,1.))")
      .replace("CARTESIAN_POINT('p6',(1.,1.,1.))", "CARTESIAN_POINT('p6',(20.,1.,1.))"),
  },
];

function sevColor(s: Severity) {
  return s === "high" ? "text-destructive" : s === "medium" ? "text-accent" : "text-primary";
}

function RiskBar({ label, value }: { label: string; value: number }) {
  const c = value > 0.66 ? "bg-destructive" : value > 0.33 ? "bg-accent" : "bg-primary";
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[10px] uppercase tracking-[0.18em]">
        <span className="text-muted-foreground">{label}</span>
        <span className="text-foreground tabular-nums">{(value * 100).toFixed(0)}%</span>
      </div>
      <div className="h-1.5 rounded bg-muted-foreground/15">
        <div className={`h-full rounded ${c}`} style={{ width: `${value * 100}%` }} />
      </div>
    </div>
  );
}

export function GeometryFeaturePanel() {
  const [variant, setVariant] = useState(0);
  const [intel, setIntel] = useState<FeatureIntelligence | null>(null);
  const [running, setRunning] = useState(false);

  const run = (idx = variant) => {
    setRunning(true);
    requestAnimationFrame(() => {
      const r = parseStep(VARIANTS[idx].src);
      const t = buildTopology(r);
      const d = describe(r, t);
      setIntel(analyzeGeometry(r, t, d));
      setRunning(false);
    });
  };

  const embView = useMemo(() => {
    if (!intel) return null;
    const W = 480, H = 56;
    const max = Math.max(...intel.embedding.map(Math.abs), 1e-3);
    return { W, H, max, vals: intel.embedding };
  }, [intel]);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            geometry · feature intelligence
          </div>
          <h2 className="font-display text-2xl text-foreground">
            From CAD to <span className="text-primary">physical</span> features.
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {intel ? <>analysis · <span className="text-foreground">{intel.ms.toFixed(1)} ms</span></> : "no model"}
        </div>
      </div>

      {/* Variant picker */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
        {VARIANTS.map((v, i) => (
          <button
            key={v.label}
            onClick={() => { setVariant(i); run(i); }}
            className={`text-left rounded-md border px-3 py-2 transition ${
              variant === i
                ? "border-primary bg-primary/10 text-foreground"
                : "border-border bg-background/40 text-muted-foreground hover:text-foreground"
            }`}
          >
            <div className="text-[10px] uppercase tracking-[0.2em]">{v.label}</div>
          </button>
        ))}
      </div>

      <div className="flex gap-2">
        <Button onClick={() => run()} disabled={running}
          className="uppercase tracking-[0.18em] text-[10px]">
          {running ? "analyzing…" : intel ? "re-analyze" : "analyze"}
        </Button>
      </div>

      {intel && (
        <>
          {/* Findings */}
          <div className="rounded-md border border-border bg-background/30 p-3">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
              manufacturability findings
            </div>
            <ul className="space-y-1 font-mono text-[10px]">
              {intel.findings.map((f, i) => (
                <li key={i} className="flex items-start gap-2">
                  <span className={`shrink-0 ${sevColor(f.severity)}`}>[{f.severity.padEnd(6)}]</span>
                  <span className="text-foreground/95 w-44">{f.kind}</span>
                  <span className="text-muted-foreground">×{f.count}</span>
                  <span className="text-muted-foreground/85">{f.detail}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Risk + curvature */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="rounded-md border border-border bg-background/30 p-3 space-y-3">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">risk heuristics</div>
              <RiskBar label="stress concentration" value={intel.risk.stressConcentration} />
              <RiskBar label="thermal risk"          value={intel.risk.thermalRisk} />
              <RiskBar label="fabrication difficulty" value={intel.risk.fabricationDifficulty} />
              <ul className="text-[10px] font-mono text-muted-foreground/90 space-y-0.5 pt-1">
                {intel.risk.notes.map((n, i) => <li key={i}>· {n}</li>)}
              </ul>
            </div>

            <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">curvature signature</div>
              {Object.entries({
                planar: intel.curvature.planar,
                cylindrical: intel.curvature.cylindrical,
                conical: intel.curvature.conical,
                spherical: intel.curvature.spherical,
                toroidal: intel.curvature.toroidal,
                spline: intel.curvature.spline,
              }).map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 text-[10px] font-mono">
                  <span className="w-24 text-muted-foreground">{k}</span>
                  <div className="flex-1 h-1.5 bg-muted-foreground/15 rounded">
                    <div className="h-full bg-primary rounded" style={{ width: `${v * 100}%` }} />
                  </div>
                  <span className="w-10 text-right text-foreground/90 tabular-nums">{(v * 100).toFixed(0)}%</span>
                </div>
              ))}
              <div className="grid grid-cols-2 gap-2 pt-1 text-[10px]">
                <Stat label="H proxy" value={intel.curvature.meanCurvatureProxy.toFixed(3)} />
                <Stat label="K proxy" value={intel.curvature.gaussianCurvatureProxy.toFixed(3)} />
              </div>
            </div>
          </div>

          {/* Topology */}
          <div className="rounded-md border border-border bg-background/30 p-3">
            <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
              topology descriptor
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
              <Stat label="components" value={String(intel.topology.components)} />
              <Stat label="max depth" value={String(intel.topology.maxDepth)} />
              <Stat label="mean out-deg" value={intel.topology.meanOutDegree.toFixed(2)} />
              <Stat label="max out-deg" value={String(intel.topology.maxOutDegree)} />
              <Stat label="face/shell" value={intel.topology.faceShellRatio.toFixed(2)} />
              <Stat label="edge/face" value={intel.topology.edgeFaceRatio.toFixed(2)} />
              <Stat label="V-E+F" value={intel.topology.euler.toFixed(0)} />
              <Stat label="cyclomatic" value={String(intel.topology.cyclomatic)} />
            </div>
          </div>

          {/* Embedding + fab vector */}
          <div className="rounded-md border border-border bg-background/30 p-3 space-y-3">
            <div className="flex items-baseline justify-between">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
                geometry embedding · 32-d L2-normalized
              </div>
              <div className="text-[10px] font-mono text-muted-foreground">
                |x| = {(Math.sqrt(intel.embedding.reduce((s, v) => s + v * v, 0))).toFixed(3)}
              </div>
            </div>
            {embView && (
              <svg viewBox={`0 0 ${embView.W} ${embView.H}`} width="100%" height={embView.H} className="block">
                <line x1={0} y1={embView.H / 2} x2={embView.W} y2={embView.H / 2}
                  stroke="currentColor" className="text-muted-foreground/30" />
                {embView.vals.map((v, i) => {
                  const w = embView.W / embView.vals.length;
                  const h = (Math.abs(v) / embView.max) * (embView.H / 2 - 2);
                  const y = v >= 0 ? embView.H / 2 - h : embView.H / 2;
                  return (
                    <rect key={i} x={i * w + 1} y={y} width={Math.max(2, w - 2)} height={h}
                      className={v >= 0 ? "fill-primary" : "fill-secondary"} />
                  );
                })}
              </svg>
            )}
            <div>
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-1">
                fab feature vector · 12-d
              </div>
              <div className="grid grid-cols-3 md:grid-cols-6 gap-1 text-[10px] font-mono">
                {intel.fabFeatureVector.map((v, i) => (
                  <div key={i} className="rounded border border-border/60 px-1.5 py-0.5 flex justify-between">
                    <span className="text-muted-foreground">f{i}</span>
                    <span className="text-foreground/90 tabular-nums">{v.toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-border/60 px-2 py-1">
      <div className="uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className="font-mono text-foreground/90 tabular-nums">{value}</div>
    </div>
  );
}
