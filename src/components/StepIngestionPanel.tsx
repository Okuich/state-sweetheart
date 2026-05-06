import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { StepTopologyGraph } from "@/components/StepTopologyGraph";
import {
  parseStep, buildTopology, describe, validate, repair,
  SAMPLE_STEP,
  type ParseReport, type TopoGraph, type GeomDescriptor, type ValidationReport, type RepairResult,
} from "@/lib/stepParser";

export function StepIngestionPanel() {
  const [src, setSrc] = useState<string>(SAMPLE_STEP);
  const [report, setReport] = useState<ParseReport | null>(null);
  const [topo, setTopo] = useState<TopoGraph | null>(null);
  const [desc, setDesc] = useState<GeomDescriptor | null>(null);
  const [valid, setValid] = useState<ValidationReport | null>(null);
  const [repaired, setRepaired] = useState<RepairResult | null>(null);
  const [tab, setTab] = useState<"summary" | "graph" | "json" | "issues">("summary");

  const ingest = () => {
    const r = parseStep(src);
    const t = buildTopology(r);
    const d = describe(r, t);
    const v = validate(r, t, d);
    setReport(r); setTopo(t); setDesc(d); setValid(v); setRepaired(null);
  };
  const doRepair = () => {
    if (!report || !topo) return;
    const res = repair(report, topo, 1e-4);
    const t = buildTopology(report);
    const d = describe(report, t);
    const v = validate(report, t, d);
    setTopo(t); setDesc(d); setValid(v); setRepaired(res);
  };
  const loadSample = () => setSrc(SAMPLE_STEP);
  const breakSample = () => {
    // Inject orphans + a degenerate edge
    setSrc(SAMPLE_STEP
      .replace("#30 = EDGE_CURVE('e0',#20,#21,$,.T.);", "#30 = EDGE_CURVE('e0',#20,#20,$,.T.);")
      .replace("#65 = ADVANCED_FACE('f5',(#33,#38,#37,#41),#55,.T.);",
               "#65 = ADVANCED_FACE('f5',(#33,#38,#37,#999),#55,.T.);"));
  };

  const topTypes = useMemo(() => {
    if (!report) return [];
    return [...report.byType.entries()]
      .map(([k, v]) => [k, v.length] as const)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 10);
  }, [report]);

  const json = useMemo(() => {
    if (!desc || !report) return "";
    return JSON.stringify({
      schema: desc.schema,
      header: report.header,
      counts: desc.counts,
      manifold: desc.manifold,
      bbox: desc.bbox,
      features: desc.features,
    }, null, 2);
  }, [desc, report]);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            cad · ingestion
          </div>
          <h2 className="font-display text-2xl text-foreground">
            STEP <span className="text-primary">→</span> canonical geometry.
          </h2>
        </div>
        <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          {desc ? <>schema · <span className="text-primary">{desc.schema}</span></> : "no file"}
          {report && <span className="ml-3">parse · <span className="text-foreground">{report.durationMs.toFixed(1)}ms</span></span>}
        </div>
      </div>

      {/* Source editor */}
      <div className="rounded-md border border-border bg-background/30 p-3 space-y-2">
        <div className="flex items-center justify-between">
          <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            STEP source · ISO-10303-21
          </div>
          <div className="flex gap-1">
            <button onClick={loadSample}
              className="text-[9px] uppercase tracking-[0.18em] px-2 py-0.5 rounded border border-border text-muted-foreground hover:text-foreground">
              sample
            </button>
            <button onClick={breakSample}
              className="text-[9px] uppercase tracking-[0.18em] px-2 py-0.5 rounded border border-border text-muted-foreground hover:text-foreground">
              corrupt
            </button>
          </div>
        </div>
        <textarea
          value={src}
          onChange={(e) => setSrc(e.target.value)}
          spellCheck={false}
          rows={8}
          className="w-full resize-y rounded-md border border-border bg-background/60 px-2 py-1.5 font-mono text-[10px] text-foreground/90 outline-none focus:ring-1 focus:ring-primary"
        />
        <div className="flex gap-2">
          <Button onClick={ingest} className="uppercase tracking-[0.18em] text-[10px]">ingest</Button>
          <Button onClick={doRepair} variant="outline"
            disabled={!report || valid?.ok}
            className="uppercase tracking-[0.18em] text-[10px]">
            repair + normalize
          </Button>
        </div>
      </div>

      {/* Tabs */}
      {report && desc && valid && topo && (
        <>
          <div className="flex gap-1">
            {(["summary", "graph", "issues", "json"] as const).map((t) => (
              <button key={t} onClick={() => setTab(t)}
                className={`text-[10px] uppercase tracking-[0.18em] px-2.5 py-1 rounded border transition ${
                  tab === t ? "border-primary bg-primary/10 text-foreground"
                  : "border-border bg-background/30 text-muted-foreground hover:text-foreground"
                }`}>
                {t}{t === "issues" && valid.issues.length > 0 ? ` · ${valid.issues.length}` : ""}
              </button>
            ))}
            <span className="ml-auto text-[10px] uppercase tracking-[0.18em] flex items-center">
              <span className={`h-1.5 w-1.5 rounded-full mr-1.5 ${
                valid.ok ? "bg-primary glow-mint animate-pulse" : "bg-destructive"
              }`} />
              <span className={valid.ok ? "text-primary" : "text-destructive"}>
                {valid.ok ? "valid" : `${valid.issues.filter((i) => i.severity === "error").length} error(s)`}
              </span>
              <span className="text-muted-foreground ml-3">manifold · </span>
              <span className={desc.manifold ? "text-primary" : "text-accent"}>
                {desc.manifold ? "yes" : "no"}
              </span>
            </span>
          </div>

          {tab === "summary" && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-[10px]">
              {Object.entries(desc.counts).map(([k, v]) => (
                <Stat key={k} label={k} value={String(v)} />
              ))}
              <Stat label="entities" value={String(report.entities.size)} />
              <Stat label="orphans" value={String(topo.orphans.length)} highlight={topo.orphans.length > 0} />
              <Stat label="roots" value={String(topo.roots.length)} />
              <Stat label="bbox" value={
                desc.bbox
                  ? `${(desc.bbox.max[0] - desc.bbox.min[0]).toFixed(2)}×${(desc.bbox.max[1] - desc.bbox.min[1]).toFixed(2)}×${(desc.bbox.max[2] - desc.bbox.min[2]).toFixed(2)}`
                  : "—"
              } />
            </div>
          )}

          {tab === "graph" && (
            <>
              <StepTopologyGraph report={report} topo={topo} />
            <div className="rounded-md border border-border bg-background/30 p-3 mt-3">
              <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
                top entity types
              </div>
              <div className="space-y-1">
                {topTypes.map(([t, n]) => {
                  const max = topTypes[0][1];
                  return (
                    <div key={t} className="flex items-center gap-2 text-[10px] font-mono">
                      <span className="w-44 text-muted-foreground truncate">{t}</span>
                      <div className="flex-1 h-1.5 bg-muted-foreground/20 rounded">
                        <div className="h-full bg-primary rounded" style={{ width: `${(n / max) * 100}%` }} />
                      </div>
                      <span className="w-10 text-right text-foreground/90 tabular-nums">{n}</span>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 text-[10px] uppercase tracking-[0.22em] text-muted-foreground mb-1">feature map</div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-2 text-[10px]">
                {Object.entries(desc.features).map(([k, v]) => (
                  <Stat key={k} label={k} value={String(v)} />
                ))}
              </div>
            </div>
          )}

          {tab === "issues" && (
            <div className="rounded-md border border-border bg-background/30 p-3">
              {repaired && (
                <div className="mb-2 rounded border border-primary/40 bg-primary/5 p-2 text-[10px] font-mono text-foreground/90">
                  repair · removed {repaired.removedOrphans} orphan ref(s) · collapsed {repaired.collapsedDegenerate} degenerate edge(s) · merged {repaired.mergedPoints} pt(s) within {repaired.toleranceBand}
                </div>
              )}
              {valid.issues.length === 0 ? (
                <div className="text-[10px] text-muted-foreground/70 font-mono">no issues</div>
              ) : (
                <ul className="space-y-1 font-mono text-[10px] max-h-[260px] overflow-auto">
                  {valid.issues.map((i, idx) => {
                    const c = i.severity === "error" ? "text-destructive"
                            : i.severity === "warning" ? "text-accent" : "text-primary";
                    return (
                      <li key={idx} className="flex items-start gap-2">
                        <span className={`shrink-0 ${c}`}>[{i.severity}]</span>
                        <span className="text-muted-foreground">{i.code}</span>
                        <span className="text-foreground/85">{i.message}</span>
                        {i.entity !== undefined && <span className="text-muted-foreground/70">#{i.entity}</span>}
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}

          {tab === "json" && (
            <pre className="rounded-md border border-border bg-background/30 p-3 text-[10px] font-mono text-foreground/85 overflow-auto max-h-[360px]">
{json}
            </pre>
          )}
        </>
      )}
    </div>
  );
}

function Stat({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className={`rounded border px-2 py-1 ${highlight ? "border-destructive/60 bg-destructive/5" : "border-border/60"}`}>
      <div className="uppercase tracking-[0.16em] text-muted-foreground">{label}</div>
      <div className={`font-mono tabular-nums ${highlight ? "text-destructive" : "text-foreground/90"}`}>{value}</div>
    </div>
  );
}
