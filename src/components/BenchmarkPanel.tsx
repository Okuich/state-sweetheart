import { useState } from "react";
import { Button } from "@/components/ui/button";
import { runAll, type BenchResult, type MetricResult } from "@/lib/benchmarks";

function Sparkline({ trace, reference }: { trace: number[]; reference: number[] }) {
  if (trace.length < 2) return null;
  const W = 220, H = 40;
  const all = trace.concat(reference);
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = Math.max(1e-9, hi - lo);
  const path = (vs: number[]) => vs.map((v, i) => {
    const x = (i / (vs.length - 1)) * W;
    const y = H - ((v - lo) / span) * (H - 4) - 2;
    return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return (
    <svg width={W} height={H} className="block">
      <path d={path(reference)} fill="none" stroke="currentColor"
        className="text-muted-foreground/40" strokeWidth={1} strokeDasharray="2 2" />
      <path d={path(trace)} fill="none" stroke="currentColor"
        className="text-primary" strokeWidth={1.4} />
    </svg>
  );
}

function MetricRow({ m }: { m: MetricResult }) {
  const fmt = (v: number) => Math.abs(v) > 0 && Math.abs(v) < 1e-3
    ? v.toExponential(2) : v.toFixed(3);
  return (
    <div className="flex items-center justify-between text-[10px] font-mono">
      <span className="text-muted-foreground">{m.name}</span>
      <span className="flex items-center gap-2">
        <span className={m.pass ? "text-foreground/90" : "text-destructive"}>
          {fmt(m.value)}{m.unit ?? ""}
        </span>
        <span className="text-muted-foreground/60">
          / {m.higherIsBetter ? "≥" : "≤"} {fmt(m.threshold)}{m.unit ?? ""}
        </span>
        <span className={`h-1.5 w-1.5 rounded-full ${m.pass ? "bg-primary" : "bg-destructive"}`} />
      </span>
    </div>
  );
}

export function BenchmarkPanel() {
  const [results, setResults] = useState<BenchResult[] | null>(null);
  const [pass, setPass] = useState<boolean | null>(null);
  const [ms, setMs] = useState(0);
  const [running, setRunning] = useState(false);

  const run = () => {
    setRunning(true);
    requestAnimationFrame(() => {
      const r = runAll();
      setResults(r.results); setPass(r.pass); setMs(r.ms); setRunning(false);
    });
  };

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            truth · benchmark suite
          </div>
          <h2 className="font-display text-2xl text-foreground">
            Validate against <span className="text-primary">analytic</span> references.
          </h2>
        </div>
        <div className="flex items-center gap-3 text-[10px] uppercase tracking-[0.18em]">
          {pass !== null && (
            <span className={pass ? "text-primary" : "text-destructive"}>
              <span className={`inline-block mr-1.5 h-1.5 w-1.5 rounded-full ${pass ? "bg-primary glow-mint animate-pulse" : "bg-destructive"}`} />
              {pass ? "build · pass" : "build · fail"}
            </span>
          )}
          {results && (
            <span className="text-muted-foreground">
              {results.length} suites · {ms.toFixed(1)} ms
            </span>
          )}
          <Button onClick={run} disabled={running}
            className="uppercase tracking-[0.18em] text-[10px]">
            {running ? "running…" : results ? "re-run" : "run all"}
          </Button>
        </div>
      </div>

      {!results && (
        <div className="rounded-md border border-dashed border-border bg-background/30 p-6 text-center text-[11px] text-muted-foreground">
          No results yet — every runtime build must pass before deployment.
        </div>
      )}

      {results && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {results.map((r) => (
            <div key={r.key} className={`rounded-md border p-3 space-y-2 ${
              r.pass ? "border-border bg-background/30" : "border-destructive/40 bg-destructive/5"
            }`}>
              <div className="flex items-baseline justify-between">
                <div>
                  <div className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">{r.key}</div>
                  <div className="text-sm text-foreground/95">{r.label}</div>
                </div>
                <span className={`text-[10px] uppercase tracking-[0.18em] ${r.pass ? "text-primary" : "text-destructive"}`}>
                  {r.pass ? "pass" : "fail"} · {r.ms.toFixed(1)}ms
                </span>
              </div>
              <div className="text-[10px] text-muted-foreground/80 leading-snug font-mono">{r.blurb}</div>
              <div className="text-muted-foreground"><Sparkline trace={r.trace} reference={r.reference} /></div>
              <div className="space-y-0.5">
                {r.metrics.map((m, i) => <MetricRow key={i} m={m} />)}
              </div>
            </div>
          ))}
        </div>
      )}

      {results && (
        <div className="rounded-md border border-border bg-background/20 px-3 py-2 text-[10px] font-mono text-muted-foreground">
          deployment gate · {pass
            ? <span className="text-primary">all thresholds satisfied → ready to ship</span>
            : <span className="text-destructive">{results.filter((r) => !r.pass).length} suite(s) failed → block deploy</span>
          }
        </div>
      )}
    </div>
  );
}
