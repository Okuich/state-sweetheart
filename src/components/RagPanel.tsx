// Retrieval-Augmented Physics Reasoning — UI panel
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sparkles, AlertTriangle, Loader2, ChevronDown, ChevronRight } from "lucide-react";
import { loadGraph, type Graph } from "@/lib/knowledgeGraph";
import {
  retrieveAll,
  synthesizeQuery,
  DEFAULT_WEIGHTS,
  type RagContext,
  type RetrievalWeights,
  type Breakdown,
} from "@/lib/ragRetrieval";
import {
  recommendSimulationParameters,
  type Recommendation,
} from "@/lib/physicsReasoner.functions";

const PRESETS = [
  "thin-wall bracket high stress aluminum",
  "deep pocket housing 5-axis milling",
  "lattice cube SLS porosity",
  "thin rib array fatigue cycle",
  "fillet arm titanium 5-axis",
];

export function RagPanel() {
  const [graph, setGraph] = useState<Graph>(() => loadGraph());
  const [text, setText] = useState(PRESETS[0]);
  const [target, setTarget] = useState(280);
  const [proc, setProc] = useState("mill-5ax");
  const [k, setK] = useState(5);
  const [weights, setWeights] = useState<RetrievalWeights>(DEFAULT_WEIGHTS);

  // re-load graph on mount + when window storage changes (other panel writes)
  useEffect(() => {
    const sync = () => setGraph(loadGraph());
    sync();
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);

  const ctx: RagContext = useMemo(() => {
    const q = synthesizeQuery(text, {
      targetStress: target,
      process: proc,
    });
    return retrieveAll(q, graph, k, weights);
  }, [text, target, proc, k, graph, weights]);

  const [reco, setReco] = useState<Recommendation | null>(null);
  const [reasoning, setReasoning] = useState(false);
  const [recoError, setRecoError] = useState<string | null>(null);

  const runReasoner = async () => {
    setReasoning(true);
    setRecoError(null);
    setReco(null);
    try {
      const r = await recommendSimulationParameters({
        data: { contextPrompt: renderPrompt(ctx), query: text },
      });
      setReco(r);
    } catch (e) {
      setRecoError(e instanceof Error ? e.message : String(e));
    } finally {
      setReasoning(false);
    }
  };

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
            module · retrieval-augmented-reasoning
          </div>
          <h2 className="font-display text-2xl md:text-3xl text-glow">
            Retrieval-Augmented <span className="text-primary">Physics</span> Reasoning
          </h2>
          <p className="text-xs text-muted-foreground max-w-xl mt-1">
            Pull relevant historical geometry, topology matches, past failures
            and optimization tweaks from the persistent knowledge graph at
            runtime — wired straight into the reasoner's prompt context.
          </p>
        </div>
        <Button variant="outline" onClick={() => setGraph(loadGraph())}
                className="uppercase tracking-[0.16em] text-[10px]">
          refresh · graph
        </Button>
      </header>

      {/* query controls */}
      <div className="grid gap-3 md:grid-cols-[1fr_140px_140px_100px]">
        <label className="flex flex-col gap-1">
          <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">query</span>
          <input
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, 200))}
            className="rounded-md border border-border bg-background/60 px-2 py-1.5 font-mono text-xs outline-none focus:ring-1 focus:ring-primary"
            placeholder="describe the design / failure mode you're investigating"
          />
        </label>
        <NumInput label="target σ (MPa)" value={target} onChange={setTarget} min={10} max={900} step={10} />
        <label className="flex flex-col gap-1">
          <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">process</span>
          <select value={proc} onChange={(e) => setProc(e.target.value)}
            className="rounded-md border border-border bg-background/60 px-2 py-1.5 font-mono text-xs outline-none focus:ring-1 focus:ring-primary">
            {["mill-3ax","mill-5ax","SLA","FDM","SLS"].map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </label>
        <NumInput label="k" value={k} onChange={setK} min={1} max={10} step={1} />
      </div>

      <div className="flex flex-wrap gap-1">
        {PRESETS.map((p) => (
          <button key={p} type="button" onClick={() => setText(p)}
            className="text-[9px] uppercase tracking-[0.14em] px-1.5 py-0.5 rounded border border-border text-muted-foreground hover:text-foreground hover:border-primary">
            {p.split(" ").slice(0, 3).join(" ")}
          </button>
        ))}
      </div>

      {/* retrieval weight sliders */}
      <div className="rounded-lg border border-border bg-background/40 p-3">
        <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
          <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
            retrieval weights · re-rank scores per channel
          </div>
          <button
            type="button"
            onClick={() => setWeights(DEFAULT_WEIGHTS)}
            className="text-[9px] uppercase tracking-[0.16em] text-muted-foreground hover:text-foreground"
          >
            reset
          </button>
        </div>
        <div className="grid gap-3 md:grid-cols-4">
          <WeightSlider label="geometry"    value={weights.geometry} onChange={(v) => setWeights((w) => ({ ...w, geometry: v }))} />
          <WeightSlider label="topology"    value={weights.topology} onChange={(v) => setWeights((w) => ({ ...w, topology: v }))} />
          <WeightSlider label="failure"     value={weights.failure}  onChange={(v) => setWeights((w) => ({ ...w, failure:  v }))} />
          <WeightSlider label="optimization" value={weights.optim}    onChange={(v) => setWeights((w) => ({ ...w, optim:    v }))} />
        </div>
      </div>

      <div className="rounded-md border border-border bg-background/40 px-3 py-2 text-[11px] font-mono text-foreground/80">
        {ctx.summary || "no context yet"}
      </div>

      {/* retrievals */}
      <div className="grid gap-4 md:grid-cols-2">
        <Bucket title="geometry · cosine kNN" empty="no geometry nodes yet">
          {ctx.geometry.map((r) => (
            <Row key={r.item.id} label={r.item.label} score={r.score} reason={r.reason} breakdown={r.breakdown} />
          ))}
        </Bucket>
        <Bucket title="topology · Jaccard match" empty="no topology overlap">
          {ctx.topology.map((r) => (
            <Row key={r.item.id} label={r.item.label} score={r.score} reason={r.reason} breakdown={r.breakdown} />
          ))}
        </Bucket>
        <Bucket title="historical failures" empty="no recorded failures">
          {ctx.failures.map((r) => (
            <Row key={r.item.id} label={r.item.label} score={r.score} reason={r.reason} tone="danger" breakdown={r.breakdown} />
          ))}
        </Bucket>
        <Bucket title="optimization memory" empty="no optimization history">
          {ctx.optimizations.map((r) => (
            <Row
              key={`${r.item.parent.id}->${r.item.child.id}`}
              label={`${r.item.parent.label} → ${r.item.child.label}`}
              score={r.score}
              reason={r.item.note || r.reason}
              tone="ok"
              breakdown={r.breakdown}
            />
          ))}
        </Bucket>
      </div>

      {/* synthesized prompt context preview */}
      <div className="rounded-md border border-border bg-card/60 p-3">
        <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
          <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
            context · ready for downstream reasoner
          </div>
          <Button
            size="sm"
            onClick={runReasoner}
            disabled={reasoning}
            className="uppercase tracking-[0.16em] text-[10px]"
          >
            {reasoning
              ? <><Loader2 className="h-3 w-3 mr-1 animate-spin" /> reasoning…</>
              : <><Sparkles className="h-3 w-3 mr-1" /> recommend sim parameters</>}
          </Button>
        </div>
        <pre className="text-[10px] font-mono text-foreground/80 whitespace-pre-wrap break-words max-h-56 overflow-auto">
{renderPrompt(ctx)}
        </pre>
      </div>

      {recoError && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-[11px] text-destructive flex items-start gap-2">
          <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0" />
          <span>{recoError}</span>
        </div>
      )}

      {reco && (
        <div className="rounded-md border border-primary/40 bg-primary/[0.04] p-3 space-y-3">
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="text-[10px] uppercase tracking-[0.22em] text-primary">
              recommended next simulation
            </div>
            <Badge variant="outline" className="text-[10px]">
              confidence · {(reco.confidence * 100).toFixed(0)}%
            </Badge>
          </div>
          <p className="text-[11px] text-foreground/85 leading-relaxed">
            {reco.rationale}
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
            {reco.parameters.map((p, i) => (
              <div key={i} className="rounded border border-border/60 bg-background/40 p-2">
                <div className="flex items-baseline justify-between gap-2">
                  <span className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                    {p.name}
                  </span>
                  <span className="font-mono text-[12px] text-foreground tabular-nums">
                    {String(p.value)}{p.unit ? ` ${p.unit}` : ""}
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground/80 mt-1">
                  {p.rationale}
                </div>
              </div>
            ))}
          </div>
          {reco.warnings.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-destructive mb-1">
                warnings
              </div>
              <ul className="space-y-0.5 text-[11px] text-destructive/90 font-mono">
                {reco.warnings.map((w, i) => <li key={i}>· {w}</li>)}
              </ul>
            </div>
          )}
          {reco.nextActions.length > 0 && (
            <div>
              <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground mb-1">
                next actions
              </div>
              <ul className="space-y-0.5 text-[11px] text-foreground/85 font-mono">
                {reco.nextActions.map((a, i) => <li key={i}>→ {a}</li>)}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function renderPrompt(c: RagContext): string {
  const lines: string[] = [];
  lines.push(`# QUERY`);
  lines.push(`hint: ${c.query.hint ?? ""}`);
  if (c.query.context) lines.push(`ctx:  ${JSON.stringify(c.query.context)}`);
  lines.push("");
  if (c.geometry.length) {
    lines.push(`# similar geometries (cosine)`);
    c.geometry.forEach((r) => lines.push(`  - ${r.item.label}  s=${r.score.toFixed(3)}`));
    lines.push("");
  }
  if (c.topology.length) {
    lines.push(`# topology matches (Jaccard)`);
    c.topology.forEach((r) => lines.push(`  - ${r.item.label}  s=${r.score.toFixed(3)}`));
    lines.push("");
  }
  if (c.failures.length) {
    lines.push(`# past failures to avoid`);
    c.failures.forEach((r) =>
      lines.push(`  - ${r.item.label}  s=${r.score.toFixed(3)}  (${r.reason})`)
    );
    lines.push("");
  }
  if (c.optimizations.length) {
    lines.push(`# optimization precedents`);
    c.optimizations.forEach((r) =>
      lines.push(`  - ${r.item.parent.label} → ${r.item.child.label}  (${r.item.note})`)
    );
  }
  return lines.join("\n");
}

// ─── tiny atoms ──────────────────────────────────────────────
function Bucket({
  title, empty, children,
}: { title: string; empty: string; children: React.ReactNode }) {
  const arr = Array.isArray(children) ? children : [children];
  const has = arr.filter(Boolean).length > 0;
  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-2">{title}</div>
      {has ? <ul className="space-y-1.5">{children}</ul>
           : <div className="text-[11px] text-muted-foreground">{empty}</div>}
    </div>
  );
}

function Row({
  label, score, reason, tone = "default", breakdown,
}: { label: string; score: number; reason: string; tone?: "default" | "ok" | "danger"; breakdown?: Breakdown }) {
  const [open, setOpen] = useState(false);
  const cls = tone === "ok" ? "text-primary" : tone === "danger" ? "text-destructive" : "text-foreground/90";
  const Icon = open ? ChevronDown : ChevronRight;
  return (
    <li className="text-[11px]">
      <div className="grid grid-cols-[1fr_60px] gap-2 items-start">
        <button
          type="button"
          onClick={() => breakdown && setOpen((o) => !o)}
          className="text-left flex items-start gap-1 group"
          disabled={!breakdown}
        >
          {breakdown && (
            <Icon className="h-3 w-3 mt-0.5 shrink-0 text-muted-foreground group-hover:text-foreground" />
          )}
          <div className="min-w-0">
            <div className={`font-mono ${cls} truncate`}>{label}</div>
            <div className="text-[9px] text-muted-foreground/80 truncate">{reason}</div>
          </div>
        </button>
        <div className="text-right">
          <span className="inline-block w-12 font-mono tabular-nums text-foreground/70">
            {score.toFixed(3)}
          </span>
          <div className="h-1 mt-0.5 rounded-sm bg-muted overflow-hidden">
            <div className="h-full bg-primary"
                 style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }} />
          </div>
        </div>
      </div>
      {open && breakdown && <Explanation b={breakdown} />}
    </li>
  );
}

function Explanation({ b }: { b: Breakdown }) {
  return (
    <div className="mt-2 ml-4 rounded-md border border-border/60 bg-background/60 p-2 space-y-2">
      <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
        why · breakdown
      </div>
      <div className="font-mono text-[10px] text-foreground/80">{b.formula}</div>

      {/* score components */}
      <div className="space-y-1">
        {b.components.map((c, i) => {
          const contrib = (c.weight ?? 1) * c.value;
          return (
            <div key={i} className="grid grid-cols-[80px_1fr_70px] gap-2 items-center">
              <span className="text-[10px] text-muted-foreground">{c.label}</span>
              <div className="h-1.5 rounded-sm bg-muted overflow-hidden">
                <div
                  className="h-full bg-primary/80"
                  style={{ width: `${Math.max(0, Math.min(1, c.value)) * 100}%` }}
                />
              </div>
              <span className="font-mono text-[10px] tabular-nums text-foreground/80 text-right">
                {c.weight !== undefined
                  ? `${c.weight.toFixed(2)}·${c.value.toFixed(2)} = ${contrib.toFixed(2)}`
                  : c.value.toFixed(3)}
              </span>
            </div>
          );
        })}
      </div>

      {/* token overlap */}
      {(b.overlap || b.onlyQuery || b.onlyCandidate) && (
        <div className="space-y-1">
          <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
            token overlap
          </div>
          <TokenList label="∩ shared"     items={b.overlap}       cls="text-primary border-primary/40 bg-primary/5" />
          <TokenList label="only query"   items={b.onlyQuery}     cls="text-foreground/70 border-border bg-background/40" />
          <TokenList label="only node"    items={b.onlyCandidate} cls="text-muted-foreground border-border bg-background/40" />
        </div>
      )}

      {/* per-dim cosine */}
      {b.cosineDims && b.cosineDims.length > 0 && (
        <div>
          <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground mb-1">
            cosine · per-dim contribution (normalized)
          </div>
          <div className="flex items-end gap-0.5 h-8">
            {b.cosineDims.map((d, i) => {
              const h = Math.min(1, Math.abs(d)) * 100;
              return (
                <div key={i} className="flex-1 flex flex-col justify-end" title={`d${i}: ${d.toFixed(3)}`}>
                  <div
                    className={d >= 0 ? "bg-primary/80" : "bg-destructive/70"}
                    style={{ height: `${h}%`, minHeight: "1px" }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function TokenList({ label, items, cls }: { label: string; items?: string[]; cls: string }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="flex items-start gap-2">
      <span className="text-[9px] uppercase tracking-[0.14em] text-muted-foreground shrink-0 w-16 mt-0.5">{label}</span>
      <div className="flex flex-wrap gap-1">
        {items.slice(0, 24).map((t, i) => (
          <span key={i} className={`text-[9px] font-mono px-1 py-0.5 rounded border ${cls}`}>
            {t}
          </span>
        ))}
        {items.length > 24 && (
          <span className="text-[9px] text-muted-foreground">+{items.length - 24}</span>
        )}
      </div>
    </div>
  );
}

function NumInput({
  label, value, onChange, min, max, step,
}: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</span>
      <input type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md border border-border bg-background/60 px-2 py-1.5 font-mono text-xs outline-none focus:ring-1 focus:ring-primary" />
    </label>
  );
}

function WeightSlider({
  label, value, onChange,
}: { label: string; value: number; onChange: (v: number) => void }) {
  return (
    <label className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between">
        <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</span>
        <span className="font-mono text-[10px] tabular-nums text-foreground/80">{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={0}
        max={2}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="accent-primary w-full"
      />
    </label>
  );
}
