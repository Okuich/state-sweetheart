// Retrieval-Augmented Physics Reasoning — UI panel
import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Sparkles, AlertTriangle, Loader2 } from "lucide-react";
import { loadGraph, type Graph } from "@/lib/knowledgeGraph";
import {
  retrieveAll,
  synthesizeQuery,
  type RagContext,
} from "@/lib/ragRetrieval";
import {
  recommendSimulationParameters,
  type Recommendation,
} from "@/server/physicsReasoner.functions";

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
    return retrieveAll(q, graph, k);
  }, [text, target, proc, k, graph]);

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

      {/* summary line */}
      <div className="rounded-md border border-border bg-background/40 px-3 py-2 text-[11px] font-mono text-foreground/80">
        {ctx.summary || "no context yet"}
      </div>

      {/* retrievals */}
      <div className="grid gap-4 md:grid-cols-2">
        <Bucket title="geometry · cosine kNN" empty="no geometry nodes yet">
          {ctx.geometry.map((r) => (
            <Row key={r.item.id} label={r.item.label} score={r.score} reason={r.reason} />
          ))}
        </Bucket>
        <Bucket title="topology · Jaccard match" empty="no topology overlap">
          {ctx.topology.map((r) => (
            <Row key={r.item.id} label={r.item.label} score={r.score} reason={r.reason} />
          ))}
        </Bucket>
        <Bucket title="historical failures" empty="no recorded failures">
          {ctx.failures.map((r) => (
            <Row key={r.item.id} label={r.item.label} score={r.score} reason={r.reason} tone="danger" />
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
            />
          ))}
        </Bucket>
      </div>

      {/* synthesized prompt context preview */}
      <div className="rounded-md border border-border bg-card/60 p-3">
        <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-1">
          context · ready for downstream reasoner
        </div>
        <pre className="text-[10px] font-mono text-foreground/80 whitespace-pre-wrap break-words max-h-56 overflow-auto">
{renderPrompt(ctx)}
        </pre>
      </div>
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
  label, score, reason, tone = "default",
}: { label: string; score: number; reason: string; tone?: "default" | "ok" | "danger" }) {
  const cls = tone === "ok" ? "text-primary" : tone === "danger" ? "text-destructive" : "text-foreground/90";
  return (
    <li className="grid grid-cols-[1fr_60px] gap-2 items-start text-[11px]">
      <div>
        <div className={`font-mono ${cls}`}>{label}</div>
        <div className="text-[9px] text-muted-foreground/80">{reason}</div>
      </div>
      <div className="text-right">
        <span className="inline-block w-12 font-mono tabular-nums text-foreground/70">
          {score.toFixed(3)}
        </span>
        <div className="h-1 mt-0.5 rounded-sm bg-muted overflow-hidden">
          <div className="h-full bg-primary"
               style={{ width: `${Math.max(0, Math.min(1, score)) * 100}%` }} />
        </div>
      </div>
    </li>
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
