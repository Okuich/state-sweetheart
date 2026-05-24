/**
 * Material Intelligence — Gating System panel.
 *
 *   Gate A · recommendation relevance > 85%
 *   Gate B · constraint satisfaction > 95%
 *   Gate C · inference latency < 200 ms
 *
 * Passing all three unlocks the Procurement Optimization Layer
 * (cost-aware sourcing view across simulated supplier quotes).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  BUILTIN_MATERIALS, buildIndex, recommend, DEFAULT_WEIGHTS,
  type CorpusIndex, type DesignConstraints, type MaterialScore,
  type ObjectiveWeights,
} from "@/lib/materials";
import {
  MatGateEvaluator, DEFAULT_MAT_GATES, type MatGateReport,
} from "@/lib/materials/gating";

// ---------------- synthetic scenario stream ----------------

type Scenario = {
  label: string;
  constraints: DesignConstraints;
  weights?: Partial<ObjectiveWeights>;
};

const SCENARIOS: Scenario[] = [
  {
    label: "lightweight aero bracket",
    constraints: {
      minYield: 250, maxDensity: 5.0, minServiceTempC: 150,
      designStress: 120, fabrication: ["machining", "additive_dmls"],
    },
    weights: { performance: 0.4, weight: 0.3, cost: 0.1, fatigue: 0.2 },
  },
  {
    label: "marine pump housing",
    constraints: {
      minYield: 200, maxDensity: 9.0, environment: ["marine"],
      fabrication: ["casting", "machining"], designStress: 90,
    },
    weights: { performance: 0.3, sustainability: 0.15, cost: 0.25, fatigue: 0.3 },
  },
  {
    label: "cost-driven structural",
    constraints: {
      minYield: 300, maxCostPerKg: 4.0, designStress: 160,
      fabrication: ["machining", "sheet_forming"],
    },
    weights: { cost: 0.45, performance: 0.3, fatigue: 0.2, weight: 0.05 },
  },
  {
    label: "high-temp manifold",
    constraints: {
      minYield: 220, minServiceTempC: 600, environment: ["high_temp"],
      designStress: 100,
    },
    weights: { performance: 0.45, fatigue: 0.25, cost: 0.15, weight: 0.15 },
  },
  {
    label: "sustainable consumer part",
    constraints: {
      minYield: 30, maxDensity: 2.0, fabrication: ["injection_molding"],
      designStress: 15,
    },
    weights: { sustainability: 0.4, cost: 0.3, weight: 0.2, performance: 0.1 },
  },
];

function buildWeights(p: Partial<ObjectiveWeights> = {}): ObjectiveWeights {
  return { ...DEFAULT_WEIGHTS, ...p };
}

// ---------------- procurement layer (unlocks) ----------------

/** Stable hash → deterministic supplier quotes per material. */
function hash(s: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
function rand(seed: number): () => number {
  let s = seed || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
}

type Quote = {
  supplier: string;
  unitCost: number;
  leadTimeDays: number;
  minOrderKg: number;
  totalCost: number;
  score: number; // procurement composite (cost + lead time + reliability)
  reliability: number;
};

const SUPPLIERS = ["AlloyWorks", "NorthMills", "DeltaSource", "OrbitMat", "PrimeCast"];

function quotesFor(matId: string, basePrice: number, qtyKg: number): Quote[] {
  const r = rand(hash(matId));
  const k = 3 + Math.floor(r() * 2); // 3–4 suppliers
  const out: Quote[] = [];
  for (let i = 0; i < k; i++) {
    const supplier = SUPPLIERS[(hash(matId + i) >>> 0) % SUPPLIERS.length];
    const premium = 0.85 + r() * 0.55; // 0.85–1.40× base
    const unitCost = basePrice * premium;
    const leadTimeDays = 5 + Math.floor(r() * 35);
    const minOrderKg = 5 + Math.floor(r() * 45);
    const reliability = 0.7 + r() * 0.28;
    const orderQty = Math.max(qtyKg, minOrderKg);
    const totalCost = unitCost * orderQty;
    // Composite procurement score: lower cost, faster lead, higher reliability.
    const costScore = 1 / (1 + premium - 0.85);
    const leadScore = 1 - Math.min(1, (leadTimeDays - 5) / 35);
    const score = 0.5 * costScore + 0.25 * leadScore + 0.25 * reliability;
    out.push({ supplier, unitCost, leadTimeDays, minOrderKg, totalCost, score, reliability });
  }
  return out.sort((a, b) => b.score - a.score);
}

// ---------------- panel ----------------

export function MaterialGatingPanel() {
  const indexRef = useRef<CorpusIndex | null>(null);
  const evalRef = useRef<MatGateEvaluator | null>(null);
  if (!indexRef.current) {
    indexRef.current = buildIndex(BUILTIN_MATERIALS);
    evalRef.current = new MatGateEvaluator(DEFAULT_MAT_GATES);
  }
  const index = indexRef.current;
  const evaluator = evalRef.current!;

  const [running, setRunning] = useState(true);
  const [report, setReport] = useState<MatGateReport>(() => evaluator.report());
  const [topPick, setTopPick] = useState<MaterialScore | null>(null);
  const tickRef = useRef(0);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      for (let k = 0; k < 2; k++) {
        const sc = SCENARIOS[tickRef.current % SCENARIOS.length];
        tickRef.current++;
        const t0 = performance.now();
        const ranked = recommend(index, sc.constraints, buildWeights(sc.weights));
        const dt = performance.now() - t0;
        evaluator.record(ranked, dt);
        if (ranked[0]?.feasible) setTopPick(ranked[0]);
      }
      setReport(evaluator.report());
    }, 220);
    return () => clearInterval(id);
  }, [index, evaluator, running]);

  const reset = () => {
    evaluator.reset();
    setTopPick(null);
    setReport(evaluator.report());
    tickRef.current = 0;
  };

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Material Intelligence · Gating System
          <Badge variant="outline" className="text-[10px]">
            {report.unlocked ? "all gates passed" : "evaluating"}
          </Badge>
        </h3>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="sm" className="h-7 px-2 text-xs"
            onClick={() => setRunning((r) => !r)}>
            {running ? "pause" : "resume"}
          </Button>
          <Button variant="ghost" size="sm" className="h-7 px-2 text-xs"
            onClick={reset}>
            reset
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
        <GateCard
          letter="A" title="Recommendation Relevance"
          fmt={(v) => `${(v * 100).toFixed(1)}%`}
          fmtThr={(v) => `> ${(v * 100).toFixed(0)}%`}
          status={report.A} higherIsBetter
        />
        <GateCard
          letter="B" title="Constraint Satisfaction"
          fmt={(v) => `${(v * 100).toFixed(1)}%`}
          fmtThr={(v) => `> ${(v * 100).toFixed(0)}%`}
          status={report.B} higherIsBetter
        />
        <GateCard
          letter="C" title="Inference Latency (p95)"
          fmt={(v) => `${v.toFixed(2)} ms`}
          fmtThr={(v) => `< ${v} ms`}
          status={report.C}
        />
      </div>

      <div className="grid grid-cols-2 gap-2 text-[10px] tabular-nums text-muted-foreground sm:grid-cols-5">
        <Stat label="scenarios" v={report.scenarios} />
        <Stat label="lat p50" v={`${report.latency.p50.toFixed(1)} ms`} />
        <Stat label="lat p95" v={`${report.latency.p95.toFixed(1)} ms`} />
        <Stat label="lat max" v={`${report.latency.max.toFixed(1)} ms`} />
        <Stat label="topK" v={DEFAULT_MAT_GATES.topK} />
      </div>

      <ProcurementOptimization unlocked={report.unlocked} top={topPick} />
    </div>
  );
}

function GateCard({
  letter, title, status, fmt, fmtThr, higherIsBetter,
}: {
  letter: "A" | "B" | "C";
  title: string;
  status: MatGateReport["A"];
  fmt: (v: number) => string;
  fmtThr: (v: number) => string;
  higherIsBetter?: boolean;
}) {
  const tone = !status.ready
    ? "border-border bg-background/40 text-muted-foreground"
    : status.pass
      ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-300"
      : "border-destructive/60 bg-destructive/10 text-destructive";
  const ratio = higherIsBetter
    ? Math.min(1, status.value / Math.max(1e-9, status.threshold))
    : Math.min(1, status.value / Math.max(1e-9, status.threshold));
  return (
    <div className={`rounded border px-3 py-2 ${tone}`}>
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.16em] opacity-80">
        <span>Gate {letter}</span>
        <span>{!status.ready ? "warming" : status.pass ? "PASS" : "FAIL"}</span>
      </div>
      <div className="mt-0.5 text-[11px] opacity-90">{title}</div>
      <div className="mt-1 flex items-baseline justify-between">
        <span className="font-mono text-lg tabular-nums">{fmt(status.value)}</span>
        <span className="text-[10px] opacity-70">{fmtThr(status.threshold)}</span>
      </div>
      <div className="mt-1.5 h-1 w-full overflow-hidden rounded bg-background/40">
        <div
          className="h-full bg-current opacity-70 transition-[width]"
          style={{ width: `${Math.max(2, Math.round(ratio * 100))}%` }}
        />
      </div>
    </div>
  );
}

function Stat({ label, v }: { label: string; v: number | string }) {
  return (
    <div className="rounded border border-border/60 bg-background/40 px-2 py-1">
      <div className="text-[9px] uppercase tracking-[0.14em] opacity-70">{label}</div>
      <div className="font-mono">{v}</div>
    </div>
  );
}

function ProcurementOptimization({
  unlocked, top,
}: { unlocked: boolean; top: MaterialScore | null }) {
  const qtyKg = 250;
  const quotes = useMemo(() => {
    if (!unlocked || !top) return [];
    return quotesFor(top.material.id, top.material.costPerKg, qtyKg);
  }, [unlocked, top]);

  if (!unlocked) {
    return (
      <div className="rounded border border-dashed border-border bg-background/40 p-3 text-[11px] text-muted-foreground">
        <div className="flex items-center justify-between">
          <span className="uppercase tracking-[0.18em]">Procurement Optimization · locked</span>
          <span className="text-[10px] opacity-70">pass all gates to unlock</span>
        </div>
        <div className="mt-1 text-[10px] opacity-70">
          Cost-aware sourcing layer — composite ranking across supplier price,
          lead time, MOQ, and reliability for the top recommended material.
        </div>
      </div>
    );
  }
  if (!top) return null;

  const best = quotes[0];
  const baseline = quotes[quotes.length - 1];
  const savings = baseline ? (baseline.totalCost - best.totalCost) / baseline.totalCost : 0;

  return (
    <div className="rounded border border-emerald-500/40 bg-emerald-500/5 p-3">
      <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.18em] text-emerald-300">
        <span>Procurement Optimization · unlocked</span>
        <span className="opacity-70">
          {top.material.name} · {qtyKg} kg order
        </span>
      </div>

      <div className="mt-2 grid grid-cols-3 gap-2 text-[10px] tabular-nums">
        <div className="rounded border border-emerald-500/30 bg-background/40 px-2 py-1">
          <div className="opacity-70 uppercase tracking-[0.14em]">best supplier</div>
          <div className="font-mono text-emerald-200">{best.supplier}</div>
        </div>
        <div className="rounded border border-emerald-500/30 bg-background/40 px-2 py-1">
          <div className="opacity-70 uppercase tracking-[0.14em]">total landed</div>
          <div className="font-mono">${best.totalCost.toFixed(0)}</div>
        </div>
        <div className="rounded border border-emerald-500/30 bg-background/40 px-2 py-1">
          <div className="opacity-70 uppercase tracking-[0.14em]">vs worst</div>
          <div className="font-mono text-emerald-300">−{(savings * 100).toFixed(1)}%</div>
        </div>
      </div>

      <table className="mt-2 w-full text-[10px] tabular-nums">
        <thead className="text-[9px] uppercase tracking-[0.14em] text-emerald-300/70">
          <tr>
            <th className="text-left font-normal py-1">supplier</th>
            <th className="text-right font-normal">$ / kg</th>
            <th className="text-right font-normal">lead</th>
            <th className="text-right font-normal">MOQ</th>
            <th className="text-right font-normal">rel.</th>
            <th className="text-right font-normal">total</th>
            <th className="text-right font-normal">score</th>
          </tr>
        </thead>
        <tbody className="font-mono">
          {quotes.map((q, i) => (
            <tr key={i} className={i === 0 ? "text-emerald-200" : "text-muted-foreground"}>
              <td className="py-0.5">{q.supplier}</td>
              <td className="text-right">${q.unitCost.toFixed(2)}</td>
              <td className="text-right">{q.leadTimeDays}d</td>
              <td className="text-right">{q.minOrderKg}kg</td>
              <td className="text-right">{(q.reliability * 100).toFixed(0)}%</td>
              <td className="text-right">${q.totalCost.toFixed(0)}</td>
              <td className="text-right">{q.score.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
