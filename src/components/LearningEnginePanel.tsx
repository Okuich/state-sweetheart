// Physics Learning Engine — UI
// Trains three heads (deformation prior, stress risk, manufacturability)
// from a synthetic dataset of geometry + sim + fab + QA features.

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  D,
  FEATURE_NAMES,
  evaluate,
  featureImportance,
  initHeads,
  makeDataset,
  predict,
  suggestTweaks,
  train,
  type Heads,
  type Sample,
  type TrainStats,
} from "@/lib/learningEngine";
import { designBridge, type PublishedDesign } from "@/lib/geometryToLearning";

const fmt = (v: number, p = 3) =>
  Number.isFinite(v) ? v.toFixed(p) : "—";

export function LearningEnginePanel() {
  const [seed, setSeed] = useState(7);
  const [nTrain, setNTrain] = useState(800);
  const [nTest, setNTest] = useState(200);
  const [epochs, setEpochs] = useState(40);
  const [lr, setLr] = useState(0.08);
  const [running, setRunning] = useState(false);
  const [heads, setHeads] = useState<Heads>(() => initHeads(1));
  const [history, setHistory] = useState<TrainStats[]>([]);
  const [testStats, setTestStats] = useState<TrainStats | null>(null);
  const cancelRef = useRef(false);

  const train_ds = useMemo(() => makeDataset(nTrain, seed), [nTrain, seed]);
  const test_ds  = useMemo(() => makeDataset(nTest, seed + 1000), [nTest, seed]);

  // candidate design (for suggestions/predictions)
  const [candidate, setCandidate] = useState<number[]>(() =>
    new Array(D).fill(0.5)
  );

  const fit = async () => {
    setRunning(true);
    cancelRef.current = false;
    const h = initHeads(seed + 31);
    const hist: TrainStats[] = [];
    // run epoch-by-epoch with yield so the UI can paint
    for (let e = 0; e < epochs; e++) {
      if (cancelRef.current) break;
      train(h, train_ds, {
        epochs: 1, lr, batch: 32, l2: 1e-4, momentum: 0.9,
      });
      const s = evaluate(h, train_ds, e);
      hist.push(s);
      setHistory([...hist]);
      setHeads({ ...h });
      // every few epochs let the browser breathe
      if (e % 2 === 0) await new Promise((r) => setTimeout(r, 0));
    }
    setTestStats(evaluate(h, test_ds, epochs));
    setRunning(false);
  };

  const reset = () => {
    cancelRef.current = true;
    setHeads(initHeads(1));
    setHistory([]);
    setTestStats(null);
  };

  const importance = useMemo(() => featureImportance(heads), [heads]);
  const pred = useMemo(() => predict(heads, candidate), [heads, candidate]);
  const tweaks = useMemo(
    () => suggestTweaks(heads, candidate, 6),
    [heads, candidate]
  );

  // training-curve sparkline
  const lossPath = useMemo(() => sparkline(history.map((h) => h.loss)), [history]);
  const accPath  = useMemo(
    () => sparkline(history.map((h) => (h.accStress + h.accFab) / 2), { invert: false }),
    [history]
  );

  // pre-train evaluation for "before vs after"
  const baseline = useMemo(
    () => evaluate(initHeads(1), test_ds, 0),
    [test_ds]
  );

  useEffect(() => () => { cancelRef.current = true; }, []);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
            module · learning-engine
          </div>
          <h2 className="font-display text-2xl md:text-3xl text-glow">
            Physics <span className="text-primary">Learning</span> Engine
          </h2>
          <p className="text-xs text-muted-foreground max-w-xl mt-1">
            Continuously improving priors over geometry, simulation, fabrication
            and inspection. Three coupled heads — deformation, stress risk,
            manufacturability — trained by SGD on a synthetic corpus.
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant={running ? "secondary" : "default"}
            disabled={running}
            onClick={fit}
            className="uppercase tracking-[0.16em] text-[10px]"
          >
            {running ? "training…" : "fit priors"}
          </Button>
          <Button
            variant="outline"
            onClick={reset}
            className="uppercase tracking-[0.16em] text-[10px]"
          >
            reset
          </Button>
        </div>
      </header>

      {/* Hyperparameters */}
      <div className="grid gap-3 md:grid-cols-5 text-xs">
        <NumInput label="seed"        value={seed}    onChange={setSeed}    min={0}    max={9999} step={1} />
        <NumInput label="N · train"   value={nTrain}  onChange={setNTrain}  min={50}   max={4000} step={50} />
        <NumInput label="N · test"    value={nTest}   onChange={setNTest}   min={50}   max={2000} step={50} />
        <NumInput label="epochs"      value={epochs}  onChange={setEpochs}  min={1}    max={300}  step={1} />
        <NumInput label="lr"          value={lr}      onChange={setLr}      min={0.001} max={1}   step={0.005} />
      </div>

      {/* Training curves + final scores */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card label="loss · train">
          <Curve path={lossPath} stroke="hsl(var(--destructive))" />
          <Legend
            rows={[
              ["epoch",     `${history.length} / ${epochs}`],
              ["mse · deform", fmt(history.at(-1)?.mseDeform ?? NaN)],
              ["bce · stress", fmt(history.at(-1)?.bceStress ?? NaN)],
              ["bce · fab",    fmt(history.at(-1)?.bceFab ?? NaN)],
            ]}
          />
        </Card>
        <Card label="accuracy · train (mean of stress + fab)">
          <Curve path={accPath} stroke="hsl(var(--primary))" />
          <Legend
            rows={[
              ["acc · stress", fmt(history.at(-1)?.accStress ?? NaN)],
              ["acc · fab",    fmt(history.at(-1)?.accFab ?? NaN)],
              ["test · loss",  testStats ? fmt(testStats.loss) : "—"],
              ["baseline · loss", fmt(baseline.loss)],
            ]}
          />
        </Card>
      </div>

      {/* Feature importance */}
      <Card label="feature importance · |w| across heads">
        <div className="space-y-1">
          {importance
            .slice()
            .sort((a, b) => b.total - a.total)
            .map((r) => (
              <div key={r.name} className="grid grid-cols-[140px_1fr_60px] items-center gap-3 text-[11px]">
                <span className="font-mono text-muted-foreground">{r.name}</span>
                <div className="h-2 rounded-sm bg-muted overflow-hidden">
                  <div
                    className="h-full bg-primary"
                    style={{ width: `${(r.normalized * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="font-mono tabular-nums text-right text-foreground/80">
                  {fmt(r.total, 2)}
                </span>
              </div>
            ))}
        </div>
      </Card>

      {/* Candidate design + suggestions */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card label="candidate · design vector (normalized 0..1)">
          <div className="space-y-2 max-h-72 overflow-auto pr-1">
            {FEATURE_NAMES.map((n, i) => (
              <div key={n} className="flex items-center gap-2 text-[11px]">
                <span className="w-32 font-mono text-muted-foreground">{n}</span>
                <input
                  type="range"
                  min={0} max={1} step={0.01}
                  value={candidate[i]}
                  onChange={(e) => {
                    const v = Number(e.target.value);
                    setCandidate((c) => c.map((x, k) => (k === i ? v : x)));
                  }}
                  className="flex-1 accent-primary"
                />
                <span className="w-10 font-mono tabular-nums text-right">
                  {candidate[i].toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 grid grid-cols-3 gap-2 text-[11px]">
            <Stat label="deform"   value={fmt(pred.deform)}             tone="muted" />
            <Stat label="P[stress]" value={fmt(pred.stressP)}            tone={pred.stressP > 0.5 ? "danger" : "ok"} />
            <Stat label="P[fab ok]" value={fmt(pred.fabP)}               tone={pred.fabP > 0.5 ? "ok" : "danger"} />
          </div>
        </Card>

        <Card label="optimization heuristic · suggested tweaks">
          {tweaks.length === 0 ? (
            <div className="text-[11px] text-muted-foreground">
              No improving tweaks found at current point. Train more or move the
              candidate.
            </div>
          ) : (
            <ul className="space-y-1.5">
              {tweaks.map((t) => (
                <li
                  key={t.feature}
                  className="grid grid-cols-[1fr_70px_60px_90px] items-center gap-2 text-[11px] font-mono"
                >
                  <span className="text-foreground/90">{t.feature}</span>
                  <span className={t.delta > 0 ? "text-primary" : "text-destructive"}>
                    {t.delta > 0 ? "+" : ""}{t.delta.toFixed(3)}
                  </span>
                  <span className="text-muted-foreground">{fmt(t.gain, 4)}</span>
                  <span className="text-[9px] uppercase tracking-[0.16em] text-muted-foreground/80">
                    {t.rationale}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      {/* Test scores */}
      {testStats && (
        <Card label="held-out test set">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-[11px]">
            <Stat label="mse · deform" value={fmt(testStats.mseDeform)} tone="muted" />
            <Stat label="bce · stress" value={fmt(testStats.bceStress)} tone="muted" />
            <Stat label="acc · stress" value={fmt(testStats.accStress)} tone="ok" />
            <Stat label="acc · fab"    value={fmt(testStats.accFab)}    tone="ok" />
          </div>
        </Card>
      )}
    </div>
  );
}

// ─── tiny presentational atoms ───────────────────────────────

function Card({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
        {label}
      </div>
      {children}
    </div>
  );
}

function NumInput({
  label, value, onChange, min, max, step,
}: { label: string; value: number; onChange: (v: number) => void; min: number; max: number; step: number }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</span>
      <input
        type="number" value={value} min={min} max={max} step={step}
        onChange={(e) => onChange(Number(e.target.value))}
        className="rounded-md border border-border bg-background/60 px-2 py-1 font-mono text-xs text-foreground outline-none focus:ring-1 focus:ring-primary"
      />
    </label>
  );
}

function Stat({
  label, value, tone,
}: { label: string; value: string; tone: "ok" | "danger" | "muted" }) {
  const cls =
    tone === "ok" ? "text-primary" :
    tone === "danger" ? "text-destructive" : "text-foreground/80";
  return (
    <div className="rounded-md border border-border bg-card/60 px-2 py-1.5">
      <div className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</div>
      <div className={`font-mono tabular-nums ${cls}`}>{value}</div>
    </div>
  );
}

function Curve({ path, stroke }: { path: string; stroke: string }) {
  return (
    <svg viewBox="0 0 100 32" className="w-full h-16">
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.2} />
    </svg>
  );
}

function Legend({ rows }: { rows: [string, string][] }) {
  return (
    <div className="mt-2 grid grid-cols-2 gap-x-4 gap-y-0.5 text-[10px] font-mono">
      {rows.map(([k, v]) => (
        <div key={k} className="flex justify-between">
          <span className="text-muted-foreground">{k}</span>
          <span className="text-foreground/80 tabular-nums">{v}</span>
        </div>
      ))}
    </div>
  );
}

function sparkline(values: number[], opts: { invert?: boolean } = {}): string {
  if (values.length === 0) return "";
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const W = 100, H = 32;
  return values
    .map((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * W;
      const norm = (v - min) / span;
      const y = opts.invert ? norm * H : H - norm * H;
      return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}
