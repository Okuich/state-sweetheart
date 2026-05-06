// Federated Industrial Learning System — UI panel
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  federatedRound,
  initGlobal,
  makeFederation,
  type ClientShard,
  type DPConfig,
  type GlobalModel,
  type RoundReport,
} from "@/lib/federated";

export function FederatedPanel() {
  const [numClients, setNumClients] = useState(5);
  const [samples, setSamples] = useState(120);
  const [seed, setSeed] = useState(7);
  const [lr, setLr] = useState(0.05);
  const [serverLR, setServerLR] = useState(1.0);
  const [localEpochs, setLocalEpochs] = useState(2);
  const [clipNorm, setClipNorm] = useState(1.0);
  const [noiseSigma, setNoiseSigma] = useState(0.05);
  const [epsPerRound, setEpsPerRound] = useState(0.4);
  const [budget, setBudget] = useState(8.0);

  const clients = useMemo<ClientShard[]>(
    () => makeFederation(numClients, samples, seed),
    [numClients, samples, seed]
  );
  const [global, setGlobal] = useState<GlobalModel>(() => initGlobal(seed + 1));
  const [history, setHistory] = useState<RoundReport[]>([]);
  const [running, setRunning] = useState(false);
  const tickRef = useRef<number | null>(null);

  const reset = () => {
    setRunning(false);
    setGlobal(initGlobal(seed + 1));
    setHistory([]);
  };

  // re-init when federation changes
  useEffect(reset, [numClients, samples, seed]); // eslint-disable-line react-hooks/exhaustive-deps

  const cfg: DPConfig = useMemo(() => ({
    clipNorm, noiseSigma, epsilonPerRound: epsPerRound,
  }), [clipNorm, noiseSigma, epsPerRound]);

  const stepOnce = () => {
    setGlobal((g) => {
      const remaining = budget - g.round * cfg.epsilonPerRound;
      if (remaining < cfg.epsilonPerRound) {
        setRunning(false);
        return g;
      }
      const { next, report } = federatedRound(
        g, clients, cfg,
        { lr, localEpochs, batch: 16, serverLR, budgetRemaining: remaining },
        seed * 31 + g.round + 1
      );
      setHistory((h) => [...h, report].slice(-200));
      return next;
    });
  };

  // streaming loop
  useEffect(() => {
    if (!running) {
      if (tickRef.current) { window.clearInterval(tickRef.current); tickRef.current = null; }
      return;
    }
    tickRef.current = window.setInterval(stepOnce, 350);
    return () => { if (tickRef.current) window.clearInterval(tickRef.current); };
  }, [running, clients, cfg, lr, serverLR, localEpochs, budget, seed]); // eslint-disable-line react-hooks/exhaustive-deps

  const last = history.at(-1);
  const lossPath = useMemo(() => sparkline(history.map((r) => r.globalLoss)), [history]);
  const noisePath = useMemo(() => sparkline(history.map((r) => r.noiseEnergy)), [history]);

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-baseline justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
            module · federated-learning
          </div>
          <h2 className="font-display text-2xl md:text-3xl text-glow">
            Federated <span className="text-primary">Industrial</span> Learning
          </h2>
          <p className="text-xs text-muted-foreground max-w-xl mt-1">
            Each company trains locally on private geometry + fab telemetry.
            Only DP-noised, L2-clipped, secret-shared gradient deltas leave
            the device. Server aggregates with secure-sum and applies FedAvg.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={stepOnce} className="uppercase tracking-[0.16em] text-[10px]">
            step · 1 round
          </Button>
          <Button
            variant={running ? "default" : "outline"}
            onClick={() => setRunning((s) => !s)}
            className={`uppercase tracking-[0.16em] text-[10px] ${running ? "bg-accent text-accent-foreground" : ""}`}
          >
            {running ? "training…" : "auto-train"}
          </Button>
          <Button variant="outline" onClick={reset} className="uppercase tracking-[0.16em] text-[10px]">
            reset
          </Button>
        </div>
      </header>

      {/* IP firewall banner */}
      <div className="rounded-md border border-border bg-background/60 px-3 py-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground flex flex-wrap gap-x-6 gap-y-1">
        <span className="text-destructive">never leaves device:</span>
        <span>STEP files</span><span>·</span>
        <span>CAD geometry</span><span>·</span>
        <span>fab blueprints</span><span>·</span>
        <span>raw telemetry</span>
        <span className="ml-auto text-primary">leaves device: clipped + noised gradient deltas only</span>
      </div>

      {/* hyperparams */}
      <div className="grid gap-3 md:grid-cols-5 text-xs">
        <NumInput label="clients · N" value={numClients} onChange={setNumClients} min={2} max={8} step={1} />
        <NumInput label="samples / client" value={samples} onChange={setSamples} min={20} max={1000} step={10} />
        <NumInput label="seed"        value={seed} onChange={setSeed} min={0} max={9999} step={1} />
        <NumInput label="local lr"    value={lr} onChange={setLr} min={0.001} max={0.5} step={0.005} />
        <NumInput label="server lr"   value={serverLR} onChange={setServerLR} min={0.1} max={2} step={0.05} />
        <NumInput label="local epochs" value={localEpochs} onChange={setLocalEpochs} min={1} max={10} step={1} />
        <NumInput label="DP · clip"   value={clipNorm} onChange={setClipNorm} min={0.1} max={5} step={0.1} />
        <NumInput label="DP · σ noise" value={noiseSigma} onChange={setNoiseSigma} min={0} max={0.5} step={0.005} />
        <NumInput label="ε / round"   value={epsPerRound} onChange={setEpsPerRound} min={0} max={2} step={0.05} />
        <NumInput label="ε budget"    value={budget} onChange={setBudget} min={0.5} max={30} step={0.5} />
      </div>

      {/* round-level metrics */}
      <div className="grid gap-3 md:grid-cols-4">
        <Card label="round">
          <div className="font-mono text-2xl text-foreground/90">{global.round}</div>
          <div className="text-[10px] text-muted-foreground">aggregated updates so far</div>
        </Card>
        <Card label="global loss">
          <Curve path={lossPath} stroke="hsl(var(--destructive))" />
          <div className="font-mono text-[11px] tabular-nums text-foreground/85 mt-1">
            {last ? last.globalLoss.toFixed(4) : "—"}
          </div>
        </Card>
        <Card label="DP noise energy">
          <Curve path={noisePath} stroke="hsl(var(--accent))" />
          <div className="font-mono text-[11px] tabular-nums text-foreground/85 mt-1">
            {last ? last.noiseEnergy.toFixed(4) : "—"}
          </div>
        </Card>
        <Card label="ε privacy budget">
          <BudgetBar
            spent={last?.epsilonSpent ?? 0}
            total={budget}
          />
        </Card>
      </div>

      {/* per-client */}
      <div className="rounded-lg border border-border bg-background/40 p-3">
        <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-2">
          per-client · this round
        </div>
        {last ? (
          <table className="w-full text-[11px] font-mono">
            <thead>
              <tr className="text-muted-foreground text-[9px] uppercase tracking-[0.18em]">
                <th className="text-left">company</th>
                <th className="text-right">loss · before</th>
                <th className="text-right">loss · after</th>
                <th className="text-right">‖clipped grad‖</th>
                <th className="text-right">Δloss</th>
              </tr>
            </thead>
            <tbody>
              {last.perClient.map((c) => {
                const d = c.lossAfter - c.lossBefore;
                return (
                  <tr key={c.id} className="border-t border-border/40">
                    <td className="py-1 text-foreground/90">{c.name}</td>
                    <td className="text-right tabular-nums">{c.lossBefore.toFixed(3)}</td>
                    <td className="text-right tabular-nums">{c.lossAfter.toFixed(3)}</td>
                    <td className="text-right tabular-nums text-muted-foreground">{c.gradNorm.toFixed(3)}</td>
                    <td className={`text-right tabular-nums ${d < 0 ? "text-primary" : "text-destructive"}`}>
                      {d > 0 ? "+" : ""}{d.toFixed(3)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <div className="text-[11px] text-muted-foreground">no rounds yet — step or auto-train.</div>
        )}
      </div>
    </div>
  );
}

function BudgetBar({ spent, total }: { spent: number; total: number }) {
  const pct = Math.min(1, spent / Math.max(1e-9, total));
  const remaining = Math.max(0, total - spent);
  const tone = pct > 0.85 ? "bg-destructive" : pct > 0.6 ? "bg-accent" : "bg-primary";
  return (
    <>
      <div className="h-2 rounded-sm bg-muted overflow-hidden">
        <div className={`h-full ${tone}`} style={{ width: `${pct * 100}%` }} />
      </div>
      <div className="mt-1 flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>spent · {spent.toFixed(2)} ε</span>
        <span>remain · {remaining.toFixed(2)} ε</span>
      </div>
    </>
  );
}

function Card({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground mb-1">{label}</div>
      {children}
    </div>
  );
}
function Curve({ path, stroke }: { path: string; stroke: string }) {
  return (
    <svg viewBox="0 0 100 28" className="w-full h-12">
      <path d={path} fill="none" stroke={stroke} strokeWidth={1.2} />
    </svg>
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
        className="rounded-md border border-border bg-background/60 px-2 py-1 font-mono text-xs outline-none focus:ring-1 focus:ring-primary" />
    </label>
  );
}
function sparkline(values: number[]): string {
  if (values.length === 0) return "";
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const W = 100, H = 28;
  return values.map((v, i) => {
    const x = (i / Math.max(1, values.length - 1)) * W;
    const y = H - ((v - min) / span) * H;
    return `${i === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`;
  }).join(" ");
}
