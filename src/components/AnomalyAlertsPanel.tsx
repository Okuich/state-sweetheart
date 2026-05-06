import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  AnomalyAlertEngine, defaultAlertRules,
  type AlertRule, type AlertMetric, type AlertOp,
  type AlertSeverity, type Alert, type TelemetrySample,
} from "@/lib/anomalyAlerts";

const METRICS: { value: AlertMetric; label: string }[] = [
  { value: "energy_drift_pct", label: "energy drift (%)" },
  { value: "constraint_l2",    label: "constraint L2" },
  { value: "divergence_risk",  label: "divergence risk" },
  { value: "velocity_max",     label: "velocity max" },
  { value: "nan_count",        label: "NaN count" },
];
const OPS: AlertOp[] = [">", ">=", "<", "<=", "=="];
const SEVS: AlertSeverity[] = ["info", "warn", "critical"];

const sevColor: Record<AlertSeverity, string> = {
  info: "text-sky-300 border-sky-500/40 bg-sky-500/10",
  warn: "text-amber-300 border-amber-500/40 bg-amber-500/10",
  critical: "text-destructive border-destructive/50 bg-destructive/10",
};

export interface AnomalyAlertsPanelProps {
  /** Optional live telemetry feed; if absent, panel uses a synthetic source. */
  sample?: TelemetrySample;
}

export function AnomalyAlertsPanel({ sample }: AnomalyAlertsPanelProps) {
  const engineRef = useRef<AnomalyAlertEngine | null>(null);
  if (!engineRef.current) {
    engineRef.current = new AnomalyAlertEngine();
    engineRef.current.setRules(defaultAlertRules());
  }
  const engine = engineRef.current;

  const [rules, setRules] = useState<AlertRule[]>(engine.getRules());
  const [, force] = useState(0);
  const refresh = () => force((n) => n + 1);

  // Synthetic ticker so the panel is useful even without a live feed.
  const tickRef = useRef(0);
  useEffect(() => {
    if (sample) {
      engine.ingest(sample);
      refresh();
      return;
    }
    const id = setInterval(() => {
      tickRef.current++;
      const t = tickRef.current;
      engine.ingest({
        t,
        energy_drift_pct: 0.5 + 4 * Math.abs(Math.sin(t / 11)) + (t % 73 === 0 ? 12 : 0),
        constraint_l2:    1e-3 * (1 + Math.abs(Math.sin(t / 7))) + (t % 97 === 0 ? 0.05 : 0),
        divergence_risk:  0.2 + 0.3 * Math.abs(Math.sin(t / 13)) + (t % 53 === 0 ? 0.6 : 0),
        velocity_max:     5 + 2 * Math.sin(t / 5),
        nan_count:        0,
      });
      refresh();
    }, 250);
    return () => clearInterval(id);
  }, [sample, engine]);

  const persistRules = (next: AlertRule[]) => {
    setRules(next);
    engine.setRules(next);
    refresh();
  };

  const updateRule = (id: string, patch: Partial<AlertRule>) => {
    persistRules(rules.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  };

  const addRule = () => {
    const id = `rule-${Date.now().toString(36)}`;
    persistRules([
      ...rules,
      {
        id, name: "new rule",
        metric: "energy_drift_pct", op: ">", threshold: 5,
        severity: "warn", consecutive: 1, clearMargin: 0.1, enabled: true,
      },
    ]);
  };

  const removeRule = (id: string) => persistRules(rules.filter((r) => r.id !== id));

  const active = useMemo(() => engine.active(), [engine, rules, sample, tickRef.current]);
  const recent = useMemo(() => engine.log().slice(-20).reverse(), [engine, rules, sample, tickRef.current]);

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          Anomaly alert rules
        </h3>
        <div className="flex items-center gap-1">
          <Button variant="outline" size="sm" onClick={addRule} className="h-7 px-2 text-xs">
            + rule
          </Button>
          <Button
            variant="ghost" size="sm" className="h-7 px-2 text-xs"
            onClick={() => { engine.reset(); refresh(); }}
          >
            reset
          </Button>
        </div>
      </div>

      {active.length > 0 && (
        <div className="space-y-1">
          {active.map((a) => <AlertCard key={a.id} a={a} />)}
        </div>
      )}

      <div className="rounded border border-border/60 bg-background/40">
        <div className="grid grid-cols-[1.4fr_1.2fr_0.5fr_0.9fr_0.9fr_0.5fr_auto] gap-1 px-2 py-1 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
          <span>name</span><span>metric</span><span>op</span><span>threshold</span>
          <span>severity</span><span>n×</span><span></span>
        </div>
        <div className="divide-y divide-border/40">
          {rules.map((r) => (
            <RuleRow
              key={r.id}
              rule={r}
              onChange={(patch) => updateRule(r.id, patch)}
              onRemove={() => removeRule(r.id)}
            />
          ))}
          {rules.length === 0 && (
            <div className="p-3 text-[11px] italic text-muted-foreground">No rules. Click “+ rule”.</div>
          )}
        </div>
      </div>

      <div className="rounded border border-border/60 bg-background/40 p-2">
        <div className="mb-1 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
          recent alerts ({recent.length})
        </div>
        {recent.length === 0 ? (
          <div className="text-[11px] italic text-muted-foreground">None yet.</div>
        ) : (
          <div className="space-y-1 max-h-56 overflow-y-auto">
            {recent.map((a) => <AlertCard key={a.id} a={a} compact />)}
          </div>
        )}
      </div>
    </div>
  );
}

function RuleRow({
  rule, onChange, onRemove,
}: {
  rule: AlertRule;
  onChange: (patch: Partial<AlertRule>) => void;
  onRemove: () => void;
}) {
  return (
    <div className="grid grid-cols-[1.4fr_1.2fr_0.5fr_0.9fr_0.9fr_0.5fr_auto] items-center gap-1 px-2 py-1.5 text-[11px]">
      <Input
        value={rule.name}
        onChange={(e) => onChange({ name: e.target.value })}
        className="h-7 text-[11px]"
      />
      <Select value={rule.metric} onValueChange={(v) => onChange({ metric: v as AlertMetric })}>
        <SelectTrigger className="h-7 text-[11px]"><SelectValue /></SelectTrigger>
        <SelectContent>
          {METRICS.map((m) => (
            <SelectItem key={m.value} value={m.value} className="text-[11px]">{m.label}</SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Select value={rule.op} onValueChange={(v) => onChange({ op: v as AlertOp })}>
        <SelectTrigger className="h-7 text-[11px]"><SelectValue /></SelectTrigger>
        <SelectContent>
          {OPS.map((o) => <SelectItem key={o} value={o} className="text-[11px]">{o}</SelectItem>)}
        </SelectContent>
      </Select>
      <Input
        type="number"
        value={rule.threshold}
        step="any"
        onChange={(e) => onChange({ threshold: Number(e.target.value) })}
        className="h-7 text-[11px] tabular-nums"
      />
      <Select value={rule.severity} onValueChange={(v) => onChange({ severity: v as AlertSeverity })}>
        <SelectTrigger className="h-7 text-[11px]"><SelectValue /></SelectTrigger>
        <SelectContent>
          {SEVS.map((s) => <SelectItem key={s} value={s} className="text-[11px]">{s}</SelectItem>)}
        </SelectContent>
      </Select>
      <Input
        type="number"
        min={1}
        value={rule.consecutive ?? 1}
        onChange={(e) => onChange({ consecutive: Math.max(1, Number(e.target.value) | 0) })}
        className="h-7 text-[11px] tabular-nums"
      />
      <div className="flex items-center gap-1">
        <button
          onClick={() => onChange({ enabled: !(rule.enabled ?? true) })}
          className={`h-7 rounded px-2 text-[10px] uppercase tracking-[0.14em] ${
            rule.enabled ?? true
              ? "border border-primary/40 text-primary"
              : "border border-border text-muted-foreground"
          }`}
        >
          {(rule.enabled ?? true) ? "on" : "off"}
        </button>
        <button
          onClick={onRemove}
          className="h-7 rounded px-2 text-[10px] uppercase tracking-[0.14em] border border-border text-muted-foreground hover:text-destructive hover:border-destructive/40"
          aria-label="remove rule"
        >
          ×
        </button>
      </div>
    </div>
  );
}

function AlertCard({ a, compact }: { a: Alert; compact?: boolean }) {
  return (
    <div className={`rounded border px-2 py-1.5 text-[11px] ${sevColor[a.severity]}`}>
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium truncate">{a.ruleName}</span>
        <span className="font-mono text-[10px] opacity-80">
          {a.metric} {a.op} {a.threshold}
        </span>
      </div>
      <div className="mt-0.5 flex justify-between text-[10px] opacity-90 tabular-nums">
        <span>peak {a.peakValue.toExponential(2)}</span>
        <span>n={a.sampleCount}</span>
        {!compact && <span>t={a.firstFiredAt}…{a.lastSeenAt}</span>}
        <span>{a.cleared ? `cleared@${a.clearedAt}` : "active"}</span>
      </div>
    </div>
  );
}
