/**
 * anomalyAlerts.ts
 *
 * Configurable alert rules over the simulator's telemetry stream.
 *
 * Each rule watches one metric (energy_drift_pct, constraint_l2,
 * divergence_risk, …), applies an operator/threshold, optionally requires
 * the condition to hold for N consecutive samples (debounce), and fires
 * an Alert with a severity. Alerts are de-duplicated per rule until the
 * condition clears (hysteresis), so a long excursion produces ONE alert
 * with rolling stats — not one per sample.
 *
 * The engine is pure & deterministic: same inputs ⇒ same alerts.
 */

export type AlertMetric =
  | "energy_drift_pct"
  | "constraint_l2"
  | "divergence_risk"
  | "velocity_max"
  | "nan_count";

export type AlertOp = ">" | ">=" | "<" | "<=" | "==";

export type AlertSeverity = "info" | "warn" | "critical";

export interface AlertRule {
  id: string;
  /** Display name. */
  name: string;
  metric: AlertMetric;
  op: AlertOp;
  threshold: number;
  severity: AlertSeverity;
  /** N consecutive samples crossing the threshold before firing. ≥1. */
  consecutive?: number;
  /** Hysteresis: condition must clear by this fraction of threshold to reset. */
  clearMargin?: number;
  enabled?: boolean;
}

export interface TelemetrySample {
  t: number;
  energy_drift_pct?: number;
  constraint_l2?: number;
  divergence_risk?: number;
  velocity_max?: number;
  nan_count?: number;
}

export interface Alert {
  /** Stable id = ruleId + "@" + firstFiredAt */
  id: string;
  ruleId: string;
  ruleName: string;
  metric: AlertMetric;
  severity: AlertSeverity;
  threshold: number;
  op: AlertOp;
  /** Sample t when the rule first fired. */
  firstFiredAt: number;
  /** Most recent sample t while the alert was open. */
  lastSeenAt: number;
  /** Worst (max for >, min for <) value observed during the excursion. */
  peakValue: number;
  /** Number of samples included in this excursion. */
  sampleCount: number;
  /** True once the value has dropped back below threshold-clearMargin. */
  cleared: boolean;
  /** Sample t when cleared (if cleared). */
  clearedAt?: number;
}

interface RuleState {
  consecutive: number;
  open: Alert | null;
}

export class AnomalyAlertEngine {
  private rules: AlertRule[] = [];
  private state = new Map<string, RuleState>();
  private history: Alert[] = [];
  private historyCap: number;

  constructor(opts: { historyCap?: number } = {}) {
    this.historyCap = Math.max(8, opts.historyCap ?? 200);
  }

  // ---- rule CRUD ----
  setRules(rules: AlertRule[]): void {
    this.rules = rules.map(normalizeRule);
    // drop state for removed rules; preserve for surviving ones
    const ids = new Set(this.rules.map((r) => r.id));
    for (const k of [...this.state.keys()]) if (!ids.has(k)) this.state.delete(k);
    for (const r of this.rules) if (!this.state.has(r.id)) this.state.set(r.id, { consecutive: 0, open: null });
  }
  getRules(): AlertRule[] { return this.rules.slice(); }
  upsertRule(rule: AlertRule): void {
    const r = normalizeRule(rule);
    const i = this.rules.findIndex((x) => x.id === r.id);
    if (i >= 0) this.rules[i] = r; else this.rules.push(r);
    if (!this.state.has(r.id)) this.state.set(r.id, { consecutive: 0, open: null });
  }
  removeRule(id: string): void {
    this.rules = this.rules.filter((r) => r.id !== id);
    this.state.delete(id);
  }

  // ---- ingest ----
  /** Feed one telemetry sample. Returns alerts that newly fired this call. */
  ingest(sample: TelemetrySample): Alert[] {
    const fired: Alert[] = [];
    for (const rule of this.rules) {
      if (rule.enabled === false) continue;
      const v = sample[rule.metric];
      if (v === undefined || !Number.isFinite(v)) continue;
      const st = this.state.get(rule.id)!;
      const triggered = compare(v, rule.op, rule.threshold);

      if (triggered) {
        st.consecutive++;
        if (st.consecutive >= (rule.consecutive ?? 1)) {
          if (!st.open) {
            st.open = {
              id: `${rule.id}@${sample.t}`,
              ruleId: rule.id,
              ruleName: rule.name,
              metric: rule.metric,
              severity: rule.severity,
              threshold: rule.threshold,
              op: rule.op,
              firstFiredAt: sample.t,
              lastSeenAt: sample.t,
              peakValue: v,
              sampleCount: 1,
              cleared: false,
            };
            this.pushHistory(st.open);
            fired.push(st.open);
          } else {
            st.open.lastSeenAt = sample.t;
            st.open.sampleCount++;
            st.open.peakValue = isWorse(rule.op, v, st.open.peakValue) ? v : st.open.peakValue;
          }
        }
      } else {
        st.consecutive = 0;
        if (st.open && hasCleared(v, rule)) {
          st.open.cleared = true;
          st.open.clearedAt = sample.t;
          st.open = null;
        }
      }
    }
    return fired;
  }

  /** All alerts currently firing (not yet cleared). */
  active(): Alert[] {
    return [...this.state.values()].map((s) => s.open).filter((a): a is Alert => !!a);
  }

  /** Rolling history (oldest → newest). */
  log(): Alert[] { return this.history.slice(); }

  reset(): void {
    for (const s of this.state.values()) { s.consecutive = 0; s.open = null; }
    this.history = [];
  }

  private pushHistory(a: Alert): void {
    this.history.push(a);
    if (this.history.length > this.historyCap) this.history.shift();
  }
}

// ---------- helpers ----------

function compare(v: number, op: AlertOp, th: number): boolean {
  switch (op) {
    case ">":  return v >  th;
    case ">=": return v >= th;
    case "<":  return v <  th;
    case "<=": return v <= th;
    case "==": return v === th;
  }
}

function isWorse(op: AlertOp, candidate: number, current: number): boolean {
  if (op === ">" || op === ">=") return candidate > current;
  if (op === "<" || op === "<=") return candidate < current;
  return false;
}

function hasCleared(v: number, rule: AlertRule): boolean {
  const margin = rule.clearMargin ?? 0;
  // For ">" rules, need v < threshold * (1 - margin); for "<" rules, v > threshold * (1 + margin).
  if (rule.op === ">" || rule.op === ">=") {
    return v < rule.threshold * (1 - margin);
  }
  if (rule.op === "<" || rule.op === "<=") {
    return v > rule.threshold * (1 + margin);
  }
  return v !== rule.threshold;
}

function normalizeRule(r: AlertRule): AlertRule {
  return {
    enabled: true,
    consecutive: 1,
    clearMargin: 0.1,
    ...r,
    consecutive: Math.max(1, Math.floor(r.consecutive ?? 1)),
  };
}

// ---------- preset rules ----------

export function defaultAlertRules(): AlertRule[] {
  return [
    {
      id: "energy-drift-warn",
      name: "Energy drift > 2%",
      metric: "energy_drift_pct",
      op: ">", threshold: 2, severity: "warn",
      consecutive: 3, clearMargin: 0.25,
    },
    {
      id: "energy-drift-crit",
      name: "Energy drift > 8%",
      metric: "energy_drift_pct",
      op: ">", threshold: 8, severity: "critical",
      consecutive: 1, clearMargin: 0.25,
    },
    {
      id: "constraint-l2",
      name: "Constraint L2 > 1e-2",
      metric: "constraint_l2",
      op: ">", threshold: 1e-2, severity: "warn",
      consecutive: 2, clearMargin: 0.2,
    },
    {
      id: "divergence-risk",
      name: "Divergence risk > 0.7",
      metric: "divergence_risk",
      op: ">", threshold: 0.7, severity: "critical",
      consecutive: 1, clearMargin: 0.2,
    },
  ];
}
