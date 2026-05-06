import { describe, it, expect } from "vitest";
import { AnomalyAlertEngine, defaultAlertRules, type AlertRule } from "./anomalyAlerts";

const rule = (overrides: Partial<AlertRule> = {}): AlertRule => ({
  id: "r1",
  name: "test",
  metric: "energy_drift_pct",
  op: ">",
  threshold: 5,
  severity: "warn",
  consecutive: 1,
  clearMargin: 0.1,
  ...overrides,
});

describe("AnomalyAlertEngine", () => {
  it("fires when threshold crossed", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule()]);
    expect(e.ingest({ t: 1, energy_drift_pct: 1 })).toHaveLength(0);
    const fired = e.ingest({ t: 2, energy_drift_pct: 6 });
    expect(fired).toHaveLength(1);
    expect(fired[0].severity).toBe("warn");
    expect(e.active()).toHaveLength(1);
  });

  it("respects consecutive debounce", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule({ consecutive: 3 })]);
    expect(e.ingest({ t: 1, energy_drift_pct: 9 })).toHaveLength(0);
    expect(e.ingest({ t: 2, energy_drift_pct: 9 })).toHaveLength(0);
    expect(e.ingest({ t: 3, energy_drift_pct: 9 })).toHaveLength(1);
  });

  it("dedupes while open and tracks peak/sampleCount", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule()]);
    e.ingest({ t: 1, energy_drift_pct: 6 });
    e.ingest({ t: 2, energy_drift_pct: 9 });
    e.ingest({ t: 3, energy_drift_pct: 7 });
    const a = e.active()[0];
    expect(a.peakValue).toBe(9);
    expect(a.sampleCount).toBe(3);
    expect(e.log()).toHaveLength(1);
  });

  it("clears with hysteresis margin", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule({ clearMargin: 0.2 })]); // clears below 5*(1-0.2)=4
    e.ingest({ t: 1, energy_drift_pct: 6 });
    expect(e.active()).toHaveLength(1);
    e.ingest({ t: 2, energy_drift_pct: 4.5 }); // still inside margin band
    expect(e.active()).toHaveLength(1);
    e.ingest({ t: 3, energy_drift_pct: 3 }); // below clear band
    expect(e.active()).toHaveLength(0);
  });

  it("disabled rules do not fire", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule({ enabled: false })]);
    expect(e.ingest({ t: 1, energy_drift_pct: 99 })).toHaveLength(0);
  });

  it("ignores undefined / non-finite metrics", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule()]);
    expect(e.ingest({ t: 1 })).toHaveLength(0);
    expect(e.ingest({ t: 2, energy_drift_pct: NaN })).toHaveLength(0);
  });

  it("supports multiple metrics independently", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([
      rule({ id: "a", metric: "energy_drift_pct", threshold: 5 }),
      rule({ id: "b", metric: "constraint_l2", threshold: 1e-3, severity: "critical" }),
    ]);
    const fired = e.ingest({ t: 1, energy_drift_pct: 10, constraint_l2: 1e-2 });
    expect(fired).toHaveLength(2);
    const sevs = fired.map((f) => f.severity).sort();
    expect(sevs).toEqual(["critical", "warn"]);
  });

  it("upsertRule and removeRule", () => {
    const e = new AnomalyAlertEngine();
    e.upsertRule(rule());
    e.upsertRule(rule({ threshold: 10 })); // same id → replace
    expect(e.getRules()[0].threshold).toBe(10);
    e.removeRule("r1");
    expect(e.getRules()).toHaveLength(0);
  });

  it("reset clears state and history", () => {
    const e = new AnomalyAlertEngine();
    e.setRules([rule()]);
    e.ingest({ t: 1, energy_drift_pct: 9 });
    e.reset();
    expect(e.active()).toHaveLength(0);
    expect(e.log()).toHaveLength(0);
  });

  it("default rules are well-formed", () => {
    const rules = defaultAlertRules();
    expect(rules.length).toBeGreaterThan(0);
    for (const r of rules) {
      expect(r.id).toBeTruthy();
      expect(["info", "warn", "critical"]).toContain(r.severity);
    }
  });
});
