// Runtime Stability Guardrails
// ─────────────────────────────────────────────────────────────
// Watchdog that observes a synthetic simulation tape, flags
// instability (energy drift, NaN, constraint violation, partition
// blow-up), and prescribes mitigations (rollback, dt-reduction,
// partition isolation). Self-contained — runs at ~30Hz.

export type GuardKey =
  | "energy_drift" | "nan" | "constraint" | "partition" | "velocity";

export interface GuardThresholds {
  energyDriftPct: number;   // |ΔE/E0| threshold
  constraintTol: number;    // max |c|
  velocityCap: number;      // |v| cap (px/s proxy)
  partitionVar: number;     // per-partition energy variance
}

export const DEFAULT_THRESHOLDS: GuardThresholds = {
  energyDriftPct: 8,
  constraintTol: 0.04,
  velocityCap: 950,
  partitionVar: 0.45,
};

export interface Mitigation {
  kind: "rollback" | "shrink_dt" | "isolate_partition" | "clamp_velocity" | "reset_partition";
  detail: string;
}

export interface GuardEvent {
  t: number;
  guard: GuardKey;
  severity: "info" | "warn" | "critical";
  value: number;
  threshold: number;
  partition?: number;
  mitigation?: Mitigation;
}

export interface Snapshot {
  t: number;
  energy: number;
  partitions: number[];   // per-partition energy
}

export interface GuardState {
  thresholds: GuardThresholds;
  history: Snapshot[];     // rolling, capped
  events: GuardEvent[];    // rolling, capped
  baselineEnergy: number;
  dt: number;              // current effective dt (sec)
  baseDt: number;
  isolated: Set<number>;
  rollbacks: number;
  shrinks: number;
  enabled: boolean;
}

export function createGuard(thresholds = DEFAULT_THRESHOLDS, baseDt = 1 / 60): GuardState {
  return {
    thresholds,
    history: [],
    events: [],
    baselineEnergy: 1,
    dt: baseDt,
    baseDt,
    isolated: new Set(),
    rollbacks: 0,
    shrinks: 0,
    enabled: true,
  };
}

// Synthetic tape — deterministic but injects faults at known times.
export function syntheticStep(g: GuardState, t: number, faultMode: "off" | "drift" | "nan" | "blowup" | "partition"): Snapshot {
  const numP = 6;
  const partitions: number[] = [];
  const baseE = 1 + 0.05 * Math.sin(t * 1.7);
  for (let i = 0; i < numP; i++) {
    let e = baseE * (0.85 + 0.15 * Math.cos(t * 0.9 + i));
    if (faultMode === "partition" && i === 2 && t > 1.5) {
      e *= 1 + (t - 1.5) * 0.6; // diverging partition
    }
    if (faultMode === "blowup" && t > 2) {
      e *= Math.exp((t - 2) * 0.4);
    }
    if (faultMode === "drift" && t > 1) {
      e *= 1 + (t - 1) * 0.05;
    }
    if (faultMode === "nan" && t > 2.4 && i === 4) {
      e = Number.NaN;
    }
    if (g.isolated.has(i)) e = baseE * 0.95; // isolated partition runs in safe mode
    partitions.push(e);
  }
  const energy = partitions.reduce((s, e) => s + (Number.isFinite(e) ? e : 0), 0);
  return { t, energy, partitions };
}

function emit(g: GuardState, ev: GuardEvent) {
  g.events.push(ev);
  while (g.events.length > 60) g.events.shift();
}

export function ingest(g: GuardState, snap: Snapshot): GuardEvent[] {
  g.history.push(snap);
  while (g.history.length > 240) g.history.shift();

  if (g.history.length === 1) g.baselineEnergy = snap.energy || 1;
  const events: GuardEvent[] = [];
  if (!g.enabled) return events;

  // 1. NaN detection (highest priority — rollback)
  for (let i = 0; i < snap.partitions.length; i++) {
    if (!Number.isFinite(snap.partitions[i])) {
      const ev: GuardEvent = {
        t: snap.t, guard: "nan", severity: "critical",
        value: NaN, threshold: 0, partition: i,
        mitigation: { kind: "rollback", detail: `restore from t=${(snap.t - g.dt * 4).toFixed(2)} · isolate p${i}` },
      };
      events.push(ev); emit(g, ev);
      // mitigate: rollback 4 steps + isolate
      g.history.splice(Math.max(0, g.history.length - 5));
      g.isolated.add(i);
      g.rollbacks++;
      return events;
    }
  }

  // 2. Energy drift
  const driftPct = Math.abs(snap.energy - g.baselineEnergy) / Math.max(g.baselineEnergy, 1e-9) * 100;
  if (driftPct > g.thresholds.energyDriftPct) {
    const sev = driftPct > g.thresholds.energyDriftPct * 2 ? "critical" : "warn";
    const ev: GuardEvent = {
      t: snap.t, guard: "energy_drift", severity: sev,
      value: driftPct, threshold: g.thresholds.energyDriftPct,
      mitigation: sev === "critical"
        ? { kind: "rollback", detail: `Δ=${driftPct.toFixed(1)}% — restore baseline + halve dt` }
        : { kind: "shrink_dt", detail: `dt ${(g.dt * 1000).toFixed(2)}ms → ${(g.dt * 500).toFixed(2)}ms` },
    };
    events.push(ev); emit(g, ev);
    if (sev === "critical") {
      g.history.splice(Math.max(0, g.history.length - 8));
      g.dt = Math.max(g.baseDt / 8, g.dt * 0.5);
      g.rollbacks++; g.shrinks++;
    } else {
      g.dt = Math.max(g.baseDt / 8, g.dt * 0.75);
      g.shrinks++;
    }
  } else if (driftPct < g.thresholds.energyDriftPct * 0.4 && g.dt < g.baseDt) {
    // recover dt slowly
    g.dt = Math.min(g.baseDt, g.dt * 1.05);
  }

  // 3. Per-partition variance (blow-up / one bad neighbor)
  const mean = snap.partitions.reduce((s, e) => s + e, 0) / snap.partitions.length;
  for (let i = 0; i < snap.partitions.length; i++) {
    if (g.isolated.has(i)) continue;
    const dev = Math.abs(snap.partitions[i] - mean) / Math.max(mean, 1e-9);
    if (dev > g.thresholds.partitionVar) {
      const ev: GuardEvent = {
        t: snap.t, guard: "partition", severity: "warn",
        value: dev, threshold: g.thresholds.partitionVar, partition: i,
        mitigation: { kind: "isolate_partition", detail: `p${i} dev ${(dev * 100).toFixed(0)}% — isolate + reseed` },
      };
      events.push(ev); emit(g, ev);
      g.isolated.add(i);
    }
  }

  // 4. Constraint violation (synthetic — proportional to dt growth)
  const cViol = Math.max(0, (g.dt / g.baseDt - 1)) * 0.05 + 0.005 * Math.abs(Math.sin(snap.t * 4));
  if (cViol > g.thresholds.constraintTol) {
    const ev: GuardEvent = {
      t: snap.t, guard: "constraint", severity: "warn",
      value: cViol, threshold: g.thresholds.constraintTol,
      mitigation: { kind: "shrink_dt", detail: `|c|=${cViol.toFixed(3)} — extra constraint iters` },
    };
    events.push(ev); emit(g, ev);
  }

  // 5. Velocity cap (rough proxy: energy spike rate)
  if (g.history.length >= 2) {
    const prev = g.history[g.history.length - 2];
    const rate = Math.abs(snap.energy - prev.energy) / Math.max(g.dt, 1e-3);
    const v = rate * 30; // arbitrary scale
    if (v > g.thresholds.velocityCap) {
      const ev: GuardEvent = {
        t: snap.t, guard: "velocity", severity: "warn",
        value: v, threshold: g.thresholds.velocityCap,
        mitigation: { kind: "clamp_velocity", detail: `|v|=${v.toFixed(0)} → cap ${g.thresholds.velocityCap}` },
      };
      events.push(ev); emit(g, ev);
    }
  }

  return events;
}

export function releaseIsolation(g: GuardState, p: number) {
  g.isolated.delete(p);
  emit(g, {
    t: g.history[g.history.length - 1]?.t ?? 0,
    guard: "partition", severity: "info",
    value: 0, threshold: 0, partition: p,
    mitigation: { kind: "reset_partition", detail: `p${p} re-joined` },
  });
}
