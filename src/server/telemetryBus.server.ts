/**
 * Shared in-memory telemetry bus.
 *
 * Module singleton lives for the lifetime of the worker instance. This is
 * fine for dev / single-instance preview; for a multi-instance production
 * deployment swap the bus for a Durable Object or Redis pub/sub.
 */

import { z } from "zod";

export const TelemetrySampleSchema = z.object({
  t: z.number().finite(),
  energy_drift_pct: z.number().finite().optional(),
  constraint_l2:    z.number().finite().min(0).optional(),
  divergence_risk:  z.number().finite().min(0).max(1).optional(),
  velocity_max:     z.number().finite().min(0).optional(),
  nan_count:        z.number().int().min(0).optional(),
  source:           z.string().min(1).max(64).optional(),
});
export type TelemetrySample = z.infer<typeof TelemetrySampleSchema>;

export const IngestSchema = z.union([
  TelemetrySampleSchema,
  z.object({ samples: z.array(TelemetrySampleSchema).min(1).max(256) }),
]);

type Listener = (s: TelemetrySample) => void;

class TelemetryBus {
  private listeners = new Set<Listener>();
  private ring: TelemetrySample[] = [];
  private cap = 512;
  private totalIngested = 0;

  publish(s: TelemetrySample): void {
    this.ring.push(s);
    if (this.ring.length > this.cap) this.ring.shift();
    this.totalIngested++;
    for (const l of this.listeners) {
      try { l(s); } catch { /* listener errors must not block ingest */ }
    }
  }

  subscribe(l: Listener): () => void {
    this.listeners.add(l);
    return () => this.listeners.delete(l);
  }

  recent(limit = 64): TelemetrySample[] {
    return this.ring.slice(-Math.max(1, Math.min(limit, this.cap)));
  }

  stats() {
    return {
      buffered: this.ring.length,
      totalIngested: this.totalIngested,
      listeners: this.listeners.size,
    };
  }
}

const g = globalThis as unknown as { __telemetryBus?: TelemetryBus };
if (!g.__telemetryBus) g.__telemetryBus = new TelemetryBus();
export const telemetryBus: TelemetryBus = g.__telemetryBus;
