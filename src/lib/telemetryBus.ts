/**
 * Pluggable telemetry bus.
 *
 * The default `InMemoryTelemetryBus` lives for the lifetime of the worker
 * instance and is fine for single-instance dev/preview. For multi-instance
 * production, build an adapter that implements `TelemetryBusLike` (Durable
 * Object, Redis pub/sub, NATS, ...) and assign it to `globalThis.__telemetryBus`
 * before any route imports `telemetryBus`.
 */

import { z } from "zod";

export interface TelemetryBusLike {
  publish(s: TelemetrySample): void;
  subscribe(l: (s: TelemetrySample) => void): () => void;
  recent(limit?: number): TelemetrySample[];
  stats(): { buffered: number; totalIngested: number; listeners: number };
}

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

export class InMemoryTelemetryBus implements TelemetryBusLike {
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

const g = globalThis as unknown as { __telemetryBus?: TelemetryBusLike };
if (!g.__telemetryBus) g.__telemetryBus = new InMemoryTelemetryBus();
export const telemetryBus: TelemetryBusLike = g.__telemetryBus;
