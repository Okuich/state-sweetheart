/**
 * Client-to-server RPC wrapper for telemetry ingest.
 *
 * Components import these helpers and call them via `useServerFn` (or
 * directly from event handlers). The transport is a typed RPC managed by
 * TanStack Start — no manual fetch/CORS/JSON wiring needed on the client,
 * and the server-only telemetry bus is never reachable from the browser
 * bundle.
 */
import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { IngestSchema, type TelemetrySample } from "@/lib/telemetrySchema";

export const publishTelemetry = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: unknown) => IngestSchema.parse(input))
  .handler(async ({ data }) => {
    const { telemetryBus } = await import("@/lib/telemetryBus");
    const samples: TelemetrySample[] =
      "samples" in data ? data.samples : [data];
    for (const s of samples) telemetryBus.publish(s);
    return { ok: true as const, accepted: samples.length, stats: telemetryBus.stats() };
  });

export const getTelemetryRecent = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input: { limit?: number } | undefined) => ({
    limit: Math.max(1, Math.min(Number(input?.limit ?? 64) || 64, 512)),
  }))
  .handler(async ({ data }) => {
    const { telemetryBus } = await import("@/lib/telemetryBus");
    return { stats: telemetryBus.stats(), recent: telemetryBus.recent(data.limit) };
  });
