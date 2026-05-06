import { createFileRoute } from "@tanstack/react-router";
import { IngestSchema, type TelemetrySample } from "@/lib/telemetrySchema";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Max-Age": "86400",
} as const;

const json = (body: unknown, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json", ...CORS, ...(init.headers ?? {}) },
  });

export const Route = createFileRoute("/api/public/telemetry")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),

      GET: async () => {
        const { telemetryBus } = await import("@/server/telemetryBus.server");
        return json({
          ok: true,
          stats: telemetryBus.stats(),
          recent: telemetryBus.recent(32),
        });
      },

      POST: async ({ request }: { request: Request }) => {
        let raw: unknown;
        try { raw = await request.json(); }
        catch { return json({ error: "invalid JSON body" }, { status: 400 }); }

        const parsed = IngestSchema.safeParse(raw);
        if (!parsed.success) {
          return json(
            { error: "validation failed", issues: parsed.error.flatten() },
            { status: 422 },
          );
        }

        const { telemetryBus } = await import("@/server/telemetryBus.server");
        const samples: TelemetrySample[] =
          "samples" in parsed.data ? parsed.data.samples : [parsed.data];
        for (const s of samples) telemetryBus.publish(s);

        return json({ ok: true, accepted: samples.length, stats: telemetryBus.stats() });
      },
    },
  },
} as never);
