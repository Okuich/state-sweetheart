import { createFileRoute } from "@tanstack/react-router";
import { IngestSchema, type TelemetrySample } from "@/lib/telemetrySchema";
import { corsHeaders } from "@/lib/cors";
import { log } from "@/lib/serverLog";

const json = (body: unknown, request: Request, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json", ...corsHeaders(request), ...(init.headers ?? {}) },
  });

/**
 * Token-gated ingest. When `TELEMETRY_INGEST_TOKEN` is set, POST callers must
 * present the same value via the `X-Telemetry-Token` header. When the env var
 * is unset (dev/preview), ingest is open. GET (status) stays open either way.
 */
function checkToken(request: Request): { ok: true } | { ok: false; reason: string } {
  const expected = (typeof process !== "undefined" ? process.env.TELEMETRY_INGEST_TOKEN : "") ?? "";
  if (!expected) return { ok: true };
  const provided = request.headers.get("x-telemetry-token") ?? "";
  if (provided.length === expected.length && provided === expected) return { ok: true };
  return { ok: false, reason: "missing or invalid X-Telemetry-Token" };
}

export const Route = createFileRoute("/api/public/telemetry")({
  server: {
    handlers: {
      OPTIONS: async ({ request }: { request: Request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      GET: async ({ request }: { request: Request }) => {
        const { telemetryBus } = await import("@/lib/telemetryBus");
        return json({
          ok: true,
          stats: telemetryBus.stats(),
          recent: telemetryBus.recent(32),
        }, request);
      },

      POST: async ({ request }: { request: Request }) => {
        const auth = checkToken(request);
        if (!auth.ok) {
          log("warn", "telemetry.ingest.unauthorized", { reason: auth.reason });
          return json({ error: "unauthorized" }, request, { status: 401 });
        }

        let raw: unknown;
        try { raw = await request.json(); }
        catch { return json({ error: "invalid JSON body" }, request, { status: 400 }); }

        const parsed = IngestSchema.safeParse(raw);
        if (!parsed.success) {
          log("warn", "telemetry.ingest.invalid", { issues: parsed.error.flatten() });
          return json(
            { error: "validation failed", issues: parsed.error.flatten() },
            request, { status: 422 },
          );
        }

        const { telemetryBus } = await import("@/lib/telemetryBus");
        const samples: TelemetrySample[] =
          "samples" in parsed.data ? parsed.data.samples : [parsed.data];
        for (const s of samples) telemetryBus.publish(s);

        log("info", "telemetry.ingest.ok", { count: samples.length });
        return json({ ok: true, accepted: samples.length, stats: telemetryBus.stats() }, request);
      },
    },
  },
} as never);
