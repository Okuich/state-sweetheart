import { createFileRoute } from "@tanstack/react-router";
import { IngestSchema, type TelemetrySample } from "@/lib/telemetrySchema";
import { corsHeaders } from "@/lib/cors";
import { log } from "@/lib/serverLog";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth.server";

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
import { createHash, timingSafeEqual } from "node:crypto";

/**
 * Auth: accepts EITHER
 *   - `Authorization: Bearer pde_...` API key with scope `telemetry:write` (preferred for service-to-service), OR
 *   - `X-Telemetry-Token` shared secret (legacy).
 * If neither env nor a valid bearer is present, ingest is open in dev.
 */
import { createHash, timingSafeEqual } from "node:crypto";

async function checkAuth(request: Request): Promise<{ ok: true; clientId: string | null } | { ok: false; reason: string }> {
  // 1. Try API key first
  const auth = request.headers.get("authorization") ?? "";
  if (/^Bearer\s+pde_/i.test(auth)) {
    const client = await verifyServiceAuth(request, "/api/public/telemetry", "telemetry:write");
    if (client) return { ok: true, clientId: client.id };
    return { ok: false, reason: "invalid api key or missing telemetry:write scope" };
  }

  // 2. Fall back to shared token
  const expected = (typeof process !== "undefined" ? process.env.TELEMETRY_INGEST_TOKEN : "") ?? "";
  if (!expected) return { ok: true, clientId: null };
  const provided = request.headers.get("x-telemetry-token") ?? "";
  const { createHash, timingSafeEqual } = await import("node:crypto");
  const a = createHash("sha256").update(provided).digest();
  const b = createHash("sha256").update(expected).digest();
  return timingSafeEqual(a, b)
    ? { ok: true, clientId: null }
    : { ok: false, reason: "missing or invalid credentials" };
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
        const auth = await checkToken(request);
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

        // Persist to the telemetry_samples table (admin write, server-only).
        // Fire-and-forget so a slow DB never blocks ingest acks.
        try {
          const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
          void supabaseAdmin
            .from("telemetry_samples")
            .insert(samples.map((s) => ({
              t: s.t,
              energy_drift_pct: s.energy_drift_pct ?? null,
              constraint_l2: s.constraint_l2 ?? null,
              divergence_risk: s.divergence_risk ?? null,
              velocity_max: s.velocity_max ?? null,
              nan_count: s.nan_count ?? null,
              source: s.source ?? null,
            })))
            .then(({ error }) => {
              if (error) log("warn", "telemetry.persist.error", { msg: error.message });
            });
        } catch (e) {
          log("warn", "telemetry.persist.skipped", { msg: (e as Error).message });
        }

        log("info", "telemetry.ingest.ok", { count: samples.length });
        return json({ ok: true, accepted: samples.length, stats: telemetryBus.stats() }, request);
      },
    },
  },
} as never);
