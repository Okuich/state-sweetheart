import { createFileRoute } from "@tanstack/react-router";
import type { TelemetrySample } from "@/lib/telemetrySchema";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth } from "@/lib/service-auth.server";

/**
 * Server-Sent Events stream of telemetry samples.
 */
export const Route = createFileRoute("/api/public/telemetry/stream")({
  server: {
    handlers: {
      OPTIONS: async ({ request }: { request: Request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      GET: async ({ request }: { request: Request }) => {
        // Auth: prefer Bearer pde_ API key (scope telemetry:read). Fall back to
        // X-Telemetry-Token header or ?token= query (since EventSource can't set
        // custom headers).
        const auth = request.headers.get("authorization") ?? "";
        let authed = false;
        if (/^Bearer\s+pde_/i.test(auth)) {
          const client = await verifyServiceAuth(request, "/api/public/telemetry/stream", "telemetry:read");
          if (client) authed = true;
        }
        if (!authed) {
          const expected = (typeof process !== "undefined" ? process.env.TELEMETRY_INGEST_TOKEN : "") ?? "";
          if (expected) {
            const url = new URL(request.url);
            const provided =
              request.headers.get("x-telemetry-token") ?? url.searchParams.get("token") ?? "";
            const { createHash, timingSafeEqual } = await import("node:crypto");
            const a = createHash("sha256").update(provided).digest();
            const b = createHash("sha256").update(expected).digest();
            if (!timingSafeEqual(a, b)) {
              return new Response("unauthorized", { status: 401, headers: corsHeaders(request) });
            }
          }
        }


        const { telemetryBus } = await import("@/lib/telemetryBus");
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            const enc = new TextEncoder();
            const send = (event: string, data: unknown) => {
              try {
                controller.enqueue(
                  enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
                );
              } catch { /* stream closed */ }
            };

            send("hello", { stats: telemetryBus.stats(), at: Date.now() });
            for (const s of telemetryBus.recent(16)) send("sample", s);

            const unsub = telemetryBus.subscribe((s: TelemetrySample) => {
              send("sample", s);
            });

            const ping = setInterval(() => {
              try { controller.enqueue(enc.encode(`: ping ${Date.now()}\n\n`)); }
              catch { /* closed */ }
            }, 15_000);

            const close = () => {
              clearInterval(ping);
              unsub();
              try { controller.close(); } catch { /* already closed */ }
            };
            request.signal.addEventListener("abort", close);
          },
        });

        return new Response(stream, {
          status: 200,
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            ...corsHeaders(request),
          },
        });
      },
    },
  },
} as never);
