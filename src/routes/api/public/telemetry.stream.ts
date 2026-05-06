import { createFileRoute } from "@tanstack/react-router";
import { telemetryBus, type TelemetrySample } from "@/server/telemetryBus.server";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
} as const;

/**
 * Server-Sent Events stream of telemetry samples.
 *
 * Why SSE and not raw WebSocket: SSE works through the standard fetch /
 * Response pipeline used by TanStack server routes (no protocol upgrade
 * required) and runs on Cloudflare Workers without extra plumbing. The
 * client-side EventSource auto-reconnects.
 */
export const Route = createFileRoute("/api/public/telemetry/stream")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),

      GET: async ({ request }) => {
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

            // Initial hello + recent backfill so late subscribers have context.
            send("hello", { stats: telemetryBus.stats(), at: Date.now() });
            for (const s of telemetryBus.recent(16)) send("sample", s);

            const unsub = telemetryBus.subscribe((s: TelemetrySample) => {
              send("sample", s);
            });

            // Keep-alive comment every 15 s — proxies kill idle SSE streams.
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
            ...CORS,
          },
        });
      },
    },
  },
});
