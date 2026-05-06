import { createFileRoute } from "@tanstack/react-router";
import type { TelemetrySample } from "@/lib/telemetrySchema";

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type",
} as const;

/**
 * Server-Sent Events stream of telemetry samples.
 */
export const Route = createFileRoute("/api/public/telemetry/stream")({
  server: {
    handlers: {
      OPTIONS: async () => new Response(null, { status: 204, headers: CORS }),

      GET: async ({ request }: { request: Request }) => {
        const { telemetryBus } = await import("@/server/telemetryBus.server");
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
            ...CORS,
          },
        });
      },
    },
  },
} as never);
