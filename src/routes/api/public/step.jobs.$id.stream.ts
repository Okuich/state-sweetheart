/**
 * GET /api/public/step/jobs/:id/stream
 *
 * Server-Sent Events stream of progress for a STEP analysis job.
 *
 * Auth: `Authorization: Bearer pde_<key>` (scope `step:read`). EventSource
 * cannot set custom headers from the browser, so a `?token=pde_...` query
 * param is also accepted as a fallback for browser clients.
 *
 * Events:
 *   event: snapshot   data: { job, events }      (initial state on connect)
 *   event: progress   data: { stage, progress, message, data, at }
 *   event: done       data: { status, reasoning, geometry }
 *   event: error      data: { message }
 *   event: ping       (every 15s, comment line)
 */
import { createFileRoute } from "@tanstack/react-router";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth.server";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { createClient } from "@supabase/supabase-js";

const ROUTE_TEMPLATE = "/api/public/step/jobs/:id/stream";

export const Route = createFileRoute("/api/public/step/jobs/$id/stream")({
  server: {
    handlers: {
      OPTIONS: async ({ request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      GET: async ({ request, params }) => {
        const t0 = Date.now();
        const cors = corsHeaders(request);
        const jobId = params.id;

        // ---- Auth: header bearer or ?token= fallback ----
        let authedClientId: string | null = null;
        const auth = request.headers.get("authorization") ?? "";
        if (/^Bearer\s+pde_/i.test(auth)) {
          const c = await verifyServiceAuth(request, ROUTE_TEMPLATE, "step:read");
          if (c) authedClientId = c.id;
        } else {
          const url = new URL(request.url);
          const token = url.searchParams.get("token");
          if (token?.startsWith("pde_")) {
            const fakeReq = new Request(request.url, {
              headers: { authorization: `Bearer ${token}` },
            });
            const c = await verifyServiceAuth(fakeReq, ROUTE_TEMPLATE, "step:read");
            if (c) authedClientId = c.id;
          }
        }
        if (!authedClientId) {
          void logRequest(null, ROUTE_TEMPLATE, "GET", 401, Date.now() - t0);
          return new Response("unauthorized", { status: 401, headers: cors });
        }

        // ---- Verify the job belongs to this caller ----
        const { data: job, error: jobErr } = await supabaseAdmin
          .from("step_jobs")
          .select("id,client_id,status,filename,error,geometry,reasoning,progress,created_at,completed_at")
          .eq("id", jobId)
          .maybeSingle();
        if (jobErr || !job) {
          void logRequest(authedClientId, ROUTE_TEMPLATE, "GET", 404, Date.now() - t0);
          return new Response("not found", { status: 404, headers: cors });
        }
        if (job.client_id !== authedClientId) {
          void logRequest(authedClientId, ROUTE_TEMPLATE, "GET", 403, Date.now() - t0);
          return new Response("forbidden", { status: 403, headers: cors });
        }

        // ---- Replay existing events ----
        const { data: existing } = await supabaseAdmin
          .from("step_job_events")
          .select("stage,progress,message,data,created_at")
          .eq("job_id", jobId)
          .order("id", { ascending: true });

        // ---- Build SSE stream subscribed to realtime inserts ----
        const supabaseUrl = process.env.SUPABASE_URL ?? "";
        const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY ?? "";
        // Per-connection client so we get an isolated realtime channel.
        const realtime = createClient(supabaseUrl, serviceKey, {
          auth: { persistSession: false, autoRefreshToken: false },
          realtime: { params: { eventsPerSecond: 20 } },
        });

        let terminated = false;
        const stream = new ReadableStream<Uint8Array>({
          start(controller) {
            const enc = new TextEncoder();
            const send = (event: string, data: unknown) => {
              if (terminated) return;
              try {
                controller.enqueue(
                  enc.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
                );
              } catch {
                /* closed */
              }
            };

            // Initial snapshot: full job + replayed events
            send("snapshot", {
              job: {
                id: job.id,
                status: job.status,
                filename: job.filename,
                progress: job.progress,
                created_at: job.created_at,
                completed_at: job.completed_at,
                error: job.error,
              },
              events: existing ?? [],
            });

            // If already terminal, finish immediately
            if (job.status === "done" || job.status === "failed" || job.status === "cancelled") {
              const evt =
                job.status === "done" ? "done" : job.status === "cancelled" ? "cancelled" : "error";
              send(evt, {
                status: job.status,
                geometry: job.geometry,
                reasoning: job.reasoning,
                error: job.error,
              });
              setTimeout(() => close(), 50);
              return;
            }

            // Subscribe to new events for this job
            const channel = realtime
              .channel(`step-job-${jobId}`)
              .on(
                "postgres_changes",
                {
                  event: "INSERT",
                  schema: "public",
                  table: "step_job_events",
                  filter: `job_id=eq.${jobId}`,
                },
                (payload: { new: Record<string, unknown> }) => {
                  const row = payload.new;
                  send("progress", {
                    stage: row.stage,
                    progress: row.progress,
                    message: row.message,
                    data: row.data,
                    at: row.created_at,
                  });

                  // Terminal stages → fetch final job row, emit, close.
                  const stage = String(row.stage);
                  if (stage === "done" || stage === "failed" || stage === "cancelled") {
                    void (async () => {
                      const { data: finalJob } = await supabaseAdmin
                        .from("step_jobs")
                        .select("status,geometry,reasoning,error")
                        .eq("id", jobId)
                        .maybeSingle();
                      const evt =
                        stage === "done" ? "done" : stage === "cancelled" ? "cancelled" : "error";
                      send(evt, finalJob ?? { status: stage });
                      setTimeout(() => close(), 50);
                    })();
                  }
                },
              )
              .subscribe();

            // Heartbeat
            const ping = setInterval(() => {
              if (terminated) return;
              try {
                controller.enqueue(enc.encode(`: ping ${Date.now()}\n\n`));
              } catch {
                /* closed */
              }
            }, 15_000);

            // Safety timeout: 10 minutes max
            const maxLife = setTimeout(() => close(), 10 * 60 * 1000);

            const close = () => {
              if (terminated) return;
              terminated = true;
              clearInterval(ping);
              clearTimeout(maxLife);
              try {
                void realtime.removeChannel(channel);
              } catch {
                /* noop */
              }
              try {
                controller.close();
              } catch {
                /* already closed */
              }
            };

            request.signal.addEventListener("abort", close);
          },
        });

        void logRequest(authedClientId, ROUTE_TEMPLATE, "GET", 200, Date.now() - t0);
        return new Response(stream, {
          status: 200,
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache, no-transform",
            Connection: "keep-alive",
            "X-Accel-Buffering": "no",
            ...cors,
          },
        });
      },
    },
  },
});
