/**
 * POST /api/public/step/jobs/:id/cancel
 *
 * Cancel a queued or in-flight STEP job. Marks the job `cancelled` and emits
 * a `cancelled` event so any open SSE stream sees a terminal frame.
 *
 * Auth: `Authorization: Bearer pde_<key>` with scope `step:ingest`.
 *       Caller must own the job (job.client_id === client.id).
 */
import { createFileRoute } from "@tanstack/react-router";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth.server";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { emitJobEvent } from "@/lib/job-events.server";

const ROUTE = "/api/public/step/jobs/:id/cancel";
const TERMINAL = new Set(["done", "failed", "cancelled"]);

export const Route = createFileRoute("/api/public/step/jobs/$id/cancel")({
  server: {
    handlers: {
      OPTIONS: async ({ request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      POST: async ({ request, params }) => {
        const t0 = Date.now();
        const cors = corsHeaders(request);
        const jobId = params.id;
        const json = (status: number, body: unknown, clientId: string | null = null) => {
          void logRequest(clientId, ROUTE, "POST", status, Date.now() - t0);
          return new Response(JSON.stringify(body), {
            status,
            headers: { ...cors, "Content-Type": "application/json" },
          });
        };

        const client = await verifyServiceAuth(request, ROUTE, "step:ingest");
        if (!client) return json(401, { error: "unauthorized" });

        const { data: job, error: loadErr } = await supabaseAdmin
          .from("step_jobs")
          .select("id,client_id,status")
          .eq("id", jobId)
          .maybeSingle();
        if (loadErr) return json(500, { error: loadErr.message }, client.id);
        if (!job) return json(404, { error: "not found" }, client.id);
        if (job.client_id !== client.id) {
          return json(403, { error: "forbidden" }, client.id);
        }
        if (TERMINAL.has(job.status)) {
          return json(
            409,
            { error: "job already terminal", status: job.status },
            client.id,
          );
        }

        const { error: upErr } = await supabaseAdmin
          .from("step_jobs")
          .update({ status: "cancelled", completed_at: new Date().toISOString() })
          .eq("id", jobId);
        if (upErr) return json(500, { error: upErr.message }, client.id);

        // Emit terminal event so SSE subscribers see it immediately. The
        // background processStepJob loop also checks status between stages
        // and exits cleanly the next time it polls.
        await emitJobEvent(jobId, {
          stage: "cancelled",
          progress: 100,
          message: "job cancelled by caller",
        });

        return json(200, { id: jobId, status: "cancelled" }, client.id);
      },
    },
  },
});
