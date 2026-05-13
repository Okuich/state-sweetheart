/**
 * GET /api/public/step/jobs/:id
 *
 * Service-auth via Bearer token (scope: step:read).
 * Returns: { id, status, geometry?, reasoning?, error?, created_at, completed_at }.
 * A client may only read jobs it created.
 */
import { createFileRoute } from "@tanstack/react-router";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth";
import { getAdmin } from "@/lib/admin";

const ROUTE = "/api/public/step/jobs/:id";

export const Route = createFileRoute("/api/public/step/jobs/$id")({
  server: {
    handlers: {
      OPTIONS: async ({ request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      GET: async ({ request, params }) => {
        const t0 = Date.now();
        const cors = corsHeaders(request);
        const json = (status: number, body: unknown, clientId: string | null = null) => {
          void logRequest(clientId, ROUTE, "GET", status, Date.now() - t0);
          return new Response(JSON.stringify(body), {
            status,
            headers: { ...cors, "Content-Type": "application/json" },
          });
        };

        const client = await verifyServiceAuth(request, ROUTE, "step:read");
        if (!client) return json(401, { error: "unauthorized" });

        const { data, error } = await getAdmin()
          .from("step_jobs")
          .select("id,client_id,filename,status,error,geometry,reasoning,mesh,progress,created_at,completed_at")
          .eq("id", params.id)
          .maybeSingle();

        if (error) return json(500, { error: error.message }, client.id);
        if (!data) return json(404, { error: "not found" }, client.id);
        if (data.client_id !== client.id) return json(404, { error: "not found" }, client.id);

        return json(
          200,
          {
            id: data.id,
            filename: data.filename,
            status: data.status,
            error: data.error,
            geometry: data.geometry,
            reasoning: data.reasoning,
            progress: data.progress,
            created_at: data.created_at,
            completed_at: data.completed_at,
          },
          client.id,
        );
      },
    },
  },
});
