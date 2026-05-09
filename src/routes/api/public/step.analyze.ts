/**
 * POST /api/public/step/analyze
 *
 * Service-auth via `Authorization: Bearer pde_<key>` (scope: step:ingest).
 * Body: multipart/form-data with field `file` (.step / .stp, ≤25 MB).
 * Returns: { job_id, status: "queued" } immediately; processing happens in background.
 */
import { createFileRoute } from "@tanstack/react-router";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { processStepJob } from "@/lib/process-step";

const MAX_BYTES = 25 * 1024 * 1024;
const ROUTE = "/api/public/step/analyze";

export const Route = createFileRoute("/api/public/step/analyze")({
  server: {
    handlers: {
      OPTIONS: async ({ request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      POST: async ({ request }) => {
        const t0 = Date.now();
        const cors = corsHeaders(request);
        const json = (status: number, body: unknown, clientId: string | null = null) => {
          void logRequest(clientId, ROUTE, "POST", status, Date.now() - t0);
          return new Response(JSON.stringify(body), {
            status,
            headers: { ...cors, "Content-Type": "application/json" },
          });
        };

        const client = await verifyServiceAuth(request, ROUTE, "step:ingest");
        if (!client) return json(401, { error: "unauthorized" });

        const ctype = request.headers.get("content-type") ?? "";
        if (!ctype.includes("multipart/form-data")) {
          return json(400, { error: "expected multipart/form-data" }, client.id);
        }

        let form: FormData;
        try {
          form = await request.formData();
        } catch {
          return json(400, { error: "invalid form data" }, client.id);
        }

        const file = form.get("file");
        if (!(file instanceof File)) {
          return json(400, { error: "missing 'file' field" }, client.id);
        }
        if (file.size > MAX_BYTES) {
          return json(413, { error: `file exceeds ${MAX_BYTES} bytes` }, client.id);
        }
        const name = file.name || "upload.step";
        const lower = name.toLowerCase();
        if (!lower.endsWith(".step") && !lower.endsWith(".stp")) {
          return json(400, { error: "filename must end in .step or .stp" }, client.id);
        }

        const ts = Date.now();
        const safe = name.replace(/[^a-zA-Z0-9._-]/g, "_");
        const storagePath = `${client.id}/${ts}-${safe}`;

        const arrayBuf = await file.arrayBuffer();
        const { error: upErr } = await supabaseAdmin.storage
          .from("step-uploads")
          .upload(storagePath, new Uint8Array(arrayBuf), {
            contentType: "application/step",
            upsert: false,
          });
        if (upErr) return json(500, { error: "upload failed", detail: upErr.message }, client.id);

        const { data: row, error: insErr } = await supabaseAdmin
          .from("step_jobs")
          .insert({
            client_id: client.id,
            filename: name,
            storage_path: storagePath,
            status: "queued",
          })
          .select("id")
          .single();
        if (insErr || !row) {
          return json(500, { error: "job create failed", detail: insErr?.message }, client.id);
        }

        // Fire-and-forget processing. The Worker keeps the function alive
        // for the response, but processStepJob writes status+results to DB
        // so callers poll /api/public/step/jobs/:id.
        void processStepJob(row.id).catch((e) => {
          console.error("[step-analyze] processing failed", row.id, e);
        });

        return json(202, { job_id: row.id, status: "queued" }, client.id);
      },
    },
  },
});
