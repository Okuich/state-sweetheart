/**
 * POST /api/public/step/parse
 *
 * Synchronous STEP parser. Accepts a .step/.stp file and returns
 * structured geometry + features for downstream reasoning.
 *
 * Service-auth via `Authorization: Bearer pde_<key>` (scope: step:ingest).
 * Body: multipart/form-data with field `file` (.step / .stp, ≤10 MB).
 *   OR: raw body with `Content-Type: application/step` and
 *       `X-Filename: foo.step` header.
 *
 * Returns: { geometry, parse_stats } as JSON.
 */
import { createFileRoute } from "@tanstack/react-router";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth";
import { parseStep, buildTopology, describe, validate } from "@/lib/stepParser";

const MAX_BYTES = 10 * 1024 * 1024; // sync limit; use /analyze for larger
const ROUTE = "/api/public/step/parse";

export const Route = createFileRoute("/api/public/step/parse")({
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
        let filename = "upload.step";
        let text: string;

        try {
          if (ctype.includes("multipart/form-data")) {
            const form = await request.formData();
            const file = form.get("file");
            if (!(file instanceof File)) {
              return json(400, { error: "missing 'file' field" }, client.id);
            }
            if (file.size > MAX_BYTES) {
              return json(
                413,
                { error: `file exceeds ${MAX_BYTES} bytes; use /api/public/step/analyze` },
                client.id,
              );
            }
            filename = file.name || filename;
            text = await file.text();
          } else {
            // Raw body path
            const headerName = request.headers.get("x-filename");
            if (headerName) filename = headerName;
            const buf = await request.arrayBuffer();
            if (buf.byteLength > MAX_BYTES) {
              return json(
                413,
                { error: `body exceeds ${MAX_BYTES} bytes; use /api/public/step/analyze` },
                client.id,
              );
            }
            text = new TextDecoder().decode(buf);
          }
        } catch (e) {
          return json(
            400,
            { error: "could not read body", detail: e instanceof Error ? e.message : String(e) },
            client.id,
          );
        }

        const lower = filename.toLowerCase();
        if (!lower.endsWith(".step") && !lower.endsWith(".stp")) {
          return json(400, { error: "filename must end in .step or .stp" }, client.id);
        }
        if (!text.trim().length) {
          return json(400, { error: "empty file" }, client.id);
        }
        if (!/ISO-10303/i.test(text.slice(0, 200))) {
          return json(
            400,
            { error: "not a STEP file (missing ISO-10303 header)" },
            client.id,
          );
        }

        try {
          const report = parseStep(text);
          const topo = buildTopology(report);
          const desc = describe(report, topo);
          const val = validate(report, topo, desc);

          const geometry = {
            filename,
            bytes: text.length,
            schema: desc.schema,
            counts: desc.counts,
            manifold: desc.manifold,
            bbox: desc.bbox,
            features: desc.features,
            header: report.header,
            validation: {
              ok: val.ok,
              stats: val.stats,
              issues: val.issues.slice(0, 50),
            },
          };

          return json(
            200,
            {
              geometry,
              parse_stats: {
                duration_ms: report.durationMs,
                entities: report.entities.size,
                warnings: report.warnings.length,
                errors: report.errors.length,
              },
            },
            client.id,
          );
        } catch (e) {
          return json(
            422,
            { error: "parse failed", detail: e instanceof Error ? e.message : String(e) },
            client.id,
          );
        }
      },
    },
  },
});
