import { createFileRoute } from "@tanstack/react-router";
import { z } from "zod";
import { corsHeaders } from "@/lib/cors";
import { verifyServiceAuth, logRequest } from "@/lib/service-auth.server";
import { RecommendationSchema } from "@/server/physicsReasoner.functions";

const InputSchema = z.object({
  contextPrompt: z.string().min(10).max(8000),
  query: z.string().min(1).max(400),
});

const SYSTEM = `You are a physics simulation reasoner advising on FEM/CFD/multiphysics setups.
Given retrieved context, recommend the next simulation's parameters. Be specific and numerical.`;

const json = (body: unknown, request: Request, init: ResponseInit = {}) =>
  new Response(JSON.stringify(body), {
    ...init,
    headers: { "Content-Type": "application/json", ...corsHeaders(request), ...(init.headers ?? {}) },
  });

export const Route = createFileRoute("/api/public/reasoner/recommend")({
  server: {
    handlers: {
      OPTIONS: async ({ request }: { request: Request }) =>
        new Response(null, { status: 204, headers: corsHeaders(request) }),

      POST: async ({ request }: { request: Request }) => {
        const start = Date.now();
        const route = "/api/public/reasoner/recommend";
        const client = await verifyServiceAuth(request, route, "reasoner:invoke");
        if (!client) {
          await logRequest(null, route, "POST", 401, Date.now() - start);
          return json({ error: "unauthorized" }, request, { status: 401 });
        }

        let raw: unknown;
        try { raw = await request.json(); }
        catch {
          await logRequest(client.id, route, "POST", 400, Date.now() - start);
          return json({ error: "invalid JSON" }, request, { status: 400 });
        }

        const parsed = InputSchema.safeParse(raw);
        if (!parsed.success) {
          await logRequest(client.id, route, "POST", 422, Date.now() - start);
          return json({ error: "validation failed", issues: parsed.error.flatten() }, request, { status: 422 });
        }

        const apiKey = process.env.LOVABLE_API_KEY;
        if (!apiKey) {
          await logRequest(client.id, route, "POST", 500, Date.now() - start);
          return json({ error: "AI gateway not configured" }, request, { status: 500 });
        }

        const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
          method: "POST",
          headers: { Authorization: `Bearer ${apiKey}`, "Content-Type": "application/json" },
          body: JSON.stringify({
            model: "google/gemini-2.5-pro",
            messages: [
              { role: "system", content: SYSTEM },
              { role: "user", content: `Query: ${parsed.data.query}\n\nRetrieved context:\n${parsed.data.contextPrompt}` },
            ],
            tools: [{
              type: "function",
              function: {
                name: "recommend_parameters",
                parameters: {
                  type: "object",
                  properties: {
                    rationale: { type: "string" },
                    confidence: { type: "number", minimum: 0, maximum: 1 },
                    parameters: {
                      type: "array",
                      items: {
                        type: "object",
                        properties: {
                          name: { type: "string" },
                          value: { type: ["number", "string"] },
                          unit: { type: "string" },
                          rationale: { type: "string" },
                        },
                        required: ["name", "value", "rationale"],
                      },
                    },
                    warnings: { type: "array", items: { type: "string" } },
                    nextActions: { type: "array", items: { type: "string" } },
                  },
                  required: ["rationale", "confidence", "parameters"],
                },
              },
            }],
            tool_choice: { type: "function", function: { name: "recommend_parameters" } },
          }),
        });

        if (!res.ok) {
          const status = res.status === 429 ? 429 : res.status === 402 ? 402 : 502;
          await logRequest(client.id, route, "POST", status, Date.now() - start);
          return json({ error: `reasoner upstream ${res.status}` }, request, { status });
        }

        const body = await res.json();
        const call = body?.choices?.[0]?.message?.tool_calls?.[0];
        const args = typeof call?.function?.arguments === "string"
          ? JSON.parse(call.function.arguments)
          : call?.function?.arguments;
        const recParsed = RecommendationSchema.safeParse(args);
        if (!recParsed.success) {
          await logRequest(client.id, route, "POST", 502, Date.now() - start);
          return json({ error: "malformed recommendation" }, request, { status: 502 });
        }

        await logRequest(client.id, route, "POST", 200, Date.now() - start);
        return json({ ok: true, client: client.name, recommendation: recParsed.data }, request);
      },
    },
  },
} as never);
