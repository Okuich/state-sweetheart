/**
 * physicsReasoner.functions.ts
 *
 * Server function that takes the synthesized RAG context preview and asks
 * Lovable AI to recommend the next simulation parameters. We use tool
 * calling to force a structured JSON response matching `RecommendationSchema`
 * — so the UI can render typed parameter cards instead of parsing prose.
 */
import { createServerFn } from "@tanstack/react-start";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { z } from "zod";

const InputSchema = z.object({
  contextPrompt: z.string().min(10).max(8000),
  query: z.string().min(1).max(400),
});

export const RecommendationSchema = z.object({
  rationale: z.string(),
  confidence: z.number().min(0).max(1),
  parameters: z.array(z.object({
    name: z.string(),
    value: z.union([z.number(), z.string()]),
    unit: z.string().optional(),
    rationale: z.string(),
  })).min(1).max(12),
  warnings: z.array(z.string()).default([]),
  nextActions: z.array(z.string()).default([]),
});
export type Recommendation = z.infer<typeof RecommendationSchema>;

const SYSTEM = `You are a physics simulation reasoner advising on FEM/CFD/multiphysics setups.
Given retrieved context (similar geometries, topology matches, past failures, optimization precedents),
recommend the next simulation's parameters. Be specific, numerical, and grounded in the context.
Cover: solver type, time step / dt, mesh density / element size, integration scheme, damping,
material assumption, boundary conditions, convergence tolerance, max iterations, and substepping
when relevant. Flag risks from past failures.`;

export const recommendSimulationParameters = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((data: unknown) => InputSchema.parse(data))
  .handler(async ({ data }): Promise<Recommendation> => {
    const apiKey = process.env.LOVABLE_API_KEY;
    if (!apiKey) throw new Error("LOVABLE_API_KEY is not configured");

    const res = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: SYSTEM },
          { role: "user", content:
            `Query: ${data.query}\n\nRetrieved context:\n${data.contextPrompt}\n\n` +
            `Return concrete recommended simulation parameters via the tool call.` },
        ],
        tools: [{
          type: "function",
          function: {
            name: "recommend_parameters",
            description: "Recommend next simulation parameters from physics context.",
            parameters: {
              type: "object",
              properties: {
                rationale: { type: "string", description: "1-3 sentence high-level reasoning." },
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

    if (res.status === 429) throw new Error("Rate limit exceeded — please retry shortly.");
    if (res.status === 402) throw new Error("AI credits exhausted — add credits in Settings → Workspace → Usage.");
    if (!res.ok) {
      const t = await res.text();
      throw new Error(`Reasoner failed (${res.status}): ${t.slice(0, 200)}`);
    }

    const json = await res.json();
    const call = json?.choices?.[0]?.message?.tool_calls?.[0];
    if (!call) throw new Error("No structured recommendation returned");
    const args = typeof call.function?.arguments === "string"
      ? JSON.parse(call.function.arguments)
      : call.function?.arguments;
    return RecommendationSchema.parse(args);
  });
