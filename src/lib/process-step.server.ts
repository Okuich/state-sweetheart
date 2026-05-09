/**
 * STEP processing pipeline: parse → describe → reason via Lovable AI Gateway.
 * Server-only.
 */
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { parseStep, buildTopology, describe, validate } from "./stepParser";
import { emitJobEvent } from "./job-events.server";

const AI_GATEWAY_URL = "https://ai.gateway.lovable.dev/v1/chat/completions";
const REASONING_MODEL = "google/gemini-2.5-pro";

class JobCancelledError extends Error {
  constructor() {
    super("cancelled");
    this.name = "JobCancelledError";
  }
}

async function assertNotCancelled(jobId: string): Promise<void> {
  const { data } = await supabaseAdmin
    .from("step_jobs")
    .select("status")
    .eq("id", jobId)
    .maybeSingle();
  if (data?.status === "cancelled") throw new JobCancelledError();
}

export async function processStepJob(jobId: string): Promise<void> {
  const { data: job, error: loadErr } = await supabaseAdmin
    .from("step_jobs")
    .select("id,storage_path,filename,status")
    .eq("id", jobId)
    .single();
  if (loadErr || !job) throw new Error(loadErr?.message ?? "job not found");
  if (job.status === "cancelled") return; // already cancelled before pickup

  try {
    await assertNotCancelled(jobId);
    await supabaseAdmin.from("step_jobs").update({ status: "parsing" }).eq("id", jobId);
    await emitJobEvent(jobId, { stage: "queued", progress: 5, message: "job picked up" });

    // Download the STEP file from storage
    await assertNotCancelled(jobId);
    await emitJobEvent(jobId, { stage: "downloading", progress: 15, message: "fetching file" });
    const { data: blob, error: dlErr } = await supabaseAdmin.storage
      .from("step-uploads")
      .download(job.storage_path);
    if (dlErr || !blob) throw new Error(dlErr?.message ?? "download failed");
    const text = await blob.text();

    // Parse + describe
    await assertNotCancelled(jobId);
    await emitJobEvent(jobId, {
      stage: "parsing",
      progress: 35,
      message: `parsing ${text.length} bytes`,
    });
    const report = parseStep(text);
    const topo = buildTopology(report);
    const desc = describe(report, topo);
    const val = validate(report, topo, desc);

    const geometry = {
      filename: job.filename,
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
      parse: {
        durationMs: report.durationMs,
        entities: report.entities.size,
        warnings: report.warnings.length,
        errors: report.errors.length,
      },
    };

    await assertNotCancelled(jobId);
    await supabaseAdmin
      .from("step_jobs")
      .update({ status: "reasoning", geometry: geometry as never })
      .eq("id", jobId);
    await emitJobEvent(jobId, {
      stage: "geometry_ready",
      progress: 60,
      message: `parsed ${desc.counts ? Object.keys(desc.counts).length : 0} entity types`,
      data: { bbox: desc.bbox, counts: desc.counts },
    });

    // AI reasoning
    await assertNotCancelled(jobId);
    await emitJobEvent(jobId, { stage: "reasoning", progress: 75, message: "calling AI gateway" });
    const reasoning = await reasonAboutGeometry(geometry);

    await assertNotCancelled(jobId);
    await supabaseAdmin
      .from("step_jobs")
      .update({
        status: "done",
        reasoning: reasoning as never,
        completed_at: new Date().toISOString(),
      })
      .eq("id", jobId);
    await emitJobEvent(jobId, {
      stage: "done",
      progress: 100,
      message: "analysis complete",
      data: { has_reasoning: !!reasoning },
    });
  } catch (e) {
    if (e instanceof JobCancelledError) {
      await supabaseAdmin
        .from("step_jobs")
        .update({ status: "cancelled", completed_at: new Date().toISOString() })
        .eq("id", jobId);
      await emitJobEvent(jobId, {
        stage: "cancelled",
        progress: 100,
        message: "job cancelled by caller",
      });
      return;
    }
    const msg = e instanceof Error ? e.message : String(e);
    await supabaseAdmin
      .from("step_jobs")
      .update({ status: "failed", error: msg, completed_at: new Date().toISOString() })
      .eq("id", jobId);
    await emitJobEvent(jobId, { stage: "failed", progress: 100, message: msg });
    throw e;
  }
}

async function reasonAboutGeometry(geometry: unknown): Promise<unknown> {
  const apiKey = process.env.LOVABLE_API_KEY;
  if (!apiKey) {
    return { error: "LOVABLE_API_KEY not configured", skipped: true };
  }

  const systemPrompt = `You are the reasoning core of Particle Dynamics Engine, a hidden analysis layer.
Given canonical geometry extracted from a STEP file, produce a structured engineering analysis.
Be concise, quantitative, and avoid speculation when the data is insufficient.`;

  const userPrompt = `Analyse this STEP geometry summary and return JSON ONLY (no prose, no markdown fences) with keys:

{
  "summary": string,                       // 1-2 sentence overview
  "manufacturability": {
    "process_recommendation": string,      // e.g. "3-axis CNC mill", "DMLS", "investment cast"
    "concerns": string[],                  // specific issues (deep pockets, thin walls, undercuts)
    "estimated_complexity": "low" | "medium" | "high"
  },
  "tolerances": {
    "suggested_general": string,           // e.g. "ISO 2768-mK"
    "critical_features": string[]
  },
  "fixturing": {
    "approach": string,
    "datum_candidates": string[]
  },
  "simulation_parameters": {
    "recommended_timestep_s": number,      // physics simulation dt
    "contact_stiffness": "soft" | "medium" | "stiff",
    "broadphase_cell_size_hint": number,   // in same units as bbox
    "rationale": string
  },
  "risks": string[]                        // anything else worth flagging
}

Geometry:
${JSON.stringify(geometry, null, 2)}`;

  const res = await fetch(AI_GATEWAY_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: REASONING_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    return { error: `AI gateway ${res.status}`, detail: body.slice(0, 500) };
  }

  const json = (await res.json()) as { choices?: Array<{ message?: { content?: string } }> };
  const content = json.choices?.[0]?.message?.content ?? "";

  // Try to parse JSON out of the response
  try {
    const cleaned = content.trim().replace(/^```(?:json)?\s*/i, "").replace(/```\s*$/, "");
    return { model: REASONING_MODEL, ...JSON.parse(cleaned) };
  } catch {
    return { model: REASONING_MODEL, raw: content };
  }
}
