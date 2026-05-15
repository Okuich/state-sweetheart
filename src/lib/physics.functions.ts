/**
 * Physics server functions — admin/enterprise gated wrappers around the
 * pure analytical engine in physics-engine.server.ts.
 *
 * All functions:
 *   - require an authenticated user (requireSupabaseAuth)
 *   - assert the `physics_engine` feature flag is enabled for the caller
 *   - log the run into the `physics_jobs` table for history/audit
 */
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import {
  handleSingle, handleBatch, handleCompare, handleValidate, handleOptimize,
  ALL_MATERIAL_KEYS, MATERIALS, GEOMETRY_PRESETS,
  type AnalyzeBody,
} from "./physics-engine.server";

// ── Access gate ──
async function assertPhysicsAccess(userId: string) {
  const [rolesRes, flagRes, overrideRes] = await Promise.all([
    supabaseAdmin.from("user_roles").select("role").eq("user_id", userId),
    supabaseAdmin.from("feature_flags").select("enabled, allowed_roles, rollout_percentage").eq("key", "physics_engine").maybeSingle(),
    supabaseAdmin.from("user_feature_flags").select("enabled").eq("user_id", userId).eq("flag_key", "physics_engine").maybeSingle(),
  ]);
  const roles = (rolesRes.data ?? []).map((r) => r.role as string);
  if (overrideRes.data) {
    if (overrideRes.data.enabled) return;
    throw new Error("Physics engine access disabled for your account");
  }
  const flag = flagRes.data;
  if (!flag?.enabled) throw new Error("Physics engine is currently disabled");
  const allowed = flag.allowed_roles ?? [];
  if (allowed.length > 0 && !roles.some((r) => allowed.includes(r))) {
    throw new Error("Physics engine requires enterprise or admin role. Request access from the admin panel.");
  }
}

async function logJob(userId: string, kind: string, input: unknown, result: unknown, durationMs: number, error?: string) {
  await supabaseAdmin.from("physics_jobs").insert({
    user_id: userId, kind,
    status: error ? "failed" : "completed",
    input: input as never,
    result: error ? null : (result as never),
    error: error ?? null,
    duration_ms: durationMs,
  });
}

// ── Schemas ──
const Vec3 = z.object({ x: z.number(), y: z.number(), z: z.number() });
const LoadProfile = z.object({ force: z.number().optional(), direction: Vec3.optional() }).optional();
const AnalyzeSchema = z.object({
  geometry: z.object({ volume: z.number().positive(), surfaceArea: z.number().positive() }),
  material: z.string().max(64).optional(),
  crossSectionalArea: z.number().positive().optional(),
  momentOfInertia: z.number().positive().optional(),
  beamLength: z.number().positive().optional(),
  loadProfile: LoadProfile,
  geometryType: z.string().max(32).optional(),
  geometryId: z.string().max(64).optional(),
});

const OptimizeSchema = z.object({
  baseGeometry: z.object({
    id: z.string().max(64), name: z.string().max(128),
    area: z.number().positive(), volume: z.number().positive(),
    length: z.number().positive(), momentOfInertia: z.number().positive(),
    crossSectionalArea: z.number().positive(),
  }),
  force: z.number().optional(),
  direction: Vec3.optional(),
  geometryType: z.string().max(32).optional(),
  constraints: z.object({
    minSafetyFactor: z.number().optional(),
    maxDeflectionMm: z.number().optional(),
    minSpanRatio: z.number().optional(),
  }).optional(),
  useAllGeometries: z.boolean().optional(),
});

// ── Server functions ──
export const analyzePhysics = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) => AnalyzeSchema.parse(input))
  .handler(async ({ data, context }) => {
    await assertPhysicsAccess(context.userId);
    const t0 = Date.now();
    try {
      const r = handleSingle(data as AnalyzeBody);
      if (r.error) throw new Error(r.error);
      await logJob(context.userId, "analyze", data, r.data, Date.now() - t0);
      return r.data;
    } catch (e) {
      await logJob(context.userId, "analyze", data, null, Date.now() - t0, String(e));
      throw e;
    }
  });

export const batchAnalyze = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) => AnalyzeSchema.parse(input))
  .handler(async ({ data, context }) => {
    await assertPhysicsAccess(context.userId);
    const t0 = Date.now();
    const r = handleBatch(data as AnalyzeBody);
    await logJob(context.userId, "batch", data, r.data, Date.now() - t0);
    return r.data;
  });

export const compareEngines = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) => AnalyzeSchema.parse(input))
  .handler(async ({ data, context }) => {
    await assertPhysicsAccess(context.userId);
    const t0 = Date.now();
    const r = handleCompare(data as AnalyzeBody);
    if ("error" in r) throw new Error(r.error);
    await logJob(context.userId, "compare", data, r.data, Date.now() - t0);
    return r.data;
  });

export const runPipeline = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) => AnalyzeSchema.parse(input))
  .handler(async ({ data, context }) => {
    await assertPhysicsAccess(context.userId);
    const t0 = Date.now();
    const r = handleSingle(data as AnalyzeBody);
    if (r.error) throw new Error(r.error);
    const d = r.data!;
    const out = {
      source: d.engineSource,
      confidence: d.confidence?.overall,
      confidenceBreakdown: d.confidence,
      load: d.load, stress: d.stress, safety: d.safety, deflection: d.deflection,
      cost: d.cost, recommendations: d.recommendations,
      materialName: d.material.name,
      timestamp: d.meta.computedAt,
    };
    await logJob(context.userId, "pipeline", data, out, Date.now() - t0);
    return out;
  });

export const optimizeDesign = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) => OptimizeSchema.parse(input))
  .handler(async ({ data, context }) => {
    await assertPhysicsAccess(context.userId);
    const t0 = Date.now();
    const r = handleOptimize(data);
    if (r.error) throw new Error(r.error);
    await logJob(context.userId, "optimize", data, { optimal: r.data!.optimal, feasibleCount: r.data!.feasibleCount }, Date.now() - t0);
    return r.data;
  });

export const validateEngine = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    await assertPhysicsAccess(context.userId);
    const t0 = Date.now();
    const r = handleValidate();
    await logJob(context.userId, "validate", {}, r.data, Date.now() - t0);
    return r.data;
  });

export const listMaterials = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    await assertPhysicsAccess(context.userId);
    return {
      keys: ALL_MATERIAL_KEYS,
      materials: Object.fromEntries(ALL_MATERIAL_KEYS.map((k) => [k, MATERIALS[k]])),
      presets: GEOMETRY_PRESETS,
    };
  });

export const listMyJobs = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    const { data, error } = await supabaseAdmin
      .from("physics_jobs")
      .select("id, kind, status, duration_ms, created_at, error")
      .eq("user_id", context.userId)
      .order("created_at", { ascending: false })
      .limit(50);
    if (error) throw new Error(error.message);
    return data ?? [];
  });
