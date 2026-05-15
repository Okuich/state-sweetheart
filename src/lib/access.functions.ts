/**
 * Role + feature-flag server functions for the unified Physics OS / Playground platform.
 *
 * - getMyAccess: current user's roles + evaluated flag map (1 round trip)
 * - requestRoleAccess: user-initiated upgrade request (e.g. enterprise / physics_engine)
 * - listRoleRequests / reviewRoleRequest: admin only
 */
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { supabaseAdmin } from "@/integrations/supabase/client.server";

export type AppRole = "user" | "enterprise" | "admin";

export interface AccessSnapshot {
  userId: string;
  roles: AppRole[];
  flags: Record<string, boolean>;
}

const ALL_FLAGS = [
  "physics_engine",
  "ml_training",
  "patent_suite",
  "materials_catalog_edit",
  "admin_panel",
] as const;

function hashToBucket(userId: string, flagKey: string): number {
  const str = `${userId}:${flagKey}`;
  let hash = 0;
  for (let i = 0; i < str.length; i++) hash = ((hash << 5) - hash + str.charCodeAt(i)) | 0;
  return Math.abs(hash) % 100;
}

export const getMyAccess = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }): Promise<AccessSnapshot> => {
    const userId = context.userId;

    const [rolesRes, flagsRes, overridesRes] = await Promise.all([
      supabaseAdmin.from("user_roles").select("role").eq("user_id", userId),
      supabaseAdmin
        .from("feature_flags")
        .select("key, enabled, allowed_roles, rollout_percentage"),
      supabaseAdmin
        .from("user_feature_flags")
        .select("flag_key, enabled")
        .eq("user_id", userId),
    ]);

    const roles = ((rolesRes.data ?? []) as { role: AppRole }[]).map((r) => r.role);
    const flagRows = (flagsRes.data ?? []) as {
      key: string;
      enabled: boolean;
      allowed_roles: string[] | null;
      rollout_percentage: number;
    }[];
    const overrides = new Map<string, boolean>(
      ((overridesRes.data ?? []) as { flag_key: string; enabled: boolean }[]).map((o) => [
        o.flag_key,
        o.enabled,
      ]),
    );

    const flags: Record<string, boolean> = {};
    for (const key of ALL_FLAGS) flags[key] = false;
    for (const f of flagRows) {
      if (overrides.has(f.key)) {
        flags[f.key] = overrides.get(f.key)!;
        continue;
      }
      if (!f.enabled) {
        flags[f.key] = false;
        continue;
      }
      const allowed = f.allowed_roles ?? [];
      if (allowed.length > 0 && roles.some((r) => allowed.includes(r))) {
        flags[f.key] = true;
        continue;
      }
      if (allowed.length === 0 && f.rollout_percentage === 100) {
        flags[f.key] = true;
        continue;
      }
      if (f.rollout_percentage > 0 && f.rollout_percentage < 100) {
        flags[f.key] = hashToBucket(userId, f.key) < f.rollout_percentage;
        continue;
      }
      flags[f.key] = false;
    }

    return { userId, roles, flags };
  });

export const requestRoleAccess = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) =>
    z
      .object({
        requested_role: z.enum(["enterprise", "admin"]).default("enterprise"),
        requested_flag: z.string().max(64).optional(),
        reason: z.string().max(2000).optional(),
      })
      .parse(input),
  )
  .handler(async ({ data, context }) => {
    const { error, data: row } = await supabaseAdmin
      .from("role_requests")
      .insert({
        user_id: context.userId,
        requested_role: data.requested_role,
        requested_flag: data.requested_flag ?? null,
        reason: data.reason ?? null,
      })
      .select("id")
      .single();
    if (error) throw new Error(error.message);
    return { id: row!.id };
  });

export const listRoleRequests = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async ({ context }) => {
    // Admin gate
    const { data: roles } = await supabaseAdmin
      .from("user_roles")
      .select("role")
      .eq("user_id", context.userId);
    const isAdmin = (roles ?? []).some((r) => r.role === "admin");
    if (!isAdmin) throw new Error("Forbidden");
    const { data, error } = await supabaseAdmin
      .from("role_requests")
      .select("id, user_id, requested_role, requested_flag, reason, status, created_at")
      .order("created_at", { ascending: false })
      .limit(200);
    if (error) throw new Error(error.message);
    return data ?? [];
  });

export const reviewRoleRequest = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((input) =>
    z
      .object({
        request_id: z.string().uuid(),
        approve: z.boolean(),
      })
      .parse(input),
  )
  .handler(async ({ data, context }) => {
    const { data: roles } = await supabaseAdmin
      .from("user_roles")
      .select("role")
      .eq("user_id", context.userId);
    const isAdmin = (roles ?? []).some((r) => r.role === "admin");
    if (!isAdmin) throw new Error("Forbidden");

    const { data: req, error: reqErr } = await supabaseAdmin
      .from("role_requests")
      .select("id, user_id, requested_role, status")
      .eq("id", data.request_id)
      .single();
    if (reqErr || !req) throw new Error("Request not found");
    if (req.status !== "pending") throw new Error("Already reviewed");

    if (data.approve) {
      await supabaseAdmin
        .from("user_roles")
        .upsert(
          { user_id: req.user_id, role: req.requested_role },
          { onConflict: "user_id,role", ignoreDuplicates: true },
        );
    }

    await supabaseAdmin
      .from("role_requests")
      .update({
        status: data.approve ? "approved" : "rejected",
        reviewed_by: context.userId,
        reviewed_at: new Date().toISOString(),
      })
      .eq("id", data.request_id);

    return { ok: true };
  });
