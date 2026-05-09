/**
 * Server functions for managing API clients (Fabrication OS, Midwater).
 * Auth-protected — only the signed-in admin (you) can call them.
 */
import { createServerFn } from "@tanstack/react-start";
import { z } from "zod";
import { requireSupabaseAuth } from "@/integrations/supabase/auth-middleware";
import { supabaseAdmin } from "@/integrations/supabase/client.server";
import { generateApiKey } from "./service-auth.server";

const CreateInput = z.object({
  name: z.string().min(1).max(64).regex(/^[a-z0-9-]+$/),
  scopes: z.array(z.string().min(1).max(64)).min(1).max(8),
});

export const createApiClient = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) => CreateInput.parse(d))
  .handler(async ({ data, context }) => {
    const { raw, hash, prefix } = await generateApiKey();
    const { data: row, error } = await supabaseAdmin
      .from("api_clients")
      .insert({
        name: data.name,
        scopes: data.scopes,
        key_hash: hash,
        key_prefix: prefix,
        created_by: context.userId,
      })
      .select("id,name,scopes,key_prefix,created_at")
      .single();
    if (error) throw new Error(error.message);
    // raw key returned ONCE — never stored in plaintext anywhere.
    return { client: row, raw_key: raw };
  });

export const listApiClients = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .handler(async () => {
    const { data, error } = await supabaseAdmin
      .from("api_clients")
      .select("id,name,scopes,key_prefix,created_at,last_used_at,revoked_at")
      .order("created_at", { ascending: false });
    if (error) throw new Error(error.message);
    return { clients: data ?? [] };
  });

const RevokeInput = z.object({ id: z.string().uuid() });

export const revokeApiClient = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) => RevokeInput.parse(d))
  .handler(async ({ data }) => {
    const { error } = await supabaseAdmin
      .from("api_clients")
      .update({ revoked_at: new Date().toISOString() })
      .eq("id", data.id);
    if (error) throw new Error(error.message);
    return { ok: true };
  });

export const rotateApiClient = createServerFn({ method: "POST" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) => RevokeInput.parse(d))
  .handler(async ({ data }) => {
    const { data: existing, error: loadErr } = await supabaseAdmin
      .from("api_clients")
      .select("id,name,revoked_at")
      .eq("id", data.id)
      .maybeSingle();
    if (loadErr) throw new Error(loadErr.message);
    if (!existing) throw new Error("client not found");
    if (existing.revoked_at) throw new Error("cannot rotate a revoked key");

    const { raw, hash, prefix } = await generateApiKey();
    const { error: upErr } = await supabaseAdmin
      .from("api_clients")
      .update({ key_hash: hash, key_prefix: prefix, last_used_at: null })
      .eq("id", data.id);
    if (upErr) throw new Error(upErr.message);

    return { client: { id: existing.id, name: existing.name }, raw_key: raw };
  });

export const getRequestLog = createServerFn({ method: "GET" })
  .middleware([requireSupabaseAuth])
  .inputValidator((d) => z.object({ clientId: z.string().uuid().optional() }).parse(d ?? {}))
  .handler(async ({ data }) => {
    let q = supabaseAdmin
      .from("api_request_log")
      .select("id,client_id,route,method,status,latency_ms,created_at")
      .order("created_at", { ascending: false })
      .limit(200);
    if (data.clientId) q = q.eq("client_id", data.clientId);
    const { data: rows, error } = await q;
    if (error) throw new Error(error.message);
    return { rows: rows ?? [] };
  });
