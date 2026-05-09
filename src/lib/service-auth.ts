/**
 * Service-to-service auth for /api/public/step/* routes.
 * Server-only (uses node:crypto and supabaseAdmin).
 */
async function admin() {
  const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
  return supabaseAdmin;
}

const KEY_PREFIX = "pde_";

export async function generateApiKey(): Promise<{ raw: string; hash: string; prefix: string }> {
  const { randomBytes, createHash } = await import("node:crypto");
  const body = randomBytes(32).toString("base64url");
  const raw = `${KEY_PREFIX}${body}`;
  const hash = createHash("sha256").update(raw).digest("hex");
  const prefix = raw.slice(0, 12); // pde_ + 8 chars
  return { raw, hash, prefix };
}

export async function hashKey(raw: string): Promise<string> {
  const { createHash } = await import("node:crypto");
  return createHash("sha256").update(raw).digest("hex");
}

export interface AuthedClient {
  id: string;
  name: string;
  scopes: string[];
}

/**
 * Verify the bearer token on a Request. Returns the client row if valid,
 * otherwise null. Logs the request to api_request_log.
 */
export async function verifyServiceAuth(
  request: Request,
  route: string,
  requiredScope: string,
): Promise<AuthedClient | null> {
  const auth = request.headers.get("authorization") ?? "";
  const match = auth.match(/^Bearer\s+(\S+)$/i);
  if (!match) return null;
  const raw = match[1];
  if (!raw.startsWith(KEY_PREFIX)) return null;

  const hash = await hashKey(raw);

  const { data, error } = await getAdmin()
    .from("api_clients")
    .select("id,name,scopes,revoked_at")
    .eq("key_hash", hash)
    .maybeSingle();

  if (error || !data) return null;
  if (data.revoked_at) return null;
  if (!data.scopes?.includes(requiredScope)) return null;

  // touch last_used_at (fire and forget)
  void getAdmin()
    .from("api_clients")
    .update({ last_used_at: new Date().toISOString() })
    .eq("id", data.id);

  return { id: data.id, name: data.name, scopes: data.scopes };
}

export async function logRequest(
  clientId: string | null,
  route: string,
  method: string,
  status: number,
  latencyMs: number,
): Promise<void> {
  void getAdmin().from("api_request_log").insert({
    client_id: clientId,
    route,
    method,
    status,
    latency_ms: latencyMs,
  });
}
