/**
 * CORS helper for public API routes.
 *
 * Origins are read from `TELEMETRY_ALLOWED_ORIGINS` (comma-separated). When the
 * env var is unset or empty we fall back to "*" so dev/preview keeps working;
 * set the env var in production to lock the surface down.
 */
const RAW = (typeof process !== "undefined" ? process.env.TELEMETRY_ALLOWED_ORIGINS : "") ?? "";
const ALLOWLIST = RAW.split(",").map((s) => s.trim()).filter(Boolean);

export function corsHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get("origin") ?? "";
  let allow = "*";
  if (ALLOWLIST.length > 0) {
    allow = ALLOWLIST.includes(origin) ? origin : ALLOWLIST[0];
  }
  return {
    "Access-Control-Allow-Origin": allow,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Telemetry-Token",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };
}
