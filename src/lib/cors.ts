/**
 * CORS helper for public API routes.
 *
 * Origins are read from `TELEMETRY_ALLOWED_ORIGINS` (comma-separated).
 * - In production (NODE_ENV=production) with NO allowlist set, we fail closed
 *   (omit Access-Control-Allow-Origin) so a forgotten env var doesn't expose
 *   the surface to every browser origin.
 * - In dev/preview we fall back to "*" for ergonomics.
 */
const RAW = (typeof process !== "undefined" ? process.env.TELEMETRY_ALLOWED_ORIGINS : "") ?? "";
const ALLOWLIST = RAW.split(",").map((s) => s.trim()).filter(Boolean);
const IS_PROD = (typeof process !== "undefined" ? process.env.NODE_ENV : "") === "production";

export function corsHeaders(request: Request): Record<string, string> {
  const origin = request.headers.get("origin") ?? "";
  const base: Record<string, string> = {
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Telemetry-Token",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin",
  };

  if (ALLOWLIST.length > 0) {
    if (ALLOWLIST.includes(origin)) base["Access-Control-Allow-Origin"] = origin;
    // No matching origin → omit the header (browser blocks the request).
    return base;
  }

  // Allowlist not configured.
  if (IS_PROD) return base; // fail closed in production
  base["Access-Control-Allow-Origin"] = "*";
  return base;
}
