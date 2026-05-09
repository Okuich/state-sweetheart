/**
 * Tiny structured logger for server routes / functions.
 *
 * Emits single-line JSON so logs are grep- and ingest-friendly. Stays in pure
 * JS so it works inside the Worker SSR runtime without any Node-only deps.
 */
type Level = "info" | "warn" | "error";

export function log(level: Level, event: string, fields: Record<string, unknown> = {}): void {
  const line = JSON.stringify({ ts: new Date().toISOString(), level, event, ...fields });
  // eslint-disable-next-line no-console
  (level === "error" ? console.error : level === "warn" ? console.warn : console.log)(line);
}
