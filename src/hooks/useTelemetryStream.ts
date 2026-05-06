/**
 * useTelemetryStream
 *
 * Subscribes to /api/public/telemetry/stream via EventSource. Returns the
 * latest sample plus a connection state indicator. Falls back to "idle"
 * if SSE is unavailable in the runtime (e.g. SSR / older Safari) so
 * callers can use a synthetic source instead.
 */
import { useEffect, useRef, useState } from "react";
import type { TelemetrySample } from "@/lib/anomalyAlerts";

export type StreamStatus = "idle" | "connecting" | "open" | "closed" | "error";

export function useTelemetryStream(url = "/api/public/telemetry/stream") {
  const [status, setStatus] = useState<StreamStatus>("idle");
  const [sample, setSample] = useState<TelemetrySample | null>(null);
  const [count, setCount] = useState(0);
  const seqRef = useRef(0);

  useEffect(() => {
    if (typeof EventSource === "undefined") {
      setStatus("error");
      return;
    }
    setStatus("connecting");
    const es = new EventSource(url);
    es.addEventListener("open", () => setStatus("open"));
    es.addEventListener("error", () => setStatus("error"));
    es.addEventListener("sample", (ev) => {
      try {
        const parsed = JSON.parse((ev as MessageEvent).data) as TelemetrySample;
        // Stamp a monotonic t if upstream omitted one — keeps the alerts
        // engine happy even with sparse sources.
        if (typeof parsed.t !== "number" || !Number.isFinite(parsed.t)) {
          parsed.t = ++seqRef.current;
        }
        setSample(parsed);
        setCount((n) => n + 1);
      } catch { /* skip malformed event */ }
    });
    return () => { es.close(); setStatus("closed"); };
  }, [url]);

  return { status, sample, count };
}
