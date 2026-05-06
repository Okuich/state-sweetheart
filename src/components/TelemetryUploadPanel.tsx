/**
 * TelemetryUploadPanel
 *
 * Lets users ingest a measured sensor time-series (CSV or JSON) instead of
 * relying on synthetic data. Parsing happens entirely client-side; rows are
 * mapped onto the TelemetrySample schema then POSTed in batches to
 * /api/public/telemetry, where they flow through the same bus as live SSE.
 *
 * Supported formats:
 *   - CSV with header row. Recognised columns (case-insensitive, common
 *     aliases): t/time/timestamp, energy_drift_pct/drift, constraint_l2,
 *     divergence_risk/risk, velocity_max/vmax, nan_count/nans, source.
 *   - JSON: either an array of samples or { samples: [...] }.
 *
 * Unknown columns are ignored. Missing `t` is auto-filled with a monotonic
 * counter so the alerts engine can still order rows. Throughput is capped
 * client-side via a configurable replay rate so live dashboards don't get
 * blasted with thousands of points at once.
 */
import { useCallback, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Upload, FileText, Play, Square, AlertTriangle, CheckCircle2 } from "lucide-react";
import type { TelemetrySample } from "@/lib/anomalyAlerts";

const NUMERIC_KEYS = [
  "t", "energy_drift_pct", "constraint_l2",
  "divergence_risk", "velocity_max", "nan_count",
] as const;

const ALIASES: Record<string, keyof TelemetrySample> = {
  t: "t", time: "t", timestamp: "t", ts: "t",
  energy_drift_pct: "energy_drift_pct", drift: "energy_drift_pct", energy_drift: "energy_drift_pct",
  constraint_l2: "constraint_l2", constraint: "constraint_l2", l2: "constraint_l2",
  divergence_risk: "divergence_risk", divergence: "divergence_risk", risk: "divergence_risk",
  velocity_max: "velocity_max", vmax: "velocity_max", v_max: "velocity_max",
  nan_count: "nan_count", nans: "nan_count", nan: "nan_count",
  source: "source", sensor: "source", channel: "source",
};

type UploadStatus = "idle" | "parsing" | "ready" | "uploading" | "done" | "error";

function parseCSV(text: string): TelemetrySample[] {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error("CSV needs a header row and at least one data row");
  const header = lines[0].split(",").map((h) => h.trim().toLowerCase());
  const cols = header.map((h) => ALIASES[h] ?? null);
  const out: TelemetrySample[] = [];
  let auto = 0;
  for (let i = 1; i < lines.length; i++) {
    const cells = lines[i].split(",");
    const row: Partial<TelemetrySample> = {};
    for (let j = 0; j < cols.length; j++) {
      const key = cols[j];
      if (!key) continue;
      const raw = (cells[j] ?? "").trim();
      if (raw === "") continue;
      if (key === "source") {
        row.source = raw.slice(0, 64);
      } else if ((NUMERIC_KEYS as readonly string[]).includes(key)) {
        const n = Number(raw);
        if (Number.isFinite(n)) (row as Record<string, number>)[key] = n;
      }
    }
    if (typeof row.t !== "number") row.t = auto++;
    out.push(row as TelemetrySample);
  }
  return out;
}

function parseJSON(text: string): TelemetrySample[] {
  const data = JSON.parse(text);
  const arr: unknown[] = Array.isArray(data)
    ? data
    : Array.isArray((data as { samples?: unknown[] })?.samples)
      ? (data as { samples: unknown[] }).samples
      : [];
  if (arr.length === 0) throw new Error("JSON must be an array or { samples: [...] }");
  let auto = 0;
  return arr.map((r) => {
    const obj = (r ?? {}) as Record<string, unknown>;
    const out: Partial<TelemetrySample> = {};
    for (const [k, v] of Object.entries(obj)) {
      const key = ALIASES[k.toLowerCase()];
      if (!key) continue;
      if (key === "source" && typeof v === "string") out.source = v.slice(0, 64);
      else if (typeof v === "number" && Number.isFinite(v)) {
        (out as Record<string, number>)[key] = v;
      }
    }
    if (typeof out.t !== "number") out.t = auto++;
    return out as TelemetrySample;
  });
}

export function TelemetryUploadPanel() {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const cancelRef = useRef(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [samples, setSamples] = useState<TelemetrySample[]>([]);
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [sent, setSent] = useState(0);
  const [batchSize, setBatchSize] = useState(64);
  const [rateHz, setRateHz] = useState(20); // batches per second

  const preview = useMemo(() => samples.slice(0, 3), [samples]);

  const handleFile = useCallback(async (file: File) => {
    setStatus("parsing");
    setError(null);
    setSent(0);
    setProgress(0);
    setFileName(file.name);
    try {
      const text = await file.text();
      const parsed = file.name.toLowerCase().endsWith(".json")
        ? parseJSON(text)
        : parseCSV(text);
      if (parsed.length === 0) throw new Error("No samples parsed");
      setSamples(parsed);
      setStatus("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
      setSamples([]);
    }
  }, []);

  const upload = useCallback(async () => {
    if (samples.length === 0) return;
    setStatus("uploading");
    cancelRef.current = false;
    const interval = Math.max(10, Math.floor(1000 / Math.max(1, rateHz)));
    let i = 0;
    while (i < samples.length && !cancelRef.current) {
      const chunk = samples.slice(i, i + batchSize);
      try {
        const res = await fetch("/api/public/telemetry", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ samples: chunk }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setStatus("error");
        return;
      }
      i += chunk.length;
      setSent(i);
      setProgress(Math.round((i / samples.length) * 100));
      await new Promise((r) => setTimeout(r, interval));
    }
    setStatus(cancelRef.current ? "ready" : "done");
  }, [samples, batchSize, rateHz]);

  const cancel = () => { cancelRef.current = true; };

  const reset = () => {
    setSamples([]); setFileName(null); setSent(0);
    setProgress(0); setStatus("idle"); setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="mt-6 rounded-lg border border-border bg-background/40 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Upload className="h-4 w-4 text-primary" />
          <h4 className="text-sm font-semibold">Sensor Time-Series Upload</h4>
        </div>
        <Badge variant="outline" className="text-[10px]">CSV · JSON</Badge>
      </div>
      <p className="text-xs text-muted-foreground mb-3">
        Replace synthetic telemetry by uploading a measured sensor file. Rows
        are streamed to the live ingest endpoint and feed the alerts engine.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-3">
        <div className="md:col-span-3">
          <Label htmlFor="telemetry-file" className="text-xs">File</Label>
          <Input
            id="telemetry-file"
            ref={inputRef}
            type="file"
            accept=".csv,.json,text/csv,application/json"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) handleFile(f);
            }}
            disabled={status === "uploading" || status === "parsing"}
            className="text-xs"
          />
        </div>
        <div>
          <Label htmlFor="batch-size" className="text-xs">Batch size</Label>
          <Input
            id="batch-size"
            type="number" min={1} max={256}
            value={batchSize}
            onChange={(e) => setBatchSize(Math.max(1, Math.min(256, Number(e.target.value) || 1)))}
            disabled={status === "uploading"}
            className="text-xs"
          />
        </div>
        <div>
          <Label htmlFor="rate-hz" className="text-xs">Replay rate (batches/s)</Label>
          <Input
            id="rate-hz"
            type="number" min={1} max={100}
            value={rateHz}
            onChange={(e) => setRateHz(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}
            disabled={status === "uploading"}
            className="text-xs"
          />
        </div>
        <div className="flex items-end gap-2">
          {status !== "uploading" ? (
            <Button
              size="sm" onClick={upload}
              disabled={samples.length === 0 || status === "parsing"}
              className="flex-1"
            >
              <Play className="h-3 w-3 mr-1" /> Stream
            </Button>
          ) : (
            <Button size="sm" variant="destructive" onClick={cancel} className="flex-1">
              <Square className="h-3 w-3 mr-1" /> Stop
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={reset} disabled={status === "uploading"}>
            Reset
          </Button>
        </div>
      </div>

      {fileName && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground mb-2">
          <FileText className="h-3 w-3" />
          <span className="truncate">{fileName}</span>
          {samples.length > 0 && (
            <Badge variant="secondary" className="text-[10px]">
              {samples.length.toLocaleString()} samples
            </Badge>
          )}
        </div>
      )}

      {(status === "uploading" || status === "done") && (
        <div className="mb-2">
          <Progress value={progress} className="h-1.5" />
          <div className="flex justify-between text-[10px] text-muted-foreground mt-1">
            <span>{sent.toLocaleString()} / {samples.length.toLocaleString()}</span>
            <span>{progress}%</span>
          </div>
        </div>
      )}

      {status === "done" && (
        <div className="flex items-center gap-2 text-xs text-emerald-500">
          <CheckCircle2 className="h-3 w-3" />
          Streamed {sent.toLocaleString()} samples to telemetry bus.
        </div>
      )}

      {error && (
        <div className="flex items-start gap-2 text-xs text-destructive">
          <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {preview.length > 0 && status !== "uploading" && (
        <div className="mt-3 rounded border border-border/60 bg-muted/20 p-2">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-1">
            Preview (first {preview.length})
          </div>
          <pre className="text-[10px] font-mono overflow-x-auto leading-tight">
{preview.map((s) => JSON.stringify(s)).join("\n")}
          </pre>
        </div>
      )}
    </div>
  );
}
