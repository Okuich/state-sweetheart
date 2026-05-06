// Post-Fabrication Scan & Quality Report Import — UI panel
import { useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { UploadCloud, FileWarning, CheckCircle2, X } from "lucide-react";
import {
  applyMapping,
  autoDetectMapping,
  parseScanFile,
  scanImportBridge,
  type FieldMapping,
  type RawRow,
} from "@/lib/scanImportStore";
import { CHANNELS, CHANNEL_UNITS, type Channel } from "@/lib/fabFeedback";

type Loaded = {
  fileName: string;
  rows: RawRow[];
  columns: string[];
};

export function ScanImportPanel() {
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [mapping, setMapping] = useState<FieldMapping | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [imported, setImported] = useState<{ ts: number; n: number; channel: Channel } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const onFile = async (f: File) => {
    setError(null);
    setImported(null);
    try {
      const text = await f.text();
      const rows = parseScanFile(f.name, text);
      if (rows.length === 0) throw new Error("file parsed but contains no rows");
      const columns = Object.keys(rows[0]);
      const m = autoDetectMapping(columns);
      setLoaded({ fileName: f.name, rows, columns });
      setMapping(m);
    } catch (e) {
      setLoaded(null);
      setMapping(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const preview = useMemo(() => {
    if (!loaded || !mapping) return null;
    const result = applyMapping(loaded.rows.slice(0, 100), mapping);
    return result;
  }, [loaded, mapping]);

  const commit = () => {
    if (!loaded || !mapping) return;
    const result = applyMapping(loaded.rows, mapping);
    if (result.observations.length === 0) {
      setError("no rows produced valid observations — check field mapping");
      return;
    }
    scanImportBridge.publish(result.observations);
    setImported({ ts: Date.now(), n: result.observations.length, channel: mapping.channel });
  };

  const clear = () => {
    setLoaded(null);
    setMapping(null);
    setError(null);
    setImported(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  return (
    <div className="space-y-6">
      <header>
        <div className="text-[10px] uppercase tracking-[0.28em] text-muted-foreground">
          module · post-fabrication-scan-import
        </div>
        <h2 className="font-display text-2xl md:text-3xl text-glow">
          Post-Fabrication <span className="text-primary">Scans</span> & Quality Reports
        </h2>
        <p className="text-xs text-muted-foreground max-w-xl mt-1">
          Drop a CSV / JSON inspection report, map its columns onto a
          fabrication channel (dimensional · thermal · tolerance · surface),
          then push the parsed observations into the calibration loop.
        </p>
      </header>

      {/* upload zone */}
      <div className="rounded-lg border border-dashed border-border bg-background/40 p-4 flex flex-wrap items-center gap-3">
        <UploadCloud className="h-5 w-5 text-primary" />
        <div className="flex-1 min-w-[200px]">
          <div className="text-[11px] text-foreground/85">
            upload measured CSV / JSON / JSONL inspection report
          </div>
          <div className="text-[9px] text-muted-foreground uppercase tracking-[0.18em]">
            columns are auto-detected · you can override below
          </div>
        </div>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,.tsv,.json,.jsonl,.ndjson,text/csv,application/json"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
          }}
          className="text-[11px] file:mr-2 file:rounded-md file:border file:border-border file:bg-background file:px-2 file:py-1 file:text-[10px] file:uppercase file:tracking-[0.16em]"
        />
        {loaded && (
          <Button variant="outline" size="sm" onClick={clear}
            className="uppercase tracking-[0.16em] text-[10px]">
            <X className="h-3 w-3 mr-1" /> clear
          </Button>
        )}
      </div>

      {error && (
        <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-[11px] text-destructive flex items-start gap-2">
          <FileWarning className="h-3.5 w-3.5 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {loaded && mapping && (
        <>
          {/* mapping editor */}
          <div className="rounded-lg border border-border bg-background/40 p-3 space-y-3">
            <div className="flex items-baseline justify-between gap-2 flex-wrap">
              <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
                field mapping · {loaded.fileName} · {loaded.rows.length} rows
              </div>
              <Badge variant="outline" className="text-[10px]">
                channel · {mapping.channel} ({CHANNEL_UNITS[mapping.channel]})
              </Badge>
            </div>

            <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-5">
              <Select
                label="channel"
                value={mapping.channel}
                onChange={(v) => setMapping({ ...mapping, channel: v as Channel })}
                options={CHANNELS.map((c) => ({ value: c, label: c }))}
              />
              <Select
                label="predicted column"
                value={mapping.predicted}
                onChange={(v) => setMapping({ ...mapping, predicted: v })}
                options={loaded.columns.map((c) => ({ value: c, label: c }))}
              />
              <Select
                label="measured column"
                value={mapping.measured}
                onChange={(v) => setMapping({ ...mapping, measured: v })}
                options={loaded.columns.map((c) => ({ value: c, label: c }))}
              />
              <Select
                label="part id (opt.)"
                value={mapping.partId ?? ""}
                onChange={(v) => setMapping({ ...mapping, partId: v || undefined })}
                options={[{ value: "", label: "— none —" }, ...loaded.columns.map((c) => ({ value: c, label: c }))]}
              />
              <Select
                label="timestamp (opt.)"
                value={mapping.ts ?? ""}
                onChange={(v) => setMapping({ ...mapping, ts: v || undefined })}
                options={[{ value: "", label: "— now —" }, ...loaded.columns.map((c) => ({ value: c, label: c }))]}
              />
            </div>
          </div>

          {/* preview */}
          {preview && (
            <div className="rounded-lg border border-border bg-background/40 p-3 space-y-2">
              <div className="flex items-center justify-between">
                <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
                  preview · first {Math.min(8, preview.observations.length)} of {preview.observations.length} mapped
                </div>
                {preview.skipped > 0 && (
                  <Badge variant="outline" className="text-[10px] text-destructive border-destructive/40">
                    skipped {preview.skipped}
                  </Badge>
                )}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px] font-mono">
                  <thead className="text-muted-foreground">
                    <tr className="text-left">
                      <th className="py-1 pr-3">part</th>
                      <th className="py-1 pr-3">channel</th>
                      <th className="py-1 pr-3">predicted</th>
                      <th className="py-1 pr-3">measured</th>
                      <th className="py-1 pr-3">residual</th>
                    </tr>
                  </thead>
                  <tbody>
                    {preview.observations.slice(0, 8).map((o) => (
                      <tr key={o.id} className="border-t border-border/40">
                        <td className="py-1 pr-3 text-foreground/80">{o.partId}</td>
                        <td className="py-1 pr-3 text-primary">{o.channel}</td>
                        <td className="py-1 pr-3 tabular-nums">{o.predicted.toFixed(3)}</td>
                        <td className="py-1 pr-3 tabular-nums">{o.measured.toFixed(3)}</td>
                        <td className="py-1 pr-3 tabular-nums text-foreground/70">
                          {(o.measured - o.predicted).toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {preview.reasons.length > 0 && (
                <div className="text-[10px] text-muted-foreground">
                  ⚠ {preview.reasons.join(" · ")}
                </div>
              )}
            </div>
          )}

          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={commit} className="uppercase tracking-[0.16em] text-[10px]">
              push to calibration loop
            </Button>
            {imported && (
              <div className="flex items-center gap-1.5 text-[11px] text-primary">
                <CheckCircle2 className="h-3.5 w-3.5" />
                imported {imported.n} → {imported.channel}
                <span className="text-muted-foreground">
                  · {new Date(imported.ts).toLocaleTimeString()}
                </span>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

function Select({
  label, value, onChange, options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-border bg-background/60 px-2 py-1.5 font-mono text-xs outline-none focus:ring-1 focus:ring-primary"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}
