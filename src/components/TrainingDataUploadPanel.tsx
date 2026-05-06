/**
 * TrainingDataUploadPanel
 *
 * Unified ingestion surface for the four training-data channels the models
 * consume: geometry embeddings, simulation outputs, fabrication telemetry,
 * and QA inspection results. Each channel:
 *   - accepts CSV / JSON / JSONL files (parsed entirely client-side)
 *   - lists uploaded datasets with row counts, columns, and a remove action
 *   - shows an expected-schema hint so users know what fields to provide
 *
 * Datasets are pushed into a shared `trainingStore` that other panels
 * (LearningEngine, FabFeedback, etc.) can subscribe to. A "Use real data"
 * badge on each section reflects whether the channel currently has any
 * uploaded rows — handy as a quick "synthetic vs measured" indicator.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Upload, Trash2, FileText, AlertTriangle, CheckCircle2, Database } from "lucide-react";
import {
  trainingStore, ingestFile,
  type Channel, type Dataset,
} from "@/lib/trainingDataStore";

interface ChannelSpec {
  id: Channel;
  title: string;
  blurb: string;
  expected: string[];
  example: string;
}

const SPECS: ChannelSpec[] = [
  {
    id: "geometry",
    title: "Geometry embeddings",
    blurb: "Per-part feature vectors from your CAD pipeline (latent dims, region tags, manifold flags).",
    expected: ["part_id", "embedding_0", "embedding_1", "…", "manifold", "region_tag"],
    example: "embeddings.csv · part_id,embedding_0,embedding_1,manifold,region_tag",
  },
  {
    id: "simulation",
    title: "Simulation outputs",
    blurb: "Per-step solver telemetry: energy, constraint residuals, divergence, runtime.",
    expected: ["t", "energy_drift_pct", "constraint_l2", "divergence_risk", "step_ms", "seed"],
    example: "sim_run_42.json · [{ t, energy_drift_pct, constraint_l2, divergence_risk, step_ms, seed }]",
  },
  {
    id: "fabrication",
    title: "Fabrication telemetry",
    blurb: "Process signals from your line: tool wear, feed rate, temperature, vibration.",
    expected: ["machine_id", "ts", "tool_wear", "feed_rate", "spindle_temp_c", "vibration_rms"],
    example: "machine_07.csv · machine_id,ts,tool_wear,feed_rate,spindle_temp_c,vibration_rms",
  },
  {
    id: "qa",
    title: "QA inspection results",
    blurb: "Pass/fail labels and measurement deltas tied back to part_id for supervised tuning.",
    expected: ["part_id", "pass", "deviation_um", "defect_class", "inspector"],
    example: "qa_batch.jsonl · {\"part_id\":\"…\",\"pass\":1,\"deviation_um\":12.4,\"defect_class\":\"none\"}",
  },
];

export function TrainingDataUploadPanel() {
  // Force re-render whenever the store changes.
  const [, setTick] = useState(0);
  useEffect(() => trainingStore.subscribe(() => setTick((n) => n + 1)), []);

  const totals = useMemo(
    () => SPECS.map((s) => ({ id: s.id, n: trainingStore.rowCount(s.id) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  );

  const grandTotal = totals.reduce((s, t) => s + trainingStore.rowCount(t.id), 0);

  return (
    <div className="space-y-4">
      <div className="flex items-baseline justify-between flex-wrap gap-2">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            training corpus
          </div>
          <h2 className="font-display text-2xl text-foreground">
            Real data <span className="text-primary">in</span>, synthetic out.
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-[10px]">
            <Database className="h-3 w-3 mr-1" />
            {grandTotal.toLocaleString()} rows total
          </Badge>
          {grandTotal > 0 && (
            <Button size="sm" variant="outline" className="h-6 text-[10px]"
              onClick={() => trainingStore.clear()}>
              clear all
            </Button>
          )}
        </div>
      </div>

      <p className="text-xs text-muted-foreground -mt-2">
        Upload measured datasets per channel. Files are parsed locally; downstream
        panels (Learning Engine, Fab Feedback) auto-switch from synthetic samples
        to your data once any channel has rows.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
        {SPECS.map((spec) => (
          <ChannelCard key={spec.id} spec={spec} />
        ))}
      </div>
    </div>
  );
}

function ChannelCard({ spec }: { spec: ChannelSpec }) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const datasets: Dataset[] = trainingStore.list(spec.id);
  const totalRows = datasets.reduce((s, d) => s + d.rows, 0);

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setBusy(true);
    setError(null);
    try {
      for (const f of Array.from(files)) {
        await ingestFile(spec.id, f);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="rounded-lg border border-border bg-background/40 p-3">
      <div className="flex items-start justify-between gap-2 mb-2">
        <div>
          <div className="text-sm font-semibold flex items-center gap-2">
            {spec.title}
            {totalRows > 0 ? (
              <Badge variant="default" className="text-[9px] h-4">
                <CheckCircle2 className="h-2.5 w-2.5 mr-0.5" /> real data
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[9px] h-4 text-muted-foreground">
                synthetic
              </Badge>
            )}
          </div>
          <p className="text-[11px] text-muted-foreground mt-0.5">{spec.blurb}</p>
        </div>
        <Badge variant="outline" className="text-[10px] shrink-0">
          {datasets.length} file{datasets.length === 1 ? "" : "s"} · {totalRows.toLocaleString()} rows
        </Badge>
      </div>

      <div className="text-[10px] font-mono text-muted-foreground/80 mb-1">
        expected: {spec.expected.join(", ")}
      </div>
      <div className="text-[10px] font-mono text-muted-foreground/60 mb-2 truncate">
        {spec.example}
      </div>

      <div className="flex items-center gap-2 mb-2">
        <Input
          ref={inputRef}
          type="file"
          multiple
          accept=".csv,.json,.jsonl,text/csv,application/json"
          onChange={(e) => handleFiles(e.target.files)}
          disabled={busy}
          className="text-xs h-8"
        />
        <Button size="sm" variant="ghost" className="h-8"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
        >
          <Upload className="h-3 w-3" />
        </Button>
      </div>

      {error && (
        <div className="flex items-start gap-1.5 text-[10px] text-destructive mb-2">
          <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {datasets.length > 0 && (
        <ul className="space-y-1 max-h-36 overflow-auto">
          {datasets.map((d) => (
            <li key={d.id}
              className="flex items-center gap-2 text-[10px] font-mono rounded border border-border/60 bg-muted/20 px-2 py-1">
              <FileText className="h-3 w-3 text-primary shrink-0" />
              <span className="truncate flex-1" title={d.name}>{d.name}</span>
              <span className="text-muted-foreground shrink-0">
                {d.rows.toLocaleString()}r · {d.columns.length}c · {(d.bytes / 1024).toFixed(1)}KB
              </span>
              <button
                onClick={() => trainingStore.remove(spec.id, d.id)}
                className="text-muted-foreground hover:text-destructive shrink-0"
                title="Remove"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
