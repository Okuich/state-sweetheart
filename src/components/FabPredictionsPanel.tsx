/**
 * Midwater · prediction & pipeline status feed.
 * Renders per-part processing progress and the resulting predictions.
 * The underlying engine is intentionally not named in the UI.
 */
import { Fragment, useEffect, useState } from "react";
import { Loader2, RotateCw, X, AlertTriangle, ChevronRight } from "lucide-react";
import {
  physicsFabFeed,
  type FeedSnapshot,
  type PartStatus,
} from "@/lib/physicsFabBridge";

function fmtTime(iso: string) {
  const d = new Date(iso);
  return d.toLocaleTimeString(undefined, { hour12: false }) +
    "." + String(d.getMilliseconds()).padStart(3, "0");
}

function fmt(n: number | undefined, digits = 2, unit = "") {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return `${n.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

const STATUS_STYLES: Record<PartStatus, string> = {
  queued: "bg-muted text-muted-foreground",
  processing: "bg-primary/15 text-primary animate-pulse",
  ready: "bg-emerald-500/15 text-emerald-500",
  failed: "bg-destructive/15 text-destructive",
};

const STATUS_LABEL: Record<PartStatus, string> = {
  queued: "queued",
  processing: "processing",
  ready: "ready",
  failed: "failed",
};

function StatusPill({ status }: { status: PartStatus }) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.14em] ${STATUS_STYLES[status]}`}
    >
      {status === "processing" || status === "queued" ? (
        <Loader2 className="h-2.5 w-2.5 animate-spin" />
      ) : (
        <span className="h-1.5 w-1.5 rounded-full bg-current" />
      )}
      {STATUS_LABEL[status]}
    </span>
  );
}

export function FabPredictionsPanel() {
  const [snap, setSnap] = useState<FeedSnapshot>(() => physicsFabFeed.snapshot());
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());

  useEffect(() => physicsFabFeed.subscribe(setSnap), []);

  const toggle = (partId: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(partId)) next.delete(partId);
      else next.add(partId);
      return next;
    });
  };

  const { predictions, progress, counts } = snap;
  const active = counts.queued + counts.processing;
  const total = progress.length;
  const pct = total === 0 ? 0 : Math.round(((counts.ready + counts.failed) / total) * 100);

  // Merge progress + predictions for the table view.
  const predById = new Map(predictions.map((p) => [p.partId, p]));
  const rows = progress.slice(0, 50).map((p) => ({
    progress: p,
    pred: predById.get(p.partId),
  }));

  return (
    <section className="rounded-xl border border-border bg-card p-6">
      <header className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
            Midwater · pipeline status
          </div>
          <h2 className="mt-1 text-lg font-semibold text-foreground">
            Per-part processing
          </h2>
          <p className="mt-1 max-w-xl text-xs text-muted-foreground">
            Each imported part flows through the Midwater pipeline: queued →
            processing → ready. Predictions feed calibration as they land.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-right">
            <div className="font-mono text-sm tabular-nums text-foreground">
              {pct}%
            </div>
            <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
              {active > 0 ? `${active} in flight` : "idle"}
            </div>
          </div>
          <div className="h-8 w-32 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full bg-primary transition-all duration-300"
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </header>

      {/* Status totals */}
      <div className="mb-4 grid grid-cols-4 gap-2 text-center text-[10px] uppercase tracking-[0.16em]">
        {(["queued", "processing", "ready", "failed"] as PartStatus[]).map((s) => (
          <div
            key={s}
            className="rounded-md border border-border bg-background/40 px-2 py-2"
          >
            <div className={`font-mono text-base tabular-nums ${
              s === "failed" ? "text-destructive" :
              s === "ready" ? "text-emerald-500" :
              s === "processing" ? "text-primary" : "text-foreground"
            }`}>
              {counts[s]}
            </div>
            <div className="text-muted-foreground">{s}</div>
          </div>
        ))}
      </div>

      {/* Failed-jobs banner */}
      {(() => {
        const failed = progress.filter((p) => p.status === "failed");
        if (failed.length === 0) return null;
        const last = failed[0];
        return (
          <div className="mb-4 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2.5">
            <div className="flex items-start justify-between gap-3">
              <div className="flex items-start gap-2 min-w-0">
                <AlertTriangle className="h-4 w-4 shrink-0 text-destructive mt-0.5" />
                <div className="min-w-0">
                  <div className="text-[11px] font-semibold text-destructive">
                    {failed.length === 1
                      ? `Prediction failed for ${last.partId}`
                      : `${failed.length} predictions failed — last: ${last.partId}`}
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[10px] text-destructive/80" title={last.error ?? ""}>
                    {last.error ?? "unknown error"}
                  </div>
                  <div className="mt-0.5 text-[10px] text-muted-foreground">
                    Auto-retried {last.attempt}× before giving up. Use re-run to try again.
                  </div>
                </div>
              </div>
              <div className="flex shrink-0 gap-1.5">
                <button
                  type="button"
                  onClick={() => physicsFabFeed.predict(last.partId)}
                  className="inline-flex items-center gap-1 rounded-md border border-destructive/40 bg-background/40 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-destructive transition hover:bg-destructive/10"
                >
                  <RotateCw className="h-3 w-3" />
                  re-run last
                </button>
                <button
                  type="button"
                  onClick={() => physicsFabFeed.dismissAllFailed()}
                  className="inline-flex items-center gap-1 rounded-md border border-border bg-background/40 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-muted-foreground transition hover:text-foreground"
                >
                  <X className="h-3 w-3" />
                  dismiss {failed.length > 1 ? "all" : ""}
                </button>
              </div>
            </div>
          </div>
        );
      })()}

      {rows.length === 0 ? (
        <div className="rounded-md border border-dashed border-border bg-background/40 px-4 py-10 text-center text-xs text-muted-foreground">
          Import a scan or quality report to start the pipeline.
        </div>
      ) : (
        <div className="overflow-hidden rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/40 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              <tr>
                <th className="w-6 px-2 py-2"></th>
                <th className="px-3 py-2">part</th>
                <th className="px-3 py-2">status</th>
                <th className="px-3 py-2">material</th>
                <th className="px-3 py-2">stress (MPa)</th>
                <th className="px-3 py-2">defl (mm)</th>
                <th className="px-3 py-2">safety</th>
                <th className="px-3 py-2">cost</th>
                <th className="px-3 py-2">conf</th>
                <th className="px-3 py-2">ms</th>
                <th className="px-3 py-2 text-right">re-run</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[11px]">
              {rows.map(({ progress: pr, pred }) => {
                const hist = snap.history[pr.partId] ?? [];
                const isOpen = expanded.has(pr.partId);
                return (
                <>
                <tr key={pr.partId} className="border-t border-border">
                  <td className="px-2 py-2">
                    <button
                      type="button"
                      onClick={() => toggle(pr.partId)}
                      disabled={hist.length === 0}
                      title={hist.length === 0 ? "No history yet" : `${hist.length} run${hist.length === 1 ? "" : "s"}`}
                      className="inline-flex h-5 w-5 items-center justify-center rounded text-muted-foreground transition hover:text-foreground disabled:opacity-30"
                    >
                      <ChevronRight className={`h-3 w-3 transition-transform ${isOpen ? "rotate-90" : ""}`} />
                    </button>
                  </td>
                  <td className="px-3 py-2 text-foreground">{pr.partId}</td>
                  <td className="px-3 py-2">
                    <StatusPill status={pr.status} />
                  </td>
                  <td className="px-3 py-2 text-muted-foreground">
                    {pred?.materialName ?? "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {pred?.ok
                      ? fmt(pred.predictedStressMPa)
                      : pr.status === "failed"
                        ? <span className="text-destructive">err</span>
                        : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {pred?.ok ? fmt(pred.predictedDeflectionMm) : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {pred?.ok ? fmt(pred.safetyFactor) : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums">
                    {pred?.ok ? fmt(pred.costUsd, 2, "$") : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-primary">
                    {pred?.ok && pred.confidence !== undefined
                      ? `${(pred.confidence * 100).toFixed(0)}%`
                      : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-muted-foreground">
                    {pr.ms ?? "—"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    <div className="inline-flex gap-1">
                      <button
                        type="button"
                        onClick={() => physicsFabFeed.predict(pr.partId)}
                        disabled={pr.status === "queued" || pr.status === "processing"}
                        title={
                          pr.status === "failed" && pr.error
                            ? `Re-run — last error: ${pr.error}`
                            : "Re-run prediction for this part"
                        }
                        className="inline-flex items-center gap-1 rounded-md border border-border bg-background/60 px-2 py-1 text-[10px] uppercase tracking-[0.14em] text-muted-foreground transition hover:border-primary/50 hover:text-primary disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:border-border disabled:hover:text-muted-foreground"
                      >
                        <RotateCw
                          className={`h-3 w-3 ${pr.status === "processing" || pr.status === "queued" ? "animate-spin" : ""}`}
                        />
                        re-run
                      </button>
                      {pr.status === "failed" && (
                        <button
                          type="button"
                          onClick={() => physicsFabFeed.dismiss(pr.partId)}
                          title="Dismiss this failed job"
                          className="inline-flex items-center justify-center rounded-md border border-border bg-background/60 px-1.5 py-1 text-muted-foreground transition hover:text-destructive"
                        >
                          <X className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
                {isOpen && (
                  <tr className="border-t border-border bg-muted/20">
                    <td></td>
                    <td colSpan={10} className="px-3 py-3">
                      <div className="mb-1.5 text-[9px] uppercase tracking-[0.22em] text-muted-foreground">
                        Run history · {hist.length} {hist.length === 1 ? "run" : "runs"}
                      </div>
                      {hist.length === 0 ? (
                        <div className="text-[11px] text-muted-foreground">No completed runs yet.</div>
                      ) : (
                        <ol className="space-y-1">
                          {hist.map((h, i) => (
                            <li key={i} className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px]">
                              <span className={`inline-flex h-1.5 w-1.5 shrink-0 rounded-full ${h.ok ? "bg-emerald-500" : "bg-destructive"}`} />
                              <span className="tabular-nums text-muted-foreground">{fmtTime(h.at)}</span>
                              <span className={`uppercase tracking-[0.14em] text-[10px] ${h.ok ? "text-emerald-500" : "text-destructive"}`}>
                                {h.ok ? "succeeded" : "failed"}
                              </span>
                              <span className="tabular-nums text-muted-foreground">{h.ms ?? "—"} ms</span>
                              <span className="tabular-nums text-primary">
                                conf {h.confidence !== undefined ? `${(h.confidence * 100).toFixed(0)}%` : "—"}
                              </span>
                              <span className="text-muted-foreground">attempt {h.attempt + 1}</span>
                              {!h.ok && h.error && (
                                <span className="min-w-0 flex-1 truncate font-mono text-destructive/80" title={h.error}>
                                  {h.error}
                                </span>
                              )}
                            </li>
                          ))}
                        </ol>
                      )}
                    </td>
                  </tr>
                )}
                </>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
