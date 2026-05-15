/**
 * Fabrication OS — physics-driven predictions feed.
 * Shows the output of the hidden Physics OS pipeline running per
 * imported part. Customers see the predictions, never the engine.
 */
import { useEffect, useState } from "react";
import { physicsFabFeed, type PhysicsPrediction } from "@/lib/physicsFabBridge";

function fmt(n: number | undefined, digits = 2, unit = "") {
  if (n === undefined || n === null || Number.isNaN(n)) return "—";
  return `${n.toFixed(digits)}${unit ? ` ${unit}` : ""}`;
}

export function FabPredictionsPanel() {
  const [items, setItems] = useState<PhysicsPrediction[]>(() => physicsFabFeed.snapshot());

  useEffect(() => physicsFabFeed.subscribe(setItems), []);

  return (
    <section className="rounded-xl border border-border bg-card p-6">
      <header className="mb-4 flex items-center justify-between">
        <div>
          <div className="text-[10px] uppercase tracking-[0.32em] text-muted-foreground">
            Midwater · prediction feed
          </div>
          <h2 className="mt-1 text-lg font-semibold text-foreground">
            Per-part predictions
          </h2>
          <p className="mt-1 max-w-xl text-xs text-muted-foreground">
            Each imported part is automatically scored by the prediction
            pipeline. Results feed calibration residuals against measured
            outcomes.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="h-1.5 w-1.5 rounded-full bg-primary animate-pulse" />
          <span className="text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            {items.length} parts
          </span>
        </div>
      </header>

      {items.length === 0 ? (
        <div className="rounded-md border border-dashed border-border bg-background/40 px-4 py-10 text-center text-xs text-muted-foreground">
          Import a scan or quality report to populate predictions.
        </div>
      ) : (
        <div className="overflow-hidden rounded-md border border-border">
          <table className="w-full text-left text-xs">
            <thead className="bg-muted/40 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              <tr>
                <th className="px-3 py-2">part</th>
                <th className="px-3 py-2">material</th>
                <th className="px-3 py-2">stress (MPa)</th>
                <th className="px-3 py-2">defl (mm)</th>
                <th className="px-3 py-2">safety</th>
                <th className="px-3 py-2">cost</th>
                <th className="px-3 py-2">conf</th>
                <th className="px-3 py-2">ms</th>
              </tr>
            </thead>
            <tbody className="font-mono text-[11px]">
              {items.slice(0, 50).map((p) => (
                <tr key={p.partId} className="border-t border-border">
                  <td className="px-3 py-2 text-foreground">{p.partId}</td>
                  <td className="px-3 py-2 text-muted-foreground">{p.materialName ?? "—"}</td>
                  <td className="px-3 py-2 tabular-nums">
                    {p.ok ? fmt(p.predictedStressMPa) : <span className="text-destructive">err</span>}
                  </td>
                  <td className="px-3 py-2 tabular-nums">{p.ok ? fmt(p.predictedDeflectionMm) : "—"}</td>
                  <td className="px-3 py-2 tabular-nums">{p.ok ? fmt(p.safetyFactor) : "—"}</td>
                  <td className="px-3 py-2 tabular-nums">{p.ok ? fmt(p.costUsd, 2, "$") : "—"}</td>
                  <td className="px-3 py-2 tabular-nums text-primary">
                    {p.ok && p.confidence !== undefined ? `${(p.confidence * 100).toFixed(0)}%` : "—"}
                  </td>
                  <td className="px-3 py-2 tabular-nums text-muted-foreground">{p.ms}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
