import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import type { SimParams } from "@/components/PhysicsCanvas";
import {
  type WorldSnapshot,
  saveSnapshot,
  deleteSnapshot,
  clearSnapshots,
  rankBySimilarity,
  loadSnapshots,
} from "@/lib/worldMemory";

/**
 * Persistent Physical World Model panel.
 *
 * Acts as the "long-term memory" of the runtime: snapshots of (params, loss)
 * tuples are persisted to localStorage. The list is re-ranked by cosine
 * similarity against the current live params, so the user sees nearest
 * historical neighbors first — useful for spotting which past run their
 * current configuration is regressing toward, and for one-click restore.
 */
export function WorldMemoryPanel({
  params,
  loss,
  onRestore,
}: {
  params: SimParams;
  loss: number | null;
  onRestore: (p: SimParams) => void;
}) {
  const [version, setVersion] = useState(0); // bump → refresh ranked list
  const [label, setLabel] = useState("");
  const [topOnly, setTopOnly] = useState(true);

  // The snapshot store is small (≤64), so re-ranking on every `params`
  // change is cheap and keeps the "most similar" list live.
  const ranked = useMemo(
    () => rankBySimilarity(params),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [params, version],
  );

  // React to other tabs editing the same store so the list stays in sync.
  useEffect(() => {
    const onStorage = () => setVersion((v) => v + 1);
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const total = loadSnapshots().length;
  const visible = topOnly ? ranked.slice(0, 6) : ranked;

  const handleSave = () => {
    saveSnapshot(label, params, loss);
    setLabel("");
    setVersion((v) => v + 1);
  };

  const handleClear = () => {
    if (typeof window !== "undefined" && window.confirm("Clear all stored world snapshots?")) {
      clearSnapshots();
      setVersion((v) => v + 1);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.22em] text-accent">
            world memory
          </div>
          <div className="text-[10px] text-muted-foreground/80">
            persistent priors · {total} snapshot{total === 1 ? "" : "s"}
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <Button
            variant={topOnly ? "default" : "outline"}
            className={`h-6 px-2 text-[9px] uppercase tracking-[0.14em] ${topOnly ? "bg-accent text-accent-foreground" : ""}`}
            onClick={() => setTopOnly((v) => !v)}
            title="Toggle between top-6 nearest and full history"
          >
            {topOnly ? "top 6" : "all"}
          </Button>
          <Button
            variant="outline"
            className="h-6 px-2 text-[9px] uppercase tracking-[0.14em]"
            onClick={handleClear}
            disabled={total === 0}
          >
            clear
          </Button>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <input
          type="text"
          value={label}
          onChange={(e) => setLabel(e.target.value)}
          placeholder="snapshot label (optional)"
          className="flex-1 rounded border border-border bg-background/40 px-2 py-1.5 text-xs outline-none focus:border-accent"
          maxLength={48}
        />
        <Button
          variant="default"
          className="h-7 px-3 text-[10px] uppercase tracking-[0.16em] bg-primary text-primary-foreground"
          onClick={handleSave}
          title="Persist current params + last loss reading to localStorage"
        >
          snapshot
        </Button>
      </div>

      {visible.length === 0 ? (
        <div className="rounded border border-dashed border-border/60 px-3 py-4 text-center text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          no priors yet — snapshot a run to start
        </div>
      ) : (
        <ul className="space-y-1.5 max-h-72 overflow-y-auto pr-1">
          {visible.map(({ snap, score }) => (
            <SnapshotRow
              key={snap.id}
              snap={snap}
              score={score}
              onRestore={() => onRestore(snap.params)}
              onDelete={() => {
                deleteSnapshot(snap.id);
                setVersion((v) => v + 1);
              }}
            />
          ))}
        </ul>
      )}
    </div>
  );
}

function SnapshotRow({
  snap,
  score,
  onRestore,
  onDelete,
}: {
  snap: WorldSnapshot;
  score: number;
  onRestore: () => void;
  onDelete: () => void;
}) {
  // Color the similarity bar: green ≥ 0.95, amber 0.8–0.95, neutral below
  const color =
    score >= 0.95 ? "oklch(0.82 0.18 150)" : score >= 0.8 ? "oklch(0.84 0.16 85)" : "oklch(0.7 0.05 240)";
  return (
    <li className="rounded border border-border/60 bg-background/30 px-2.5 py-2 hover:border-accent/60 transition-colors">
      <div className="flex items-center justify-between gap-2">
        <button
          onClick={onRestore}
          className="flex-1 text-left text-xs font-medium text-foreground hover:text-accent truncate"
          title="Restore these params"
        >
          {snap.label}
        </button>
        <span className="font-mono text-[10px] tabular-nums" style={{ color }}>
          {(score * 100).toFixed(1)}%
        </span>
        <button
          onClick={onDelete}
          className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground hover:text-destructive"
          title="Delete snapshot"
        >
          ×
        </button>
      </div>
      <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-[9px] font-mono text-muted-foreground">
        <span>N={snap.params.particleCount}</span>
        <span>g={snap.params.gravity}</span>
        <span>k={snap.params.springK}</span>
        <span>{snap.params.integrator}</span>
        <span>{snap.params.boundary}</span>
        {snap.loss !== null && <span>loss={snap.loss.toExponential(1)}</span>}
        <span className="text-muted-foreground/60">
          {new Date(snap.ts).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
      {/* similarity bar */}
      <div className="mt-1.5 h-0.5 w-full bg-border/40 overflow-hidden rounded">
        <div className="h-full" style={{ width: `${Math.max(0, score) * 100}%`, background: color }} />
      </div>
    </li>
  );
}
