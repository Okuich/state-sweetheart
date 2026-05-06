import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { getPrecisionPolicy, type PrecisionMode } from "@/lib/precisionPolicy";

export function PrecisionPolicyPanel() {
  const policy = getPrecisionPolicy();
  const [, force] = useState(0);
  const refresh = () => force((n) => n + 1);

  // Demo buttons: simulate the rest of the app handing arrays to the GPU.
  const pushSafe = () => {
    policy.toGpuFloat32(new Float32Array([1, 2, 3, 4]), "vertices (f32)");
    refresh();
  };
  const pushLossy = () => {
    const a = new Float64Array([1 + 1e-12, 2 - 1e-12, Math.PI, Math.E]);
    try { policy.toGpuFloat32(a, "Sv (f64 → f32)"); } catch { /* swallow strict */ }
    refresh();
  };
  const pushOverflow = () => {
    try { policy.toGpuFloat32(new Float64Array([1e40, -1e39]), "Fp (overflow)"); }
    catch { /* swallow strict */ }
    refresh();
  };

  const setMode = (m: PrecisionMode) => { policy.setMode(m); refresh(); };
  const reset = () => { policy.reset(); refresh(); };

  // Re-render every 1 s so banner reflects external converters too.
  useEffect(() => {
    const t = setInterval(refresh, 1000);
    return () => clearInterval(t);
  }, []);

  const stats = useMemo(() => policy.snapshot(), [policy, refresh]);
  const banner = policy.shouldShowBanner();

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-xs uppercase tracking-[0.18em] text-muted-foreground">
          GPU precision policy (f64 → f32)
        </h3>
        <Button variant="ghost" size="sm" onClick={reset} className="h-7 px-2 text-xs">
          reset
        </Button>
      </div>

      {banner && (
        <div className="rounded border border-amber-500/40 bg-amber-500/10 p-3 text-[12px] text-amber-300">
          <div className="flex items-start gap-2">
            <span aria-hidden className="text-base leading-none">⚠</span>
            <div className="space-y-1">
              <div className="font-medium">
                Some f64 buffers were downgraded to f32 for the GPU.
              </div>
              <div className="opacity-90">
                {stats.conversions} conversion{stats.conversions === 1 ? "" : "s"}
                · worst rel-loss {stats.worstRelLoss.toExponential(2)}
                {stats.anyOverflow && " · overflow detected"}
                . Switch to <span className="font-mono">strict</span> mode below to
                fail fast and surface the offending call site.
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="space-y-3">
          <div className="space-y-1">
            <div className="text-[11px] text-muted-foreground">downgrade mode</div>
            <Select value={stats.mode} onValueChange={(v) => setMode(v as PrecisionMode)}>
              <SelectTrigger className="h-9 text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                <SelectItem value="auto-downgrade" className="text-xs">
                  auto-downgrade — silently truncate, track loss
                </SelectItem>
                <SelectItem value="strict" className="text-xs">
                  strict — throw if downgrade would lose precision
                </SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="rounded border border-border/60 bg-background/40 p-2 text-[11px] space-y-1">
            <div className="flex justify-between">
              <span className="text-muted-foreground">host f32-only</span>
              <span className="tabular-nums">{stats.hostIsF32Only ? "yes (WebGPU)" : "no (CPU only)"}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">conversions</span>
              <span className="tabular-nums">{stats.conversions}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">elements converted</span>
              <span className="tabular-nums">{stats.elementsConverted.toLocaleString()}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">worst rel-loss</span>
              <span className="tabular-nums">{stats.worstRelLoss.toExponential(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">overflow seen</span>
              <span className="tabular-nums">{stats.anyOverflow ? "yes" : "no"}</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-1">
            <Button size="sm" variant="outline" onClick={pushSafe} className="h-8 px-3 text-xs">
              upload f32
            </Button>
            <Button size="sm" variant="outline" onClick={pushLossy} className="h-8 px-3 text-xs">
              upload f64 (lossy)
            </Button>
            <Button size="sm" variant="outline" onClick={pushOverflow} className="h-8 px-3 text-xs">
              upload f64 (overflow)
            </Button>
          </div>
        </div>

        <div className="rounded border border-border/60 bg-background/40 p-2">
          <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            recent events ({stats.events.length})
          </div>
          {stats.events.length === 0 ? (
            <div className="text-[11px] italic text-muted-foreground">
              No downgrades recorded.
            </div>
          ) : (
            <div className="space-y-1 max-h-56 overflow-y-auto">
              {stats.events.slice().reverse().map((e, i) => (
                <div key={i} className="rounded bg-background/40 px-2 py-1 text-[11px]">
                  <div className="flex justify-between">
                    <span className="font-mono">{e.label}</span>
                    <span className="text-muted-foreground tabular-nums">n={e.length}</span>
                  </div>
                  <div className="flex justify-between text-[10px] text-muted-foreground tabular-nums">
                    <span>|x|max {e.maxAbs.toExponential(2)}</span>
                    <span>relLoss {e.maxRelLoss.toExponential(2)}</span>
                    {e.overflow && <span className="text-destructive">overflow</span>}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
