import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import {
  Popover, PopoverContent, PopoverTrigger,
} from "@/components/ui/popover";
import { getPrecisionPolicy, type PrecisionMode } from "@/lib/precisionPolicy";
import {
  listKernelPaths,
  decideDtypeToggle,
  type KernelF64Status,
} from "@/lib/kernelCapabilities";

type Dtype = "f32" | "f64";

function WhyThisMatters() {
  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button variant="ghost" size="sm" className="h-7 px-2 text-xs">
          why this matters
        </Button>
      </PopoverTrigger>
      <PopoverContent
        side="bottom"
        align="end"
        className="w-[360px] space-y-3 p-4 text-[12px] leading-relaxed"
      >
        <div className="space-y-1">
          <div className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            f32 vs f64 in this simulation
          </div>
          <p className="text-muted-foreground">
            f32 has ~7 decimal digits of precision (ULP ≈ 1.2e-7 at value 1).
            f64 has ~16 (ULP ≈ 2.2e-16). Every dot-product, cross-product, and
            time integration step accumulates rounding at this scale.
          </p>
        </div>

        <div className="space-y-1">
          <div className="font-medium text-foreground">Stability</div>
          <p className="text-muted-foreground">
            Implicit solvers (CG, Newton) lose orthogonality faster in f32.
            Stiff materials and small Δt amplify cancellation in
            <span className="font-mono"> F = I + ∇u</span>; conditioning above
            ~1e6 typically needs f64 to converge.
          </p>
        </div>

        <div className="space-y-1">
          <div className="font-medium text-foreground">Energy drift</div>
          <p className="text-muted-foreground">
            Symplectic integrators conserve energy up to round-off. In f32,
            drift is ~1e-7 per step and visibly accumulates over thousands of
            substeps; in f64 it stays at the noise floor for the whole run.
          </p>
        </div>

        <div className="space-y-1">
          <div className="font-medium text-foreground">Determinism</div>
          <p className="text-muted-foreground">
            f32 reductions on GPU are non-associative and depend on workgroup
            scheduling — same inputs, different sums. f64 doesn't fix
            non-associativity, but the per-op error is small enough that
            replays match bitwise across runs in practice.
          </p>
        </div>

        <div className="rounded border border-border/60 bg-background/40 p-2 text-[11px] text-muted-foreground">
          WebGPU is f32-only today. When the active kernel can't run f64, the
          dtype toggle is locked and any f64 inputs are downgraded according
          to the policy below.
        </div>
      </PopoverContent>
    </Popover>
  );
}

export function PrecisionPolicyPanel() {
  const policy = getPrecisionPolicy();
  const [, force] = useState(0);
  const refresh = () => force((n) => n + 1);

  // Active kernel paths the simulator currently uses. In a real app this
  // would come from the simulator config; we let the user toggle them so
  // the gating behaviour is observable.
  const allPaths = useMemo(() => listKernelPaths(), []);
  const [activeIds, setActiveIds] = useState<string[]>([
    "cpu-reference",
    "webgpu-spatial-hash",
  ]);
  const decision = useMemo(() => decideDtypeToggle(activeIds), [activeIds]);

  const [dtype, setDtype] = useState<Dtype>("f32");
  // If support is removed (kernel path activated), force back to f32.
  useEffect(() => {
    if (!decision.f64Enabled && dtype === "f64") setDtype("f32");
  }, [decision.f64Enabled, dtype]);

  const toggleActive = (id: string) => {
    setActiveIds((cur) =>
      cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id]
    );
  };

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
        <div className="flex items-center gap-1">
          <WhyThisMatters />
          <Button variant="ghost" size="sm" onClick={reset} className="h-7 px-2 text-xs">
            reset
          </Button>
        </div>
      </div>

      {!decision.f64Enabled && (
        <div className="rounded border border-amber-500/40 bg-amber-500/10 p-3 text-[12px] text-amber-300">
          <div className="flex items-start gap-2">
            <span aria-hidden className="text-base leading-none">⚠</span>
            <div className="space-y-1">
              <div className="font-medium">
                f64 dtype unavailable on the current kernel path.
              </div>
              <div className="opacity-90">{decision.reason}</div>
            </div>
          </div>
        </div>
      )}

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
            <div className="text-[11px] text-muted-foreground">simulation dtype</div>
            <Select
              value={dtype}
              onValueChange={(v) => setDtype(v as Dtype)}
              disabled={!decision.f64Enabled && dtype === "f32"
                ? false
                : !decision.f64Enabled}
            >
              <SelectTrigger
                className="h-9 text-xs"
                disabled={!decision.f64Enabled}
                title={decision.f64Enabled ? undefined : decision.reason}
              >
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="f32" className="text-xs">f32 (always available)</SelectItem>
                <SelectItem value="f64" className="text-xs" disabled={!decision.f64Enabled}>
                  f64 {decision.f64Enabled ? "" : "— unsupported by active path"}
                </SelectItem>
              </SelectContent>
            </Select>
            {!decision.f64Enabled && (
              <div className="text-[10px] text-muted-foreground">
                Toggle disabled: {decision.reason}
              </div>
            )}
          </div>

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

        <div className="space-y-3">
          <div className="rounded border border-border/60 bg-background/40 p-2">
            <div className="mb-2 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
              active kernel paths
            </div>
            <div className="space-y-1">
              {allPaths.map((s: KernelF64Status) => {
                const on = activeIds.includes(s.path.id);
                return (
                  <button
                    key={s.path.id}
                    onClick={() => toggleActive(s.path.id)}
                    className={`w-full rounded px-2 py-1 text-left text-[11px] transition ${
                      on ? "bg-primary/15 border border-primary/40" : "border border-border/40 hover:bg-background/60"
                    }`}
                    title={s.reason}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono">{s.path.label}</span>
                      <span className={`tabular-nums text-[10px] ${s.supportsF64 ? "text-emerald-400" : "text-amber-400"}`}>
                        {s.supportsF64 ? "f64 ok" : "f32 only"}
                      </span>
                    </div>
                    {!s.supportsF64 && s.reason && (
                      <div className="mt-0.5 text-[10px] text-muted-foreground">{s.reason}</div>
                    )}
                  </button>
                );
              })}
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
    </div>
  );
}
