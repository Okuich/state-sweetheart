// Simulation Economics — runtime cost / energy / scheduler dashboard.

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  estimateCost,
  rankDevices,
  efficiencyTips,
  fmtFlops,
  type DeviceClass,
} from "@/lib/simEconomics";
import type { SimParams } from "@/components/PhysicsCanvas";

const DEVICES: DeviceClass[] = ["cpu_laptop", "cpu_server", "gpu_consumer", "gpu_datacenter"];

export function EconomicsPanel({
  params,
  onApplyPatch,
}: {
  params: SimParams;
  onApplyPatch: (patch: Partial<SimParams>) => void;
}) {
  const [device, setDevice] = useState<DeviceClass>("gpu_consumer");
  const [fps, setFps] = useState(60);

  const report = useMemo(() => estimateCost(params, device, fps), [params, device, fps]);
  const schedule = useMemo(() => rankDevices(params, fps), [params, fps]);
  const tips = useMemo(() => efficiencyTips(params), [params]);

  const bd = report.breakdown;
  const segments: { key: string; label: string; v: number; cls: string }[] = [
    { key: "pair", label: "pairwise",   v: bd.flopsPairwise,    cls: "bg-primary" },
    { key: "spr",  label: "springs",    v: bd.flopsSprings,     cls: "bg-secondary" },
    { key: "int",  label: "integrate",  v: bd.flopsIntegrate,   cls: "bg-accent" },
    { key: "fld",  label: "field",      v: bd.flopsField,       cls: "bg-primary/60" },
    { key: "con",  label: "constraints",v: bd.flopsConstraints, cls: "bg-secondary/60" },
    { key: "ens",  label: "ensemble",   v: bd.flopsEnsemble,    cls: "bg-accent/60" },
  ].filter((s) => s.v > 0);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            economics/ — runtime cost · energy · cluster scheduling
          </div>
          <div className="text-sm font-display mt-1">
            {fmtFlops(bd.flopsTotal)}/step · {report.msPerStep.toFixed(2)} ms ·{" "}
            <span className={report.saturation > 1 ? "text-destructive" : "text-primary"}>
              {(report.utilization * 100).toFixed(0)}% util
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value as DeviceClass)}
            className="bg-background border border-border rounded-sm px-2 py-1 text-xs"
          >
            {DEVICES.map((d) => (
              <option key={d} value={d}>
                {d}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-1.5 text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            target fps
            <input
              type="number"
              value={fps}
              min={1}
              max={240}
              onChange={(e) => setFps(Math.max(1, +e.target.value || 60))}
              className="w-14 bg-background border border-border rounded-sm px-1 py-0.5 text-xs text-foreground tabular-nums"
            />
          </label>
        </div>
      </div>

      {/* Stacked FLOP breakdown bar */}
      <div className="mb-4">
        <div className="flex h-3 w-full overflow-hidden rounded-sm border border-border">
          {segments.map((s) => (
            <div
              key={s.key}
              className={s.cls}
              style={{ width: `${(s.v / bd.flopsTotal) * 100}%` }}
              title={`${s.label}: ${fmtFlops(s.v)}`}
            />
          ))}
        </div>
        <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
          {segments.map((s) => (
            <span key={s.key}>
              <span className={`inline-block h-2 w-2 mr-1 rounded-sm ${s.cls}`} />
              {s.label} · {((s.v / bd.flopsTotal) * 100).toFixed(0)}%
            </span>
          ))}
        </div>
      </div>

      {/* Top-level numbers */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 mb-5">
        <Stat label="watts (active)" value={report.watts.toFixed(1) + " W"} />
        <Stat label="energy / hour" value={(report.joulesPerHour / 3600).toFixed(2) + " Wh"} />
        <Stat label="$ / hour" value={"$" + report.costPerHour.toFixed(2)} />
        <Stat
          label="gCO₂ / hour"
          value={report.gCO2PerHour.toFixed(1) + " g"}
          accent={report.gCO2PerHour > 200 ? "text-destructive" : "text-primary"}
        />
      </div>

      {/* Energy-aware scheduler */}
      <div className="mb-5 rounded-md border border-border bg-background/40 p-3">
        <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-2">
          scheduler · greenest feasible device first ({fps} fps)
        </div>
        <ul className="space-y-1.5">
          {schedule.map((s, i) => (
            <li
              key={s.device}
              className={`flex items-center justify-between gap-3 text-xs rounded-sm px-2 py-1.5 ${
                i === 0 ? "bg-primary/10 border border-primary/30" : ""
              }`}
            >
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] uppercase tracking-[0.18em] ${s.feasible ? "text-primary" : "text-destructive"}`}>
                    {s.feasible ? "ok" : "saturated"}
                  </span>
                  <span className="font-medium truncate">{s.label}</span>
                </div>
                <div className="text-muted-foreground text-[10px] mt-0.5">{s.rationale}</div>
              </div>
              <div className="text-right tabular-nums shrink-0">
                <div className="text-primary">{s.effectiveGCO2PerHour.toFixed(0)} gCO₂/h</div>
                <div className="text-muted-foreground text-[10px]">${s.effectiveCostPerHour.toFixed(2)}/h</div>
              </div>
            </li>
          ))}
        </ul>
      </div>

      {/* Efficiency tips */}
      <div className="rounded-md border border-border bg-background/40 p-3">
        <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-2">
          efficiency tips
        </div>
        {tips.length === 0 ? (
          <div className="text-xs text-muted-foreground italic">No obvious savings — workload looks balanced.</div>
        ) : (
          <ul className="space-y-2">
            {tips.map((t) => (
              <li key={t.id} className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-sm font-medium">
                    {t.title}{" "}
                    <span className="text-primary text-xs">~{t.estSavingPct.toFixed(0)}% saved</span>
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">{t.detail}</div>
                </div>
                {t.patch && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="text-[10px] uppercase tracking-[0.18em] shrink-0"
                    onClick={() => onApplyPatch(t.patch!)}
                  >
                    apply
                  </Button>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, accent = "text-foreground" }: { label: string; value: string; accent?: string }) {
  return (
    <div className="rounded-md border border-border bg-background/40 px-3 py-2">
      <div className="text-[9px] uppercase tracking-[0.22em] text-muted-foreground">{label}</div>
      <div className={`text-base font-display tabular-nums mt-0.5 ${accent}`}>{value}</div>
    </div>
  );
}
