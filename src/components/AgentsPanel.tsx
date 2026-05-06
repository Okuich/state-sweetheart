// Autonomous Physics Agents — UI panel.
//
// Renders the live agent action queue as approvable cards. Each card shows
// agent kind, severity, rationale and either an "Apply patch" or "Run tool"
// button. An auto-pilot toggle dispatches `critical` actions automatically.

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  runAgents,
  summarizeWorld,
  PRESET_PATCHES,
  type AgentAction,
  type AgentGoal,
  type AgentKind,
  type ParamPatch,
  type ToolCall,
} from "@/lib/physicsAgents";
import type { SimParams, ValidationReport } from "@/components/PhysicsCanvas";

const GOALS: { value: AgentGoal; label: string }[] = [
  { value: "stability", label: "Stability" },
  { value: "performance", label: "Performance" },
  { value: "exploration", label: "Exploration" },
  { value: "energy_conservation", label: "Energy" },
  { value: "minimize_loss", label: "Min loss" },
];

const KIND_COLOR: Record<AgentKind, string> = {
  constructor: "text-primary",
  optimizer: "text-accent",
  diagnostician: "text-secondary",
  stabilizer: "text-destructive",
  designer: "text-foreground/80",
};

export function AgentsPanel({
  params,
  validation,
  loss,
  onApplyPatch,
  onReset,
  onSnapshot,
}: {
  params: SimParams;
  validation: ValidationReport | null;
  loss: number | null;
  onApplyPatch: (patch: ParamPatch) => void;
  onReset: () => void;
  onSnapshot: (label: string) => void;
}) {
  const [goal, setGoal] = useState<AgentGoal>("stability");
  const [autopilot, setAutopilot] = useState(false);
  const [enabled, setEnabled] = useState<Record<AgentKind, boolean>>({
    constructor: true, optimizer: true, diagnostician: true, stabilizer: true, designer: true,
  });
  const [log, setLog] = useState<{ ts: number; text: string }[]>([]);
  const dispatchedRef = useRef<Set<string>>(new Set());

  const actions: AgentAction[] = useMemo(
    () => runAgents({ params, validation, loss, goal }, enabled),
    [params, validation, loss, goal, enabled],
  );

  const dispatchTool = (t: ToolCall) => {
    switch (t.name) {
      case "reset_sim": onReset(); return `tool · reset_sim()`;
      case "apply_patch": onApplyPatch(t.patch); return `tool · apply_patch(${Object.keys(t.patch).join(",")})`;
      case "set_preset": onApplyPatch(PRESET_PATCHES[t.preset]); return `tool · set_preset(${t.preset})`;
      case "snapshot_world": onSnapshot(t.label); return `tool · snapshot_world("${t.label}")`;
    }
  };

  const apply = (a: AgentAction) => {
    let line = `${a.agent} · ${a.title}`;
    if (a.patch) { onApplyPatch(a.patch); line += ` → patch(${Object.keys(a.patch).join(",")})`; }
    if (a.tool) line += ` → ${dispatchTool(a.tool)}`;
    setLog((l) => [{ ts: Date.now(), text: line }, ...l].slice(0, 30));
  };

  // Autopilot: auto-dispatch any *critical* action exactly once per identity.
  useEffect(() => {
    if (!autopilot) return;
    for (const a of actions) {
      if (a.severity !== "critical") continue;
      const key = a.title; // stable across re-renders unlike a.id
      if (dispatchedRef.current.has(key)) continue;
      dispatchedRef.current.add(key);
      apply(a);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autopilot, actions]);

  const summary = summarizeWorld(params);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            agents/ — autonomous physics reasoning layer
          </div>
          <div className="text-sm font-display mt-1">
            {actions.length} proposal{actions.length === 1 ? "" : "s"} ·{" "}
            <span className="text-muted-foreground">{summary}</span>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-md border border-border bg-background/60 p-1">
            {GOALS.map((g) => (
              <button
                key={g.value}
                onClick={() => setGoal(g.value)}
                className={`px-2.5 py-1 text-[10px] uppercase tracking-[0.18em] rounded-sm transition ${
                  goal === g.value ? "bg-primary/20 text-primary" : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {g.label}
              </button>
            ))}
          </div>
          <Button
            variant={autopilot ? "default" : "outline"}
            onClick={() => setAutopilot((a) => !a)}
            className="text-[10px] uppercase tracking-[0.18em] h-8"
          >
            {autopilot ? "autopilot · on" : "autopilot · off"}
          </Button>
        </div>
      </div>

      {/* Agent enable/disable chips */}
      <div className="flex flex-wrap gap-1.5 mb-4">
        {(Object.keys(enabled) as AgentKind[]).map((k) => (
          <button
            key={k}
            onClick={() => setEnabled((e) => ({ ...e, [k]: !e[k] }))}
            className={`px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] rounded-full border transition ${
              enabled[k]
                ? `border-current ${KIND_COLOR[k]} bg-background/40`
                : "border-border text-muted-foreground/50"
            }`}
          >
            {k}
          </button>
        ))}
      </div>

      {/* Action queue */}
      <div className="grid gap-2 lg:grid-cols-2">
        {actions.length === 0 && (
          <div className="text-xs text-muted-foreground italic col-span-full py-4">
            All agents quiet — current world satisfies every active rule for goal={goal}.
          </div>
        )}
        {actions.map((a) => (
          <div
            key={a.id}
            className={`rounded-md border p-3 bg-background/40 ${
              a.severity === "critical"
                ? "border-destructive/60"
                : a.severity === "warn"
                ? "border-secondary/40"
                : "border-border"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-[10px] uppercase tracking-[0.18em]">
                  <span className={KIND_COLOR[a.agent]}>{a.agent}</span>
                  <span
                    className={
                      a.severity === "critical"
                        ? "text-destructive"
                        : a.severity === "warn"
                        ? "text-secondary"
                        : "text-muted-foreground"
                    }
                  >
                    · {a.severity}
                  </span>
                  <span className="text-muted-foreground/70">· conf {(a.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="text-sm mt-1 font-medium">{a.title}</div>
                <div className="text-xs text-muted-foreground mt-1 leading-relaxed">{a.rationale}</div>
              </div>
              <Button
                size="sm"
                variant={a.severity === "critical" ? "destructive" : "outline"}
                onClick={() => apply(a)}
                className="text-[10px] uppercase tracking-[0.18em] shrink-0"
              >
                {a.tool ? `run ${a.tool.name}` : "apply"}
              </Button>
            </div>
          </div>
        ))}
      </div>

      {/* Activity log */}
      {log.length > 0 && (
        <div className="mt-4 rounded-md border border-border bg-background/30 p-3">
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-2">activity</div>
          <ul className="space-y-1 text-[11px] font-mono text-muted-foreground max-h-40 overflow-auto">
            {log.map((e, i) => (
              <li key={i}>
                <span className="text-muted-foreground/50">{new Date(e.ts).toISOString().slice(11, 19)}</span>{" "}
                {e.text}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
