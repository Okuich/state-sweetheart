// Physics Service Gateway — UI panel.
//
// Lets you submit jobs against the in-process gateway for any of the four
// verbs, watch live telemetry stream in, inspect per-job checkpoints,
// cancel running jobs, and rotate the active tenant/token. Mirrors what
// an enterprise CLI would expose.

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { getGateway, listTenants } from "@/lib/gateway/gateway";
import type { Job, TelemetryEvent, Verb } from "@/lib/gateway/types";
import type { SimParams } from "@/components/PhysicsCanvas";

const VERBS: Verb[] = ["simulate", "optimize", "forecast", "validate"];
const STATUS_COLOR: Record<Job["status"], string> = {
  queued: "text-muted-foreground",
  running: "text-primary",
  succeeded: "text-primary",
  failed: "text-destructive",
  cancelled: "text-secondary",
};

export function GatewayPanel({ params }: { params: SimParams }) {
  const gw = useMemo(() => getGateway(), []);
  const tenants = useMemo(() => listTenants(), []);
  const [tenant, setTenant] = useState(tenants[0].name);
  const [token, setToken] = useState(tenants[0].token);
  const [verb, setVerb] = useState<Verb>("simulate");
  const [steps, setSteps] = useState(30);
  const [ckptEvery, setCkptEvery] = useState(10);
  const [jobs, setJobs] = useState<Job[]>(() => gw.list());
  const [activeId, setActiveId] = useState<string | null>(null);
  const [stream, setStream] = useState<TelemetryEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Refresh job list every 250ms (in-memory, cheap).
  useEffect(() => {
    const t = setInterval(() => setJobs(gw.list()), 250);
    return () => clearInterval(t);
  }, [gw]);

  // Subscribe to SSE-equivalent stream for the active job.
  useEffect(() => {
    if (!activeId) { setStream([]); return; }
    const job = gw.get(activeId);
    setStream(job ? [...job.telemetry] : []);
    const off = gw.subscribe(activeId, (e) => {
      setStream((s) => [...s.slice(-199), e]);
    });
    return off;
  }, [activeId, gw]);

  const onSubmit = () => {
    setError(null);
    const tenantObj = tenants.find((t) => t.name === tenant);
    const res = gw.submit({
      verb,
      tenant,
      token,
      params,
      steps,
      checkpoint: { every: ckptEvery > 0 ? ckptEvery : undefined },
      budget: { walltimeMs: 30_000 },
    });
    if (!res.ok) {
      setError(`${res.code}: ${res.message}`);
      // Soft hint: token may not match tenant
      if (tenantObj && tenantObj.token !== token) {
        setError((e) => (e ?? "") + ` (expected ${tenantObj.token.slice(0, 8)}…)`);
      }
      return;
    }
    setActiveId(res.job.id);
  };

  const active = activeId ? jobs.find((j) => j.id === activeId) : null;

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div>
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
            gateway/ — POST /v1/{"{verb}"} · bearer auth · per-tenant isolation
          </div>
          <div className="text-sm font-display mt-1">
            {jobs.length} job{jobs.length === 1 ? "" : "s"} · tenant=
            <span className="text-primary">{tenant}</span> · workers across 2 regions
          </div>
        </div>
      </div>

      {/* Submit form */}
      <div className="grid lg:grid-cols-[1fr_1fr_auto] gap-3 mb-3 p-3 rounded-md border border-border bg-background/40">
        <div className="grid grid-cols-2 gap-2">
          <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            tenant
            <select
              value={tenant}
              onChange={(e) => {
                const next = tenants.find((t) => t.name === e.target.value)!;
                setTenant(next.name); setToken(next.token);
              }}
              className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs text-foreground"
            >
              {tenants.map((t) => (
                <option key={t.name} value={t.name}>{t.name} (cap {t.concurrency})</option>
              ))}
            </select>
          </label>
          <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            bearer token
            <input
              value={token}
              onChange={(e) => setToken(e.target.value)}
              className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs font-mono text-foreground"
            />
          </label>
        </div>
        <div className="grid grid-cols-3 gap-2">
          <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            verb
            <select
              value={verb}
              onChange={(e) => setVerb(e.target.value as Verb)}
              className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs text-foreground"
            >
              {VERBS.map((v) => <option key={v}>{v}</option>)}
            </select>
          </label>
          <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            steps
            <input
              type="number" min={1} max={200} value={steps}
              onChange={(e) => setSteps(Math.max(1, +e.target.value || 1))}
              className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs text-foreground tabular-nums"
            />
          </label>
          <label className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
            ckpt every
            <input
              type="number" min={0} max={200} value={ckptEvery}
              onChange={(e) => setCkptEvery(Math.max(0, +e.target.value || 0))}
              className="mt-1 w-full bg-background border border-border rounded-sm px-2 py-1 text-xs text-foreground tabular-nums"
            />
          </label>
        </div>
        <div className="flex items-end">
          <Button
            onClick={onSubmit}
            className="w-full lg:w-auto bg-primary text-primary-foreground hover:bg-primary/90 uppercase tracking-[0.18em] text-xs"
          >
            POST /v1/{verb}
          </Button>
        </div>
      </div>
      {error && (
        <div className="mb-3 text-xs text-destructive border border-destructive/40 rounded-sm px-2 py-1.5 bg-destructive/10">
          {error}
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-3">
        {/* Job list */}
        <div className="rounded-md border border-border bg-background/40 p-3">
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-2">
            jobs (newest first)
          </div>
          {jobs.length === 0 && (
            <div className="text-xs italic text-muted-foreground">No jobs submitted yet.</div>
          )}
          <ul className="space-y-1.5 max-h-72 overflow-auto">
            {jobs.map((j) => (
              <li
                key={j.id}
                onClick={() => setActiveId(j.id)}
                className={`cursor-pointer rounded-sm border px-2 py-1.5 text-xs transition ${
                  activeId === j.id ? "border-primary/50 bg-primary/10" : "border-border hover:bg-background/60"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="font-mono text-[11px] truncate">{j.id}</span>
                  <span className={`uppercase tracking-[0.18em] text-[10px] ${STATUS_COLOR[j.status]}`}>
                    {j.status}
                  </span>
                </div>
                <div className="text-[10px] text-muted-foreground mt-0.5 flex flex-wrap gap-x-2">
                  <span>{j.spec.verb}</span>
                  <span>·</span>
                  <span>{j.spec.tenant}</span>
                  <span>·</span>
                  <span>{j.worker}</span>
                  <span>·</span>
                  <span>ckpts {j.checkpoints.length}</span>
                </div>
              </li>
            ))}
          </ul>
        </div>

        {/* Telemetry / detail */}
        <div className="rounded-md border border-border bg-background/40 p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
              telemetry · GET /v1/jobs/{active ? active.id : "{id}"}/stream
            </div>
            {active && active.status === "running" && (
              <Button
                variant="outline" size="sm"
                className="h-6 text-[10px] uppercase tracking-[0.18em] border-destructive/50 text-destructive"
                onClick={() => gw.cancel(active.id, token)}
              >
                cancel
              </Button>
            )}
          </div>
          {!active ? (
            <div className="text-xs italic text-muted-foreground">Select a job to stream telemetry.</div>
          ) : (
            <TelemetryStream events={stream} job={active} />
          )}
        </div>
      </div>
    </div>
  );
}

function TelemetryStream({ events, job }: { events: TelemetryEvent[]; job: Job }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => { ref.current?.scrollTo({ top: ref.current.scrollHeight }); }, [events]);

  const lastProgress = [...events].reverse().find((e) => e.kind === "progress");
  const pct = lastProgress && lastProgress.kind === "progress"
    ? (lastProgress.step / lastProgress.total) * 100 : 0;

  return (
    <div>
      <div className="mb-2">
        <div className="h-1.5 w-full rounded-full bg-background overflow-hidden">
          <div className="h-full bg-primary transition-all" style={{ width: `${pct}%` }} />
        </div>
        <div className="flex justify-between text-[10px] text-muted-foreground mt-1 tabular-nums">
          <span>{pct.toFixed(0)}%</span>
          <span>queue {job.queueWaitMs}ms · {job.worker}</span>
        </div>
      </div>
      <div ref={ref} className="font-mono text-[10px] leading-relaxed text-muted-foreground max-h-48 overflow-auto bg-background/60 rounded-sm p-2">
        {events.map((e, i) => (
          <div key={i}>
            <span className="text-muted-foreground/50">{new Date(e.t).toISOString().slice(11, 19)}</span>{" "}
            <span className={tagColor(e)}>[{e.kind}]</span>{" "}
            <span>{fmt(e)}</span>
          </div>
        ))}
        {events.length === 0 && <div className="italic">awaiting first event…</div>}
      </div>
      {job.checkpoints.length > 0 && (
        <div className="mt-3">
          <div className="text-[10px] uppercase tracking-[0.3em] text-muted-foreground mb-1">
            checkpoints ({job.checkpoints.length})
          </div>
          <div className="flex flex-wrap gap-1">
            {job.checkpoints.map((c) => (
              <span key={c.id} className="text-[10px] font-mono px-1.5 py-0.5 rounded border border-border bg-background/60">
                @{c.step} · {c.id} · {(c.bytes / 1024).toFixed(1)}KB
              </span>
            ))}
          </div>
        </div>
      )}
      {job.result != null && (
        <pre className="mt-3 text-[10px] font-mono p-2 rounded-sm bg-background/60 border border-border overflow-auto max-h-32">
{JSON.stringify(job.result, null, 2)}
        </pre>
      )}
      {job.error && (
        <div className="mt-3 text-[10px] text-destructive border border-destructive/40 rounded-sm p-2">
          {job.error.code}: {job.error.message}
        </div>
      )}
    </div>
  );
}

function tagColor(e: TelemetryEvent): string {
  switch (e.kind) {
    case "log": return "text-muted-foreground";
    case "metric": return "text-primary";
    case "progress": return "text-accent";
    case "checkpoint": return "text-secondary";
    case "status": return "text-foreground";
  }
}
function fmt(e: TelemetryEvent): string {
  switch (e.kind) {
    case "log": return e.msg;
    case "metric": return `${e.name}=${e.value.toFixed(4)}`;
    case "progress": return `${e.step}/${e.total}`;
    case "checkpoint": return `${e.id} @step ${e.step}`;
    case "status": return e.status;
  }
}
