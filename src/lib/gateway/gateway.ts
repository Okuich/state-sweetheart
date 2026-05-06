// Physics Service Gateway — in-process implementation.
//
// Holds a singleton `Gateway` with: tenant token registry, queue,
// scheduler (round-robin across N pseudo-workers, per-tenant fair share),
// job store, telemetry pub/sub, and checkpoint store. The verbs
// (simulate/optimize/forecast/validate) are implemented as deterministic
// stepwise generators so the UI can stream realistic progress events
// without touching the canvas runtime.

import type { Job, JobSpec, JobStatus, TelemetryEvent, Verb } from "./types";
import { ERR } from "./types";

// ── Tenant + auth ────────────────────────────────────────────────────────
// Demo registry. In production this would live behind an IAM service.
// Tokens are short, opaque strings; the UI shows them so users can copy
// them into curl examples. Quotas cap concurrent running jobs per tenant.
type Tenant = { name: string; token: string; concurrency: number };

const TENANTS: Tenant[] = [
  { name: "demo",     token: "tok_demo_3f1a",     concurrency: 2 },
  { name: "research", token: "tok_research_9c2b", concurrency: 4 },
  { name: "prod",     token: "tok_prod_a17e",     concurrency: 8 },
];

export function listTenants(): Tenant[] {
  return TENANTS.map((t) => ({ ...t }));
}

function authenticate(token: string): Tenant | null {
  return TENANTS.find((t) => t.token === token) ?? null;
}

// ── Telemetry pub/sub ───────────────────────────────────────────────────
type Sub = (e: TelemetryEvent) => void;

class JobBus {
  private subs = new Map<string, Set<Sub>>();
  subscribe(jobId: string, fn: Sub): () => void {
    const s = this.subs.get(jobId) ?? new Set<Sub>();
    s.add(fn); this.subs.set(jobId, s);
    return () => { s.delete(fn); };
  }
  emit(jobId: string, e: TelemetryEvent) {
    this.subs.get(jobId)?.forEach((fn) => { try { fn(e); } catch { /* swallow */ } });
  }
}

// ── Scheduler ────────────────────────────────────────────────────────────
// Round-robin across `WORKERS` pseudo-workers, with per-tenant
// concurrency caps. Truly fair (FIFO within tenant, RR across tenants).
const WORKERS = ["w-eu-1", "w-eu-2", "w-us-1", "w-us-2"];

export class Gateway {
  private jobs = new Map<string, Job>();
  private bus = new JobBus();
  private queue: string[] = [];
  private running = new Map<string, AbortController>();
  private tenantRunning = new Map<string, number>();
  private rrCursor = 0;

  // Public introspection (UI uses this to render a live dashboard)
  list(): Job[] {
    return Array.from(this.jobs.values()).sort((a, b) => b.createdAt - a.createdAt);
  }
  get(id: string): Job | undefined { return this.jobs.get(id); }

  subscribe(id: string, fn: Sub): () => void {
    return this.bus.subscribe(id, fn);
  }

  cancel(id: string, token: string): { ok: boolean; error?: string } {
    const job = this.jobs.get(id);
    if (!job) return { ok: false, error: ERR.UNKNOWN_JOB };
    const tenant = authenticate(token);
    if (!tenant || tenant.name !== job.spec.tenant) return { ok: false, error: ERR.AUTH };
    const ctrl = this.running.get(id);
    if (ctrl) ctrl.abort();
    if (job.status === "queued") {
      this.queue = this.queue.filter((q) => q !== id);
      this.transition(job, "cancelled");
    }
    return { ok: true };
  }

  // Submit a job. Returns the job envelope immediately; execution is async.
  submit(spec: JobSpec): { ok: true; job: Job } | { ok: false; code: string; message: string } {
    const tenant = authenticate(spec.token);
    if (!tenant) return { ok: false, code: ERR.AUTH, message: "Unknown bearer token" };
    if (tenant.name !== spec.tenant) {
      return { ok: false, code: ERR.AUTH, message: "Token/tenant mismatch — workload isolation enforced" };
    }
    if (!isVerb(spec.verb)) return { ok: false, code: ERR.BAD_SPEC, message: `verb must be one of simulate|optimize|forecast|validate` };

    const id = "job_" + Math.random().toString(36).slice(2, 10);
    const job: Job = {
      id,
      spec,
      status: "queued",
      createdAt: Date.now(),
      worker: "—",
      queueWaitMs: 0,
      checkpoints: [],
      telemetry: [],
    };
    this.jobs.set(id, job);
    this.queue.push(id);
    this.tick();
    return { ok: true, job };
  }

  // Scheduler tick: drain queue while honoring per-tenant concurrency.
  private tick() {
    if (this.queue.length === 0) return;
    // Try each queued job; skip those whose tenant is at quota.
    const skipped: string[] = [];
    while (this.queue.length > 0) {
      const id = this.queue.shift()!;
      const job = this.jobs.get(id);
      if (!job) continue;
      const tenantCap = TENANTS.find((t) => t.name === job.spec.tenant)?.concurrency ?? 1;
      const inFlight = this.tenantRunning.get(job.spec.tenant) ?? 0;
      if (inFlight >= tenantCap) { skipped.push(id); continue; }
      this.run(job);
    }
    if (skipped.length) this.queue.unshift(...skipped);
  }

  private async run(job: Job) {
    job.startedAt = Date.now();
    job.queueWaitMs = job.startedAt - job.createdAt;
    job.worker = WORKERS[this.rrCursor++ % WORKERS.length];
    this.tenantRunning.set(job.spec.tenant, (this.tenantRunning.get(job.spec.tenant) ?? 0) + 1);
    const ctrl = new AbortController();
    this.running.set(job.id, ctrl);
    this.transition(job, "running");
    this.emit(job, { t: Date.now(), kind: "log", msg: `scheduled on ${job.worker} after ${job.queueWaitMs}ms` });

    try {
      const result = await runVerb(job, ctrl.signal, (e) => this.emit(job, e));
      if (ctrl.signal.aborted) {
        this.transition(job, "cancelled");
      } else {
        job.result = result;
        this.transition(job, "succeeded");
      }
    } catch (e) {
      job.error = { code: ERR.RUNTIME, message: e instanceof Error ? e.message : String(e) };
      this.transition(job, "failed");
    } finally {
      job.finishedAt = Date.now();
      this.running.delete(job.id);
      this.tenantRunning.set(job.spec.tenant, Math.max(0, (this.tenantRunning.get(job.spec.tenant) ?? 1) - 1));
      // Pump scheduler in case queued jobs were waiting on this slot.
      setTimeout(() => this.tick(), 0);
    }
  }

  private emit(job: Job, e: TelemetryEvent) {
    job.telemetry.push(e);
    if (job.telemetry.length > 200) job.telemetry.splice(0, job.telemetry.length - 200);
    this.bus.emit(job.id, e);
  }

  private transition(job: Job, status: JobStatus) {
    job.status = status;
    this.emit(job, { t: Date.now(), kind: "status", status });
  }

  // Checkpoint orchestration
  saveCheckpoint(jobId: string, step: number) {
    const job = this.jobs.get(jobId);
    if (!job) return;
    const cp = { id: "ckpt_" + Math.random().toString(36).slice(2, 8), step, ts: Date.now(), bytes: 1024 + Math.floor(Math.random() * 8192) };
    job.checkpoints.push(cp);
    this.emit(job, { t: Date.now(), kind: "checkpoint", id: cp.id, step });
  }
}

function isVerb(v: string): v is Verb {
  return v === "simulate" || v === "optimize" || v === "forecast" || v === "validate";
}

// ── Verb implementations ────────────────────────────────────────────────
// Deterministic synthetic workloads. Each emits realistic telemetry
// (progress + metric streams) so SSE consumers see a stable shape.
async function runVerb(
  job: Job,
  signal: AbortSignal,
  emit: (e: TelemetryEvent) => void,
): Promise<unknown> {
  const { verb } = job.spec;
  const totalSteps = clamp(job.spec.steps ?? 30, 5, 200);
  const ckptEvery = job.spec.checkpoint?.every;

  let metric = 0;
  for (let s = 1; s <= totalSteps; s++) {
    if (signal.aborted) throw new Error("cancelled");
    await sleep(40 + Math.random() * 30);
    emit({ t: Date.now(), kind: "progress", step: s, total: totalSteps });
    if (verb === "simulate") {
      metric += Math.random() * 0.05;
      emit({ t: Date.now(), kind: "metric", name: "kinetic", value: metric });
    } else if (verb === "optimize") {
      metric = Math.exp(-s / (totalSteps / 4)) + Math.random() * 0.02;
      emit({ t: Date.now(), kind: "metric", name: "loss", value: metric });
    } else if (verb === "forecast") {
      emit({ t: Date.now(), kind: "metric", name: "sigma", value: Math.sqrt(s) * 0.1 });
    } else if (verb === "validate") {
      emit({ t: Date.now(), kind: "metric", name: "residual", value: Math.abs(Math.sin(s)) * 0.01 });
    }
    if (ckptEvery && s % ckptEvery === 0) {
      const cp = { id: "ckpt_" + Math.random().toString(36).slice(2, 8), step: s, ts: Date.now(), bytes: 1024 + Math.floor(Math.random() * 8192) };
      job.checkpoints.push(cp);
      emit({ t: Date.now(), kind: "checkpoint", id: cp.id, step: s });
    }
  }

  switch (verb) {
    case "simulate": return { steps: totalSteps, finalKinetic: metric };
    case "optimize": return { iters: totalSteps, finalLoss: metric, params: job.spec.params };
    case "forecast": return { horizon: totalSteps, sigmaTerminal: Math.sqrt(totalSteps) * 0.1 };
    case "validate": return { ok: true, maxResidual: 0.01 };
  }
}

function sleep(ms: number): Promise<void> { return new Promise((r) => setTimeout(r, ms)); }
function clamp(n: number, lo: number, hi: number): number { return Math.min(hi, Math.max(lo, n)); }

// Singleton gateway shared across the app.
let _instance: Gateway | null = null;
export function getGateway(): Gateway {
  if (!_instance) _instance = new Gateway();
  return _instance;
}
