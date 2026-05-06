// Physics Service Gateway — unified API surface in front of the runtime.
//
// Verbs: simulate · optimize · forecast · validate.
// Concerns owned here (not by the runtime): job lifecycle, scheduling,
// authentication (bearer-token), per-tenant workload isolation,
// checkpoint orchestration, and SSE telemetry streaming.
//
// This is intentionally a *gateway* — it never runs a real solver. It
// forwards canonical job specs to a deterministic in-memory pipeline so
// downstream consumers (CLIs, notebooks, dashboards) get a stable
// contract while the actual runtime evolves.

import type { SimParams } from "@/components/PhysicsCanvas";

export type Verb = "simulate" | "optimize" | "forecast" | "validate";

export type JobStatus = "queued" | "running" | "succeeded" | "failed" | "cancelled";

export type JobSpec = {
  verb: Verb;
  tenant: string;                       // logical isolation key
  params: Partial<SimParams>;           // sparse override on top of defaults
  steps?: number;                       // for simulate / forecast
  budget?: { walltimeMs?: number; iters?: number };
  checkpoint?: { every?: number; restoreFrom?: string };
  // Token used to authenticate the request. The gateway hashes & compares
  // against a tenant→token map; never logged in plain text.
  token: string;
};

export type Job = {
  id: string;
  spec: JobSpec;
  status: JobStatus;
  createdAt: number;
  startedAt?: number;
  finishedAt?: number;
  // Distributed scheduling fields
  worker: string;                       // pseudo-worker id assigned by scheduler
  queueWaitMs: number;
  // Result + error envelope
  result?: unknown;
  error?: { code: string; message: string };
  // Checkpoints captured during run, indexed by step.
  checkpoints: { id: string; step: number; ts: number; bytes: number }[];
  // Compact telemetry buffer for late subscribers (SSE replay).
  telemetry: TelemetryEvent[];
};

export type TelemetryEvent =
  | { t: number; kind: "log"; msg: string }
  | { t: number; kind: "metric"; name: string; value: number }
  | { t: number; kind: "progress"; step: number; total: number }
  | { t: number; kind: "checkpoint"; id: string; step: number }
  | { t: number; kind: "status"; status: JobStatus };

// Stable error codes returned by the gateway.
export const ERR = {
  AUTH: "auth/invalid-token",
  TENANT_QUOTA: "tenant/quota-exceeded",
  BAD_SPEC: "spec/invalid",
  UNKNOWN_JOB: "job/not-found",
  UNKNOWN_CHECKPOINT: "checkpoint/not-found",
  RUNTIME: "runtime/failure",
} as const;
