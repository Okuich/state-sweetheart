/**
 * Hidden prediction bridge powering the Midwater dashboard.
 *
 * Subscribes to scan-import observations, then for each unique part
 * triggers a background prediction. Results and per-part processing
 * status are surfaced to Midwater UI panels. Customers see only the
 * Midwater pipeline — the underlying engine names are never exposed.
 */
import type { Observation } from "./fabFeedback";
import { scanImportBridge } from "./scanImportStore";
import { predictForMidwater } from "./physics.functions";

export type PartStatus = "queued" | "processing" | "ready" | "failed";

export interface PartProgress {
  partId: string;
  status: PartStatus;
  startedAt: string;
  updatedAt: string;
  ms?: number;
  error?: string;
  attempt: number;
  nextRetryAt?: string;
}

const MAX_AUTO_RETRIES = 2;
const RETRY_BASE_MS = 1500;

export interface PhysicsPrediction {
  partId: string;
  source: string;
  confidence?: number;
  predictedStressMPa?: number;
  predictedDeflectionMm?: number;
  safetyFactor?: number;
  costUsd?: number;
  recommendations?: { type: string; title: string; detail: string }[];
  materialName?: string;
  computedAt: string;
  ms: number;
  ok: boolean;
  error?: string;
}

export interface PartHistoryEntry {
  at: string;
  ok: boolean;
  ms?: number;
  confidence?: number;
  error?: string;
  attempt: number;
}

export interface FeedSnapshot {
  predictions: PhysicsPrediction[];
  progress: PartProgress[];
  history: Record<string, PartHistoryEntry[]>;
  counts: { queued: number; processing: number; ready: number; failed: number };
}

type Listener = (snap: FeedSnapshot) => void;

const HISTORY_LIMIT = 20;
const cache = new Map<string, PhysicsPrediction>();
const progress = new Map<string, PartProgress>();
const history = new Map<string, PartHistoryEntry[]>();
const listeners = new Set<Listener>();
let started = false;

function buildSnapshot(): FeedSnapshot {
  const predictions = Array.from(cache.values()).sort((a, b) =>
    b.computedAt.localeCompare(a.computedAt),
  );
  const prog = Array.from(progress.values()).sort((a, b) =>
    b.updatedAt.localeCompare(a.updatedAt),
  );
  const counts = { queued: 0, processing: 0, ready: 0, failed: 0 };
  for (const p of prog) counts[p.status] += 1;
  return { predictions, progress: prog, counts };
}

function emit() {
  const snap = buildSnapshot();
  for (const l of listeners) l(snap);
}

function setStatus(partId: string, status: PartStatus, extra: Partial<PartProgress> = {}) {
  const now = new Date().toISOString();
  const prev = progress.get(partId);
  progress.set(partId, {
    partId,
    status,
    startedAt: prev?.startedAt ?? now,
    updatedAt: now,
    ms: extra.ms ?? prev?.ms,
    error: status === "failed" ? extra.error ?? prev?.error : undefined,
    attempt: extra.attempt ?? prev?.attempt ?? 0,
    nextRetryAt: extra.nextRetryAt,
  });
}

const DEFAULT_REF = {
  geometry: { volume: 0.035, surfaceArea: 2.4 },
  material: "steel",
  crossSectionalArea: 5.89e-3,
  momentOfInertia: 4.54e-5,
  beamLength: 6,
  loadProfile: { force: 1500, direction: { x: 0, y: -1, z: 0 } },
  geometryType: "beam",
};

async function runForPart(partId: string, attempt = 0) {
  const cur = progress.get(partId);
  if (cur && (cur.status === "queued" || cur.status === "processing")) return;

  setStatus(partId, "queued", { attempt });
  emit();
  // Yield so UI shows "queued" before flipping to "processing".
  await Promise.resolve();
  setStatus(partId, "processing", { attempt });
  emit();

  const t0 = Date.now();
  try {
    const out = await predictForMidwater({
      data: { ...DEFAULT_REF, geometryId: partId },
    });
    const ms = Date.now() - t0;
    cache.set(partId, {
      partId,
      source: out?.source ?? "analytical",
      confidence: out?.confidence,
      predictedStressMPa: out?.stress?.vonMises,
      predictedDeflectionMm: out?.deflection?.deflectionMm,
      safetyFactor: out?.safety?.safetyFactor,
      costUsd: out?.cost?.totalCost,
      recommendations: out?.recommendations,
      materialName: out?.materialName,
      computedAt: out?.timestamp ?? new Date().toISOString(),
      ms,
      ok: true,
    });
    setStatus(partId, "ready", { ms, attempt });
    emit();
  } catch (e) {
    const ms = Date.now() - t0;
    const error = e instanceof Error ? e.message : String(e);

    if (attempt < MAX_AUTO_RETRIES) {
      // Exponential backoff retry. Surface the in-flight retry to the UI
      // by keeping status "queued" with nextRetryAt set.
      const delay = RETRY_BASE_MS * Math.pow(2, attempt);
      const nextRetryAt = new Date(Date.now() + delay).toISOString();
      setStatus(partId, "queued", {
        ms,
        error: `${error} — retrying (${attempt + 1}/${MAX_AUTO_RETRIES})`,
        attempt: attempt + 1,
        nextRetryAt,
      });
      emit();
      const timer = setTimeout(() => {
        retryTimers.delete(partId);
        void runForPart(partId, attempt + 1);
      }, delay);
      retryTimers.set(partId, timer);
      return;
    }

    cache.set(partId, {
      partId,
      source: "error",
      computedAt: new Date().toISOString(),
      ms,
      ok: false,
      error,
    });
    setStatus(partId, "failed", { ms, error, attempt });
    emit();
  }
}

function handleBatch(batch: Observation[]) {
  const unique = new Set<string>();
  for (const o of batch) if (o.partId) unique.add(o.partId);
  for (const partId of unique) void runForPart(partId);
}

const retryTimers = new Map<string, ReturnType<typeof setTimeout>>();

function cancelRetry(partId: string) {
  const t = retryTimers.get(partId);
  if (t) {
    clearTimeout(t);
    retryTimers.delete(partId);
  }
}

/** Idempotent — call once on dashboard mount. */
export function startPhysicsFabBridge(): () => void {
  if (started) return () => {};
  started = true;
  const unsubscribe = scanImportBridge.subscribe(handleBatch);
  return () => {
    unsubscribe();
    started = false;
  };
}

export const physicsFabFeed = {
  subscribe(l: Listener): () => void {
    listeners.add(l);
    l(buildSnapshot());
    return () => listeners.delete(l);
  },
  snapshot(): FeedSnapshot {
    return buildSnapshot();
  },
  /** Manual fire-and-forget for parts without a scan import. */
  predict(partId: string) {
    cancelRetry(partId);
    void runForPart(partId);
  },
  /** Dismiss a failed part — clears its row and cached error. */
  dismiss(partId: string) {
    cancelRetry(partId);
    cache.delete(partId);
    progress.delete(partId);
    emit();
  },
  /** Dismiss every currently-failed part. */
  dismissAllFailed() {
    for (const [id, p] of progress) {
      if (p.status === "failed") {
        cancelRetry(id);
        cache.delete(id);
        progress.delete(id);
      }
    }
    emit();
  },
  clear() {
    for (const t of retryTimers.values()) clearTimeout(t);
    retryTimers.clear();
    cache.clear();
    progress.clear();
    emit();
  },
};
