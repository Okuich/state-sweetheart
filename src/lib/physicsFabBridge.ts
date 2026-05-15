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
}

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

export interface FeedSnapshot {
  predictions: PhysicsPrediction[];
  progress: PartProgress[];
  counts: { queued: number; processing: number; ready: number; failed: number };
}

type Listener = (snap: FeedSnapshot) => void;

const cache = new Map<string, PhysicsPrediction>();
const progress = new Map<string, PartProgress>();
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
    error: extra.error ?? prev?.error,
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

async function runForPart(partId: string) {
  const cur = progress.get(partId);
  if (cur && (cur.status === "queued" || cur.status === "processing")) return;

  setStatus(partId, "queued");
  emit();
  // Yield so UI shows "queued" before flipping to "processing".
  await Promise.resolve();
  setStatus(partId, "processing");
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
    setStatus(partId, "ready", { ms });
  } catch (e) {
    const ms = Date.now() - t0;
    const error = e instanceof Error ? e.message : String(e);
    cache.set(partId, {
      partId,
      source: "error",
      computedAt: new Date().toISOString(),
      ms,
      ok: false,
      error,
    });
    setStatus(partId, "failed", { ms, error });
  } finally {
    emit();
  }
}

function handleBatch(batch: Observation[]) {
  const unique = new Set<string>();
  for (const o of batch) if (o.partId) unique.add(o.partId);
  for (const partId of unique) void runForPart(partId);
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
    void runForPart(partId);
  },
  clear() {
    cache.clear();
    progress.clear();
    emit();
  },
};
