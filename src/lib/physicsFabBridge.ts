/**
 * Physics OS → Fabrication OS bridge (client-side).
 *
 * Subscribes to scan-import observations, then for each unique part
 * triggers the Physics OS `runPipeline` server function in the
 * background. Results are stored in a small in-memory cache and
 * surfaced to the Fabrication OS dashboard as predictions feeding
 * calibration. Customers never see Physics OS directly — they only
 * see the resulting fab predictions.
 */
import type { Observation } from "./fabFeedback";
import { scanImportBridge } from "./scanImportStore";
import { runPipeline } from "./physics.functions";

export interface PhysicsPrediction {
  partId: string;
  source: string; // engine source (analytical / hybrid / etc.)
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

type Listener = (preds: PhysicsPrediction[]) => void;

const cache = new Map<string, PhysicsPrediction>();
const inflight = new Set<string>();
const listeners = new Set<Listener>();
let started = false;

function emit() {
  const snapshot = Array.from(cache.values()).sort((a, b) =>
    b.computedAt.localeCompare(a.computedAt),
  );
  for (const l of listeners) l(snapshot);
}

// Default reference geometry — used when the scan does not carry
// part dimensions. Mirrors the analytical engine's medium-beam preset.
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
  if (inflight.has(partId) || cache.has(partId)) return;
  inflight.add(partId);
  const t0 = Date.now();
  try {
    const out = await runPipeline({
      data: { ...DEFAULT_REF, geometryId: partId },
    });
    cache.set(partId, {
      partId,
      source: out?.source ?? "analytical",
      confidence: out?.confidence,
      predictedStressMPa: out?.stress?.vonMisesMPa,
      predictedDeflectionMm: out?.deflection?.maxMm,
      safetyFactor: out?.safety?.factor,
      costUsd: out?.cost?.totalUsd,
      recommendations: out?.recommendations,
      materialName: out?.materialName,
      computedAt: out?.timestamp ?? new Date().toISOString(),
      ms: Date.now() - t0,
      ok: true,
    });
  } catch (e) {
    cache.set(partId, {
      partId,
      source: "error",
      computedAt: new Date().toISOString(),
      ms: Date.now() - t0,
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    });
  } finally {
    inflight.delete(partId);
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
    l(Array.from(cache.values()));
    return () => listeners.delete(l);
  },
  snapshot(): PhysicsPrediction[] {
    return Array.from(cache.values());
  },
  /** Manual fire-and-forget for parts without a scan import. */
  predict(partId: string) {
    void runForPart(partId);
  },
  clear() {
    cache.clear();
    emit();
  },
};
