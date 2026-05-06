// ── persistent_world_model.py ────────────────────────────────────────
// A long-term store of (SimParams, summary, feature-vector) snapshots.
// Persisted in localStorage so runs survive reloads. Retrieval uses
// cosine similarity over a normalized numeric feature vector — cheap
// O(K·D) scan against the in-memory cache (K = #snapshots, D ≈ 18).
//
// What this gives us:
//   • simulation histories         — snapshots ordered by time
//   • learned physical priors      — per-snapshot loss / drift / σ̄ baked in
//   • reusable optimization knowledge — restore a past SimParams 1-click
//   • material behavior memory     — springK / restLength / cMax travel together
//   • similarity search            — "show runs that look like THIS one"

import type { SimParams } from "@/components/PhysicsCanvas";

export type WorldSnapshot = {
  id: string;
  label: string;
  ts: number;          // epoch ms
  params: SimParams;
  // Live diagnostics captured at snapshot time (best-effort; may be null
  // if the canvas hasn't reported yet).
  loss: number | null;
  // Pre-computed feature vector + its L2 norm for fast cosine sim.
  feat: number[];
  fnorm: number;
};

const STORAGE_KEY = "physicsstate.worldMemory.v1";
const MAX_SNAPSHOTS = 64; // cap so localStorage stays well under 5 MB

// Numeric feature extractor. We normalize each axis to roughly [0, 1]
// so no single dimension dominates the cosine score.
const FEATURE_KEYS: { k: keyof SimParams; scale: number }[] = [
  { k: "gravity",          scale: 1 / 200 },
  { k: "damping",          scale: 1 / 2 },
  { k: "attractor",        scale: 1 / 5 },
  { k: "particleCount",    scale: 1 / 2000 },
  { k: "springK",          scale: 1 / 400 },
  { k: "restLength",       scale: 1 / 200 },
  { k: "edgesPerNode",     scale: 1 / 8 },
  { k: "pairwiseStrength", scale: 1 / 1000 },
  { k: "pairwiseRadius",   scale: 1 / 200 },
  { k: "constraintIters",  scale: 1 / 16 },
  { k: "fieldStrength",    scale: 1 / 3 },
  { k: "subSteps",         scale: 1 / 16 },
  { k: "workers",          scale: 1 / 16 },
  { k: "objectiveLR",      scale: 1 / 0.5 },
  { k: "noiseSigma",       scale: 1 / 60 },
  { k: "ensembleK",        scale: 1 / 64 },
];

export function extractFeatures(p: SimParams): number[] {
  const out: number[] = FEATURE_KEYS.map(({ k, scale }) => {
    const v = p[k];
    return typeof v === "number" ? v * scale : 0;
  });
  // Categorical one-hots — kept tiny so they don't drown out the numerics.
  out.push(p.integrator === "verlet" ? 1 : p.integrator === "semi-euler" ? 0.5 : 0);
  out.push(p.boundary === "periodic" ? 1 : p.boundary === "wrap" ? 0.5 : 0);
  return out;
}

function l2(v: number[]): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += v[i] * v[i];
  return Math.sqrt(s);
}

export function cosine(a: number[], an: number, b: number[], bn: number): number {
  if (an === 0 || bn === 0) return 0;
  const L = Math.min(a.length, b.length);
  let dot = 0;
  for (let i = 0; i < L; i++) dot += a[i] * b[i];
  return dot / (an * bn);
}

export function loadSnapshots(): WorldSnapshot[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const arr = JSON.parse(raw) as WorldSnapshot[];
    return Array.isArray(arr) ? arr : [];
  } catch {
    return [];
  }
}

function persist(snaps: WorldSnapshot[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snaps));
  } catch {
    // quota exceeded — drop the oldest half and retry once
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(snaps.slice(-Math.floor(snaps.length / 2))));
    } catch { /* give up silently */ }
  }
}

export function saveSnapshot(label: string, params: SimParams, loss: number | null): WorldSnapshot {
  const feat = extractFeatures(params);
  const snap: WorldSnapshot = {
    id: `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`,
    label: label.trim() || `run @ ${new Date().toLocaleTimeString()}`,
    ts: Date.now(),
    params: { ...params },
    loss,
    feat,
    fnorm: l2(feat),
  };
  const snaps = loadSnapshots();
  snaps.push(snap);
  while (snaps.length > MAX_SNAPSHOTS) snaps.shift();
  persist(snaps);
  return snap;
}

export function deleteSnapshot(id: string) {
  persist(loadSnapshots().filter((s) => s.id !== id));
}

export function clearSnapshots() {
  persist([]);
}

/** Rank all stored snapshots by similarity to the given live params. */
export function rankBySimilarity(params: SimParams): { snap: WorldSnapshot; score: number }[] {
  const f = extractFeatures(params);
  const fn = l2(f);
  return loadSnapshots()
    .map((snap) => ({ snap, score: cosine(f, fn, snap.feat, snap.fnorm) }))
    .sort((a, b) => b.score - a.score);
}
