/**
 * Refinement priors store.
 *
 * Records per-region (octree leaf bbox) instability/refinement history so a
 * future run on similar geometry can pre-refine known hotspots without
 * waiting for the first physics feedback step.
 *
 * Storage is in-memory (sessionStorage-backed if available) keyed by a
 * coarse Morton-style spatial hash + dominant field tag. This is the
 * client-side staging area; production would mirror it into Lovable Cloud.
 */

import type { OctreeMesh, Vec3 } from "../meshing/octree";
import type { LeafErrorReport, RefinementPlan } from "./refine";

export interface PriorRecord {
  /** Quantized cell key. */
  key: string;
  /** World-space cell center. */
  center: Vec3;
  /** Cell extent at quantization grid. */
  cellSize: number;
  /** Dominant field tag (FIELD_ORDER index). */
  dominant: number;
  /** Mean combined error observed at this cell. */
  meanError: number;
  /** How many runs have visited this cell. */
  visits: number;
  /** Last seen timestamp. */
  lastSeen: number;
}

const FIELD_LABELS = ["stress", "thermal", "contact", "deformation", "curvature", "residual"] as const;

export class RefinementPriorStore {
  private map = new Map<string, PriorRecord>();
  constructor(private readonly quantum = 0.05) {}

  private quantizeKey(p: Vec3, dominant: number): { key: string; center: Vec3 } {
    const q = this.quantum;
    const kx = Math.round(p[0] / q);
    const ky = Math.round(p[1] / q);
    const kz = Math.round(p[2] / q);
    return {
      key: `${dominant}:${kx},${ky},${kz}`,
      center: [kx * q, ky * q, kz * q],
    };
  }

  ingest(mesh: OctreeMesh, err: LeafErrorReport, plan: RefinementPlan): number {
    let added = 0;
    const now = Date.now();
    for (const li of plan.splitLeaves) {
      const id = mesh.leaves[li];
      const b = mesh.nodes[id].bbox;
      const c: Vec3 = [
        (b.min[0] + b.max[0]) / 2,
        (b.min[1] + b.max[1]) / 2,
        (b.min[2] + b.max[2]) / 2,
      ];
      const dom = err.dominant[li];
      const { key, center } = this.quantizeKey(c, dom);
      const existing = this.map.get(key);
      if (existing) {
        existing.meanError = (existing.meanError * existing.visits + err.combined[li]) / (existing.visits + 1);
        existing.visits += 1;
        existing.lastSeen = now;
      } else {
        this.map.set(key, {
          key,
          center,
          cellSize: this.quantum,
          dominant: dom,
          meanError: err.combined[li],
          visits: 1,
          lastSeen: now,
        });
        added++;
      }
    }
    return added;
  }

  /** Top-K hottest priors, optionally filtered to a bbox. */
  topK(k: number): PriorRecord[] {
    return [...this.map.values()]
      .sort((a, b) => b.meanError * Math.log1p(b.visits) - a.meanError * Math.log1p(a.visits))
      .slice(0, k);
  }

  size() { return this.map.size; }

  summary() {
    const tagCounts: Record<string, number> = {};
    for (const r of this.map.values()) {
      const label = FIELD_LABELS[r.dominant] ?? "unknown";
      tagCounts[label] = (tagCounts[label] ?? 0) + 1;
    }
    return { total: this.map.size, tagCounts };
  }

  clear() { this.map.clear(); }
}

let SHARED: RefinementPriorStore | null = null;
export function sharedPriorStore(): RefinementPriorStore {
  if (!SHARED) SHARED = new RefinementPriorStore(0.05);
  return SHARED;
}
