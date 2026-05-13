/**
 * Partition-aware broadphase collision pruning.
 *
 * Each rank builds a uniform spatial hash over its locally-owned tet AABBs
 * plus the AABBs of halo (ghost) tets received from neighbouring ranks.
 * Pair candidates are emitted within each rank-local hash, and tagged as:
 *
 *   - local       both tets owned by this rank
 *   - ghost       one local + one ghost  (cross-partition pair)
 *   - duplicate   ghost↔ghost (suppressed; another rank owns the pair)
 *
 * Without halo exchange the cross-partition pairs would be missed; the
 * `coverage` metric reports how many of those would-be-missed pairs are
 * recovered by the halo plan. This is the partition-aware analog of a
 * single-GPU sweep-and-prune broadphase.
 */

import type { OctreeMesh } from "../meshing/octree";
import type { HaloPlan } from "./comm";

export interface BroadphaseOptions {
  /** Cells per axis on the spatial hash. Defaults to ⌈cbrt(T/8)⌉. */
  gridResolution?: number;
  /** Optional pre-padding (fraction of bbox extent) applied to AABBs. */
  padding?: number;
}

export interface BroadphaseResult {
  partitionCount: number;
  gridResolution: number;
  /** Per-rank pair counts. */
  perRank: {
    rank: number;
    local: number;
    ghost: number;
    occupied: number;
  }[];
  totalLocalPairs: number;
  totalGhostPairs: number;
  /** Unique cross-partition pairs (no double-count across owners). */
  uniqueCrossPairs: number;
  /** O(N²) naive pair count for the same AABB set. */
  naiveCandidatePairs: number;
  /** 1 - (local+unique cross) / naive — pruning rate vs naive. */
  pruningRate: number;
  /** Fraction of cross-partition candidates surfaced by halo (0..1). */
  haloCoverage: number;
  buildMs: number;
}

interface TetAABB {
  min: [number, number, number];
  max: [number, number, number];
}

function tetAABBs(mesh: OctreeMesh, padding: number): TetAABB[] {
  const T = mesh.tets.length / 4;
  const out: TetAABB[] = new Array(T);
  const ext = Math.max(
    mesh.bbox.max[0] - mesh.bbox.min[0],
    mesh.bbox.max[1] - mesh.bbox.min[1],
    mesh.bbox.max[2] - mesh.bbox.min[2],
  );
  const pad = ext * padding;
  for (let t = 0; t < T; t++) {
    let xmin = Infinity, ymin = Infinity, zmin = Infinity;
    let xmax = -Infinity, ymax = -Infinity, zmax = -Infinity;
    for (let k = 0; k < 4; k++) {
      const v = mesh.tets[t * 4 + k] * 3;
      const x = mesh.vertices[v];
      const y = mesh.vertices[v + 1];
      const z = mesh.vertices[v + 2];
      if (x < xmin) xmin = x; if (x > xmax) xmax = x;
      if (y < ymin) ymin = y; if (y > ymax) ymax = y;
      if (z < zmin) zmin = z; if (z > zmax) zmax = z;
    }
    out[t] = {
      min: [xmin - pad, ymin - pad, zmin - pad],
      max: [xmax + pad, ymax + pad, zmax + pad],
    };
  }
  return out;
}

function aabbOverlap(a: TetAABB, b: TetAABB): boolean {
  return (
    a.min[0] <= b.max[0] && a.max[0] >= b.min[0] &&
    a.min[1] <= b.max[1] && a.max[1] >= b.min[1] &&
    a.min[2] <= b.max[2] && a.max[2] >= b.min[2]
  );
}

export function partitionBroadphase(
  mesh: OctreeMesh,
  tetPart: Uint16Array,
  halo: HaloPlan,
  opts: BroadphaseOptions = {},
): BroadphaseResult {
  const t0 = Date.now();
  const T = mesh.tets.length / 4;
  const P = halo.partitionCount;
  const aabbs = tetAABBs(mesh, opts.padding ?? 0.0);
  const gridRes = Math.max(2, opts.gridResolution ?? Math.ceil(Math.cbrt(T / 8)));
  

  const bx = mesh.bbox.min[0], by = mesh.bbox.min[1], bz = mesh.bbox.min[2];
  const sx = (mesh.bbox.max[0] - bx) || 1;
  const sy = (mesh.bbox.max[1] - by) || 1;
  const sz = (mesh.bbox.max[2] - bz) || 1;

  const cellOf = (x: number, axis: 0 | 1 | 2): number => {
    const base = axis === 0 ? bx : axis === 1 ? by : bz;
    const span = axis === 0 ? sx : axis === 1 ? sy : sz;
    const c = Math.floor(((x - base) / span) * gridRes);
    return Math.max(0, Math.min(gridRes - 1, c));
  };

  // Hash (rank, cellIdx) -> array of [tetId, isGhost]
  type Cell = { tet: number; ghost: boolean }[];
  const rankHash: Map<number, Cell>[] = Array.from({ length: P }, () => new Map());
  const rankOccupied = new Uint32Array(P);

  const insert = (rank: number, tet: number, ghost: boolean) => {
    const a = aabbs[tet];
    const cx0 = cellOf(a.min[0], 0), cx1 = cellOf(a.max[0], 0);
    const cy0 = cellOf(a.min[1], 1), cy1 = cellOf(a.max[1], 1);
    const cz0 = cellOf(a.min[2], 2), cz1 = cellOf(a.max[2], 2);
    const map = rankHash[rank];
    for (let z = cz0; z <= cz1; z++)
      for (let y = cy0; y <= cy1; y++)
        for (let x = cx0; x <= cx1; x++) {
          const idx = (z * gridRes + y) * gridRes + x;
          let bucket = map.get(idx);
          if (!bucket) { bucket = []; map.set(idx, bucket); rankOccupied[rank]++; }
          bucket.push({ tet, ghost });
        }
  };

  for (let t = 0; t < T; t++) insert(tetPart[t], t, false);
  for (let p = 0; p < P; p++) for (const t of halo.haloRecv[p]) insert(p, t, true);

  // Candidate pair extraction per rank.
  const perRank: BroadphaseResult["perRank"] = [];
  let totalLocal = 0;
  let totalGhost = 0;
  const crossSeen = new Set<number>();
  for (let rank = 0; rank < P; rank++) {
    let local = 0;
    let ghost = 0;
    const seenLocal = new Set<number>();
    const seenGhost = new Set<number>();
    for (const bucket of rankHash[rank].values()) {
      const n = bucket.length;
      if (n < 2) continue;
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const a = bucket[i], b = bucket[j];
          if (a.ghost && b.ghost) continue; // skip ghost↔ghost
          if (a.tet === b.tet) continue;
          if (!aabbOverlap(aabbs[a.tet], aabbs[b.tet])) continue;
          const lo = Math.min(a.tet, b.tet);
          const hi = Math.max(a.tet, b.tet);
          const key = lo * (T + 1) + hi;
          if (!a.ghost && !b.ghost) {
            if (seenLocal.has(key)) continue;
            seenLocal.add(key);
            local++;
          } else {
            if (seenGhost.has(key)) continue;
            seenGhost.add(key);
            ghost++;
            crossSeen.add(key);
          }
        }
      }
    }
    perRank.push({ rank, local, ghost, occupied: rankOccupied[rank] });
    totalLocal += local;
    totalGhost += ghost;
  }

  // Ground-truth cross-partition pairs (full-mesh broadphase, but we just
  // need an upper bound — use AABB grid on the full mesh once).
  const globalMap = new Map<number, number[]>();
  for (let t = 0; t < T; t++) {
    const a = aabbs[t];
    const cx0 = cellOf(a.min[0], 0), cx1 = cellOf(a.max[0], 0);
    const cy0 = cellOf(a.min[1], 1), cy1 = cellOf(a.max[1], 1);
    const cz0 = cellOf(a.min[2], 2), cz1 = cellOf(a.max[2], 2);
    for (let z = cz0; z <= cz1; z++)
      for (let y = cy0; y <= cy1; y++)
        for (let x = cx0; x <= cx1; x++) {
          const idx = (z * gridRes + y) * gridRes + x;
          let arr = globalMap.get(idx);
          if (!arr) { arr = []; globalMap.set(idx, arr); }
          arr.push(t);
        }
  }
  const allCross = new Set<number>();
  let allPairs = 0;
  const seenAll = new Set<number>();
  for (const arr of globalMap.values()) {
    for (let i = 0; i < arr.length; i++)
      for (let j = i + 1; j < arr.length; j++) {
        const a = arr[i], b = arr[j];
        if (!aabbOverlap(aabbs[a], aabbs[b])) continue;
        const lo = Math.min(a, b), hi = Math.max(a, b);
        const key = lo * (T + 1) + hi;
        if (seenAll.has(key)) continue;
        seenAll.add(key);
        allPairs++;
        if (tetPart[a] !== tetPart[b]) allCross.add(key);
      }
  }

  const naivePairs = (T * (T - 1)) / 2;
  const surfaced = totalLocal + crossSeen.size;
  const pruningRate = naivePairs ? 1 - surfaced / naivePairs : 0;
  const haloCoverage = allCross.size ? crossSeen.size / allCross.size : 1;

  return {
    partitionCount: P,
    gridResolution: gridRes,
    perRank,
    totalLocalPairs: totalLocal,
    totalGhostPairs: totalGhost,
    uniqueCrossPairs: crossSeen.size,
    naiveCandidatePairs: allPairs,
    pruningRate,
    haloCoverage,
    buildMs: Date.now() - t0,
  };
  // Note: cellsTotal kept implicit via gridResolution; we don't expose it.
  void cellsTotal;
}
