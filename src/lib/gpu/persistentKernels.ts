/**
 * Persistent traversal kernel simulator.
 *
 * Models a GPU "persistent thread" launch where W warps (32 lanes each) sit in
 * a single dispatch and atomically pull batches of queries from a shared work
 * queue until exhausted. This avoids the per-query kernel-launch overhead
 * naive traversal pays.
 *
 * The simulator runs the actual TS traversal of BVH/SpatialHash/KD-tree but
 * instruments it to record per-lane step counts, memory access patterns, and
 * branch outcomes so we can derive warp-level metrics:
 *
 *   - occupancy  = Σ active_lane_steps / (warp_steps × 32)
 *   - divergence = 1 − (mean lane steps / max lane steps in warp)
 *   - coalesce   = fraction of warp loads where ≥ 16 lanes hit the same
 *                  64-byte cache line (modeled via node-id locality)
 *   - persistent reuse = queries / warpDispatches
 *
 * Inputs are reordered along the Morton curve before being chunked into warps
 * to mirror a "branch-minimized" execution path: lanes in a warp share a
 * neighborhood, so they take similar tree paths and access adjacent memory.
 *
 * Pure simulator — no WebGPU dependency, deterministic, no allocations during
 * inner loops besides per-query result vectors.
 */

import type { BVH, Vec3 } from "../spatial/bvh";
import type { SpatialHash } from "../spatial/spatialHash";
import type { KDTree } from "../spatial/kdtree";
import { mortonCodes, mortonOrder } from "../spatial/morton";

export type KernelKind = "bvh-ray" | "bvh-aabb" | "hash-point" | "kd-knn";

export interface KernelOptions {
  /** Lanes per warp (CUDA = 32, AMD = 64). */
  warpSize?: number;
  /** Number of resident warps (persistent thread count = warps × warpSize). */
  warpsResident?: number;
  /** Bytes per node load — used to compute coalescence. */
  nodeStrideBytes?: number;
  /** Cache line size in bytes (default 128 = 4 nodes/line at 32B nodes). */
  cacheLineBytes?: number;
  /** Per-warp launch overhead in microseconds (naive baseline only). */
  naiveLaunchUs?: number;
  /** Per-step compute cost in nanoseconds (for wall-clock estimate). */
  perStepNs?: number;
}

export interface KernelMetrics {
  kind: KernelKind;
  /** Total queries processed. */
  queries: number;
  /** Number of warps actually dispatched in this kernel. */
  warpsDispatched: number;
  /** Total per-lane traversal steps (sum across warp lifetime). */
  totalSteps: number;
  /** Σ(max-lane-steps) per warp — the wall clock warp-time. */
  warpClockSteps: number;
  /** Lanes active across all warp-clock steps / (warps × warpSize × clockSteps). */
  occupancy: number;
  /** 1 − mean(steps)/max(steps) averaged over warps. */
  divergence: number;
  /** Fraction of node loads where ≥ ½ the warp hits the same cache line. */
  coalesceRatio: number;
  /** queries / warpsDispatched — persistent kernel reuse factor. */
  persistentReuse: number;
  /** Bytes loaded (totalSteps × nodeStrideBytes). */
  bytesLoaded: number;
  /** Estimated kernel duration in microseconds (persistent layout). */
  kernelUs: number;
  /** Estimated naive per-query launch duration. */
  naiveUs: number;
  /** naiveUs / kernelUs — speedup of persistent over naive. */
  speedup: number;
}

const DEFAULTS: Required<KernelOptions> = {
  warpSize: 32,
  warpsResident: 32,
  nodeStrideBytes: 32,
  cacheLineBytes: 128,
  naiveLaunchUs: 5,
  perStepNs: 4,
};

/** Reorder queries along Morton curve for warp coalescence. */
export function coalesceOrder(queries: Float32Array, bbox: { min: Vec3; max: Vec3 }): Uint32Array {
  return mortonOrder(mortonCodes(queries, bbox));
}

// ─────────────────────────────────────────────────────────────────────────────
// Instrumented traversal: returns per-query (steps, lastNodeId).
// We track lastNodeId so that warp-level coalescence can be computed from
// neighboring lanes' final node IDs (approximation of access locality).
// ─────────────────────────────────────────────────────────────────────────────

interface QueryTrace { steps: number; touched: number[] }

function traceBvhRay(bvh: BVH, origin: Vec3, dir: Vec3): QueryTrace {
  const touched: number[] = [];
  let steps = 0;
  const stack: number[] = [0];
  const invDir: Vec3 = [1 / (dir[0] || 1e-12), 1 / (dir[1] || 1e-12), 1 / (dir[2] || 1e-12)];
  while (stack.length) {
    const n = stack.pop()!;
    steps++;
    touched.push(n);
    const i3 = n * 3;
    let tNear = -Infinity, tFar = Infinity, miss = false;
    for (let k = 0; k < 3; k++) {
      const o = origin[k], inv = invDir[k];
      let t1 = (bvh.nodeMin[i3 + k] - o) * inv, t2 = (bvh.nodeMax[i3 + k] - o) * inv;
      if (t1 > t2) { const t = t1; t1 = t2; t2 = t; }
      if (t1 > tNear) tNear = t1;
      if (t2 < tFar) tFar = t2;
      if (tNear > tFar || tFar < 0) { miss = true; break; }
    }
    if (miss) continue;
    if (bvh.left[n] === -1) continue;
    stack.push(bvh.left[n], bvh.right[n]);
  }
  return { steps, touched };
}

function traceBvhAabb(bvh: BVH, qLo: Vec3, qHi: Vec3): QueryTrace {
  const touched: number[] = [];
  let steps = 0;
  const stack: number[] = [0];
  while (stack.length) {
    const n = stack.pop()!;
    steps++;
    touched.push(n);
    const i3 = n * 3;
    if (
      bvh.nodeMax[i3]     < qLo[0] || bvh.nodeMin[i3]     > qHi[0] ||
      bvh.nodeMax[i3 + 1] < qLo[1] || bvh.nodeMin[i3 + 1] > qHi[1] ||
      bvh.nodeMax[i3 + 2] < qLo[2] || bvh.nodeMin[i3 + 2] > qHi[2]
    ) continue;
    if (bvh.left[n] === -1) continue;
    stack.push(bvh.left[n], bvh.right[n]);
  }
  return { steps, touched };
}

function traceHash(hash: SpatialHash, q: Vec3): QueryTrace {
  const touched: number[] = [];
  const inv = 1 / hash.cellSize;
  const ix = Math.floor((q[0] - hash.origin[0]) * inv);
  const iy = Math.floor((q[1] - hash.origin[1]) * inv);
  const iz = Math.floor((q[2] - hash.origin[2]) * inv);
  let steps = 0;
  // Visit the 27 surrounding cells (typical broadphase neighbor sweep).
  for (let dz = -1; dz <= 1; dz++) {
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        const h = (((ix + dx) * 73856093) ^ ((iy + dy) * 19349663) ^ ((iz + dz) * 83492791)) >>> 0;
        const cell = h % hash.hashSize;
        const a = hash.bucketStart[cell], b = hash.bucketStart[cell + 1];
        steps += (b - a) + 1;
        touched.push(cell);
      }
    }
  }
  return { steps, touched };
}

function traceKnn(tree: KDTree, q: Vec3, k: number): QueryTrace {
  const touched: number[] = [];
  let steps = 0;
  const heap: number[] = []; // d2 max-heap as sorted array (descending)
  const stack: number[] = [0];
  while (stack.length) {
    const node = stack.pop()!;
    if (node < 0 || tree.axis[node] < 0) continue;
    steps++;
    touched.push(node);
    const ax = tree.axis[node];
    const aIdx = tree.order[tree.anchor[node]];
    const dx = tree.points[aIdx * 3]     - q[0];
    const dy = tree.points[aIdx * 3 + 1] - q[1];
    const dz = tree.points[aIdx * 3 + 2] - q[2];
    const d2 = dx * dx + dy * dy + dz * dz;
    if (heap.length < k) {
      heap.push(d2);
      heap.sort((a, b) => b - a);
    } else if (d2 < heap[0]) {
      heap[0] = d2;
      heap.sort((a, b) => b - a);
    }
    const split = tree.points[aIdx * 3 + ax];
    const diff = q[ax] - split;
    const near = diff < 0 ? tree.left[node] : tree.right[node];
    const far  = diff < 0 ? tree.right[node] : tree.left[node];
    if (near >= 0) stack.push(near);
    if (far >= 0 && (heap.length < k || diff * diff < heap[0])) stack.push(far);
  }
  return { steps, touched };
}

// ─────────────────────────────────────────────────────────────────────────────
// Persistent dispatcher: groups query traces into warps and aggregates.
// ─────────────────────────────────────────────────────────────────────────────

function aggregateWarps(
  traces: QueryTrace[],
  permutation: Uint32Array,
  opts: Required<KernelOptions>,
  kind: KernelKind,
): KernelMetrics {
  const warpSize = opts.warpSize;
  const Q = traces.length;
  // nodes per cache line (avoid div-by-zero).
  const nodesPerLine = Math.max(1, Math.floor(opts.cacheLineBytes / opts.nodeStrideBytes));

  let totalSteps = 0;
  let warpClockSteps = 0;
  let activeLaneSteps = 0;
  let coalescedLoads = 0;
  let totalLoads = 0;
  let warpsDispatched = 0;
  let divergenceSum = 0;

  for (let w = 0; w < Q; w += warpSize) {
    const laneCount = Math.min(warpSize, Q - w);
    let maxSteps = 0;
    let sumSteps = 0;
    for (let l = 0; l < laneCount; l++) {
      const s = traces[permutation[w + l]].steps;
      sumSteps += s;
      if (s > maxSteps) maxSteps = s;
    }
    if (!maxSteps) continue;
    warpsDispatched++;
    totalSteps += sumSteps;
    warpClockSteps += maxSteps;
    activeLaneSteps += sumSteps; // each step = one active lane-step
    divergenceSum += 1 - sumSteps / (maxSteps * laneCount);

    // Coalescence: walk through warp clock-step k. Lanes that still have
    // step k pending visit their k-th touched node. Count cache line shared.
    for (let k = 0; k < maxSteps; k++) {
      const lines = new Map<number, number>(); // line id → lane count
      let activeLanes = 0;
      for (let l = 0; l < laneCount; l++) {
        const tr = traces[permutation[w + l]];
        if (k >= tr.steps) continue;
        activeLanes++;
        const nodeId = tr.touched[k] | 0;
        const line = (nodeId / nodesPerLine) | 0;
        lines.set(line, (lines.get(line) ?? 0) + 1);
      }
      if (!activeLanes) continue;
      totalLoads += lines.size; // each unique cache line = one transaction
      // count loads where the dominant line was hit by ≥½ the active lanes.
      let dominant = 0;
      for (const c of lines.values()) if (c > dominant) dominant = c;
      if (dominant >= Math.ceil(activeLanes / 2)) coalescedLoads++;
    }
  }

  const denom = warpsDispatched * warpSize * (warpClockSteps / Math.max(1, warpsDispatched));
  const occupancy = denom > 0 ? activeLaneSteps / denom : 0;
  const divergence = warpsDispatched ? divergenceSum / warpsDispatched : 0;
  const coalesceRatio = totalLoads ? coalescedLoads / totalLoads : 0;

  // Persistent kernel: scheduled across `warpsResident` warps simultaneously.
  // Wall clock ≈ ceil(warpsDispatched / warpsResident) × per-warp clock steps.
  const perWarpClockSteps = warpsDispatched ? warpClockSteps / warpsDispatched : 0;
  const waves = warpsDispatched ? Math.ceil(warpsDispatched / opts.warpsResident) : 0;
  const kernelUs =
    waves > 0
      ? waves * perWarpClockSteps * (opts.perStepNs / 1000) + opts.naiveLaunchUs
      : 0;
  // Naive: one launch per warp (or per query — pick warp-grain to be charitable).
  const naiveUs =
    warpsDispatched * (opts.naiveLaunchUs + perWarpClockSteps * (opts.perStepNs / 1000));

  const bytesLoaded = totalLoads * opts.cacheLineBytes;

  return {
    kind,
    queries: Q,
    warpsDispatched,
    totalSteps,
    warpClockSteps,
    occupancy,
    divergence,
    coalesceRatio,
    persistentReuse: warpsDispatched ? Q / warpsDispatched : 0,
    bytesLoaded,
    kernelUs,
    naiveUs,
    speedup: kernelUs > 0 ? naiveUs / kernelUs : 1,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry points — one per kernel kind.
// ─────────────────────────────────────────────────────────────────────────────

export interface RayQuerySet { origins: Float32Array; dirs: Float32Array }
export interface AabbQuerySet { los: Float32Array; his: Float32Array }
export interface PointQuerySet { points: Float32Array }

export function persistentBvhRay(
  bvh: BVH,
  set: RayQuerySet,
  bbox: { min: Vec3; max: Vec3 },
  opts: KernelOptions = {},
): KernelMetrics {
  const o = { ...DEFAULTS, ...opts };
  const Q = set.origins.length / 3;
  const traces: QueryTrace[] = new Array(Q);
  for (let i = 0; i < Q; i++) {
    traces[i] = traceBvhRay(
      bvh,
      [set.origins[i * 3], set.origins[i * 3 + 1], set.origins[i * 3 + 2]],
      [set.dirs[i * 3], set.dirs[i * 3 + 1], set.dirs[i * 3 + 2]],
    );
  }
  const perm = coalesceOrder(set.origins, bbox);
  return aggregateWarps(traces, perm, o, "bvh-ray");
}

export function persistentBvhAabb(
  bvh: BVH,
  set: AabbQuerySet,
  bbox: { min: Vec3; max: Vec3 },
  opts: KernelOptions = {},
): KernelMetrics {
  const o = { ...DEFAULTS, ...opts };
  const Q = set.los.length / 3;
  const traces: QueryTrace[] = new Array(Q);
  const centers = new Float32Array(Q * 3);
  for (let i = 0; i < Q; i++) {
    centers[i * 3]     = (set.los[i * 3]     + set.his[i * 3])     * 0.5;
    centers[i * 3 + 1] = (set.los[i * 3 + 1] + set.his[i * 3 + 1]) * 0.5;
    centers[i * 3 + 2] = (set.los[i * 3 + 2] + set.his[i * 3 + 2]) * 0.5;
    traces[i] = traceBvhAabb(
      bvh,
      [set.los[i * 3], set.los[i * 3 + 1], set.los[i * 3 + 2]],
      [set.his[i * 3], set.his[i * 3 + 1], set.his[i * 3 + 2]],
    );
  }
  const perm = coalesceOrder(centers, bbox);
  return aggregateWarps(traces, perm, o, "bvh-aabb");
}

export function persistentHashPoint(
  hash: SpatialHash,
  set: PointQuerySet,
  bbox: { min: Vec3; max: Vec3 },
  opts: KernelOptions = {},
): KernelMetrics {
  const o = { ...DEFAULTS, ...opts };
  const Q = set.points.length / 3;
  const traces: QueryTrace[] = new Array(Q);
  for (let i = 0; i < Q; i++) {
    traces[i] = traceHash(hash, [set.points[i * 3], set.points[i * 3 + 1], set.points[i * 3 + 2]]);
  }
  const perm = coalesceOrder(set.points, bbox);
  return aggregateWarps(traces, perm, o, "hash-point");
}

export function persistentKdKnn(
  tree: KDTree,
  set: PointQuerySet,
  k: number,
  bbox: { min: Vec3; max: Vec3 },
  opts: KernelOptions = {},
): KernelMetrics {
  const o = { ...DEFAULTS, ...opts };
  const Q = set.points.length / 3;
  const traces: QueryTrace[] = new Array(Q);
  for (let i = 0; i < Q; i++) {
    traces[i] = traceKnn(tree, [set.points[i * 3], set.points[i * 3 + 1], set.points[i * 3 + 2]], k);
  }
  const perm = coalesceOrder(set.points, bbox);
  return aggregateWarps(traces, perm, o, "kd-knn");
}

export interface PersistentKernelSuite {
  bvhRay: KernelMetrics;
  bvhAabb: KernelMetrics;
  hashPoint: KernelMetrics;
  kdKnn: KernelMetrics;
  totalKernelUs: number;
  totalNaiveUs: number;
  totalSpeedup: number;
}

export interface SuiteInputs {
  bvh: BVH;
  hash: SpatialHash;
  kd: KDTree;
  bbox: { min: Vec3; max: Vec3 };
  queryCount?: number;
  k?: number;
  opts?: KernelOptions;
}

/** Generate a deterministic mixed query set and run all four kernels. */
export function runPersistentSuite(inputs: SuiteInputs): PersistentKernelSuite {
  const Q = inputs.queryCount ?? 4096;
  const k = inputs.k ?? 8;
  const opts = inputs.opts ?? {};
  const ext: Vec3 = [
    inputs.bbox.max[0] - inputs.bbox.min[0],
    inputs.bbox.max[1] - inputs.bbox.min[1],
    inputs.bbox.max[2] - inputs.bbox.min[2],
  ];

  // Deterministic LCG so traces are reproducible.
  let s = 0x9e3779b9;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 0xffffffff; };

  const origins = new Float32Array(Q * 3);
  const dirs = new Float32Array(Q * 3);
  const los = new Float32Array(Q * 3);
  const his = new Float32Array(Q * 3);
  const points = new Float32Array(Q * 3);

  for (let i = 0; i < Q; i++) {
    const px = inputs.bbox.min[0] + rand() * ext[0];
    const py = inputs.bbox.min[1] + rand() * ext[1];
    const pz = inputs.bbox.min[2] + rand() * ext[2];
    points[i * 3]     = px; points[i * 3 + 1] = py; points[i * 3 + 2] = pz;
    origins[i * 3]    = inputs.bbox.min[0] - 0.1 * ext[0];
    origins[i * 3 + 1] = py;
    origins[i * 3 + 2] = pz;
    dirs[i * 3] = 1; dirs[i * 3 + 1] = 0; dirs[i * 3 + 2] = 0;
    const r = 0.02 * Math.min(ext[0], ext[1], ext[2]);
    los[i * 3]     = px - r; los[i * 3 + 1] = py - r; los[i * 3 + 2] = pz - r;
    his[i * 3]     = px + r; his[i * 3 + 1] = py + r; his[i * 3 + 2] = pz + r;
  }

  const bvhRay = persistentBvhRay(inputs.bvh, { origins, dirs }, inputs.bbox, opts);
  const bvhAabb = persistentBvhAabb(inputs.bvh, { los, his }, inputs.bbox, opts);
  const hashPoint = persistentHashPoint(inputs.hash, { points }, inputs.bbox, opts);
  const kdKnn = persistentKdKnn(inputs.kd, { points }, k, inputs.bbox, opts);

  const totalKernelUs = bvhRay.kernelUs + bvhAabb.kernelUs + hashPoint.kernelUs + kdKnn.kernelUs;
  const totalNaiveUs = bvhRay.naiveUs + bvhAabb.naiveUs + hashPoint.naiveUs + kdKnn.naiveUs;
  return {
    bvhRay,
    bvhAabb,
    hashPoint,
    kdKnn,
    totalKernelUs,
    totalNaiveUs,
    totalSpeedup: totalKernelUs > 0 ? totalNaiveUs / totalKernelUs : 1,
  };
}
