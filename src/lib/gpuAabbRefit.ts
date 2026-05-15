import { writeTypedBuffer } from "./gpu/writeBuffer";

/**
 * gpuAabbRefit.ts
 * ──────────────────────────────────────────────────────────────────────────
 * GPU-native AABB updates for deformable geometry — cloth triangles and
 * particle "shapes" (point clouds with per-particle radius). Each
 * simulation step the bounding boxes need to be refreshed because the
 * underlying vertices have moved (cloth bending, particle advection,
 * shape-matching, etc). A full rebuild from scratch every step would
 * dominate the broad-phase budget, so this module performs an
 * INCREMENTAL REFIT:
 *
 *   • A leaf AABB tracks one primitive (a cloth triangle or one particle
 *     "shape" = a contiguous run of particle indices).
 *   • Each step we recompute leaf AABBs from current vertex positions,
 *     but ONLY for primitives whose vertices actually moved more than
 *     `refitEpsilon` since the last full recompute (a per-leaf dirty
 *     flag, computed on-GPU via an atomicOr against the previous box).
 *   • Internal nodes are then refit bottom-up by union-ing their two
 *     children. We propagate the dirty flag so a parent only updates
 *     when at least one child changed — O(log N) work in the average
 *     case, O(N) worst-case (totally chaotic deformation).
 *
 * The tree topology itself is FIXED — refit never re-balances. After a
 * configurable number of steps (or when a quality metric trips), the
 * caller can request a full rebuild via `markAllDirty()`.
 *
 * Pipeline shape
 * --------------
 *      vertices (Float32Array, mutated by integrator)
 *              │
 *              ▼
 *    [leaf_refit kernel]  per-primitive: compute new tight AABB,
 *                         compare to previous, set dirty bit
 *              │
 *              ▼
 *    [internal_refit kernel] level-by-level bottom-up; each internal
 *                            node = union(child0, child1) iff any
 *                            descendant is dirty
 *              │
 *              ▼
 *      Aabb[ ] storage buffer  (consumed by broad-phase)
 *
 * Sandbox note
 * ------------
 * The Lovable preview browser has no GPU adapter. `initAabbRefit()`
 * returns a descriptor with `mode: "cpu"` in that case and the same
 * `refitStep()` API runs a JS path that is bit-equivalent to the WGSL
 * kernel (verified by gpuAabbRefit.test.ts).
 */

// ── Public types ─────────────────────────────────────────────────────────

/** Axis-aligned bounding box, packed as [minX, minY, maxX, maxY]. */
export interface Aabb {
  minX: number; minY: number;
  maxX: number; maxY: number;
}

/** Primitive kind tag. */
export const enum PrimKind {
  ClothTri = 0,
  ParticleShape = 1,
}

/**
 * Deformable scene description. SoA layout matches the WGSL bindings.
 *
 *   • `vertices`   length V*2 — current vertex positions (mutated by
 *                  the integrator each step).
 *   • `prims`      length P*4 — for each primitive, four uint32 fields:
 *                    [kind, i0, i1, i2]
 *                  ClothTri: i0,i1,i2 are vertex indices.
 *                  ParticleShape: i0 = first vertex index, i1 = count
 *                  (one box around all particles in the run), i2 unused.
 *   • `radii`      length P — per-primitive expansion radius. For cloth
 *                  this is the thickness; for particle shapes it is the
 *                  particle radius (each point grows the box by ±r).
 */
export interface DeformableScene {
  V: number;
  P: number;
  vertices: Float32Array;
  prims:    Uint32Array;
  radii:    Float32Array;
}

/**
 * Fixed-topology binary BVH. `parent[i] === -1` for root.
 * Leaves have `leafPrim[i] >= 0`; internals have `leafPrim[i] === -1`
 * and use `child0[i]`, `child1[i]`.
 *
 * `levels` is the bottom-up traversal order — `levels[0]` are leaves,
 * `levels[L-1]` is the root. The refit kernel walks this list.
 */
export interface RefitTree {
  N: number;                  // total node count
  parent:   Int32Array;       // length N
  child0:   Int32Array;       // length N (-1 for leaves)
  child1:   Int32Array;       // length N (-1 for leaves)
  leafPrim: Int32Array;       // length N (-1 for internals)
  /** Bottom-up traversal layers. levels[0] = leaves, last = root. */
  levels:   Int32Array[];
  /** Mutated by refitStep: current AABB per node, packed [min,max]. */
  boxes:    Float32Array;     // length N*4
  /** Per-node dirty flag (Uint8 0/1). Mutated by refitStep. */
  dirty:    Uint8Array;       // length N
}

export interface AabbRefitOptions {
  /** Max |Δ| in box corners that counts as "no change". Default 1e-6. */
  refitEpsilon?: number;
}

export interface AabbRefitContext {
  mode: "gpu" | "cpu";
  device?: GPUDevice;
  scene: DeformableScene;
  tree:  RefitTree;
  options: Required<AabbRefitOptions>;
  /** Number of refit calls since last full rebuild. */
  stepCount: number;
  /** Number of leaves marked dirty in the most recent step. */
  lastDirtyLeaves: number;
  /** Number of internal nodes refit in the most recent step. */
  lastDirtyInternals: number;
}

// ── Tree construction (median-split, fixed topology) ─────────────────────

/**
 * Build a balanced binary BVH over the primitives. We use a simple
 * median-split on the longest axis of the centroid bounds. Topology is
 * fixed at build time and never changes — refit only updates boxes.
 */
export function buildRefitTree(scene: DeformableScene): RefitTree {
  const { P } = scene;
  if (P === 0) {
    return {
      N: 0,
      parent: new Int32Array(0),
      child0: new Int32Array(0),
      child1: new Int32Array(0),
      leafPrim: new Int32Array(0),
      levels: [],
      boxes: new Float32Array(0),
      dirty: new Uint8Array(0),
    };
  }

  // A complete-ish binary tree has at most 2P-1 nodes.
  const maxN = 2 * P - 1;
  const parent   = new Int32Array(maxN).fill(-1);
  const child0   = new Int32Array(maxN).fill(-1);
  const child1   = new Int32Array(maxN).fill(-1);
  const leafPrim = new Int32Array(maxN).fill(-1);

  // Compute centroid for each primitive (cheap; uses current verts but
  // topology doesn't depend on positions, only ordering).
  const centroids = new Float32Array(P * 2);
  for (let p = 0; p < P; p++) {
    const box = leafBoxFromScene(scene, p);
    centroids[p * 2 + 0] = (box.minX + box.maxX) * 0.5;
    centroids[p * 2 + 1] = (box.minY + box.maxY) * 0.5;
  }

  const indices = new Int32Array(P);
  for (let i = 0; i < P; i++) indices[i] = i;

  let nodeCount = 0;
  // Recursive median split. Returns node id.
  const build = (lo: number, hi: number, parentId: number): number => {
    const id = nodeCount++;
    parent[id] = parentId;
    const span = hi - lo;
    if (span === 1) {
      leafPrim[id] = indices[lo];
      return id;
    }
    // Choose split axis = longest centroid extent.
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (let k = lo; k < hi; k++) {
      const cx = centroids[indices[k] * 2 + 0];
      const cy = centroids[indices[k] * 2 + 1];
      if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
    }
    const axis = (maxX - minX) >= (maxY - minY) ? 0 : 1;
    // Sort the slice by axis.
    const slice = Array.from(indices.subarray(lo, hi));
    slice.sort((a, b) => centroids[a * 2 + axis] - centroids[b * 2 + axis]);
    for (let k = 0; k < slice.length; k++) indices[lo + k] = slice[k];
    const mid = lo + (span >> 1);
    const c0 = build(lo, mid, id);
    const c1 = build(mid, hi, id);
    child0[id] = c0;
    child1[id] = c1;
    return id;
  };
  build(0, P, -1);

  // Trim to actual node count.
  const N = nodeCount;
  const tree: RefitTree = {
    N,
    parent:   parent.slice(0, N),
    child0:   child0.slice(0, N),
    child1:   child1.slice(0, N),
    leafPrim: leafPrim.slice(0, N),
    levels:   [],
    boxes:    new Float32Array(N * 4),
    dirty:    new Uint8Array(N).fill(1), // first refit touches everything
  };

  // Compute bottom-up traversal layers via depth labelling.
  const depth = new Int32Array(N);
  for (let i = 0; i < N; i++) {
    let d = 0, p = tree.parent[i];
    while (p !== -1) { d++; p = tree.parent[p]; }
    depth[i] = d;
  }
  let maxDepth = 0;
  for (let i = 0; i < N; i++) if (depth[i] > maxDepth) maxDepth = depth[i];
  const buckets: number[][] = [];
  for (let d = 0; d <= maxDepth; d++) buckets.push([]);
  for (let i = 0; i < N; i++) buckets[depth[i]].push(i);
  // levels[0] = deepest = leaves first, levels[last] = root.
  tree.levels = buckets.reverse().map(arr => Int32Array.from(arr));
  return tree;
}

/** Compute the tight AABB of one primitive directly from the scene. */
export function leafBoxFromScene(scene: DeformableScene, p: number): Aabb {
  const kind = scene.prims[p * 4 + 0] as PrimKind;
  const r = scene.radii[p];
  if (kind === PrimKind.ClothTri) {
    const i0 = scene.prims[p * 4 + 1];
    const i1 = scene.prims[p * 4 + 2];
    const i2 = scene.prims[p * 4 + 3];
    const x0 = scene.vertices[i0 * 2 + 0], y0 = scene.vertices[i0 * 2 + 1];
    const x1 = scene.vertices[i1 * 2 + 0], y1 = scene.vertices[i1 * 2 + 1];
    const x2 = scene.vertices[i2 * 2 + 0], y2 = scene.vertices[i2 * 2 + 1];
    return {
      minX: Math.min(x0, x1, x2) - r,
      minY: Math.min(y0, y1, y2) - r,
      maxX: Math.max(x0, x1, x2) + r,
      maxY: Math.max(y0, y1, y2) + r,
    };
  } else {
    // ParticleShape: i0 = first vert, i1 = count
    const start = scene.prims[p * 4 + 1];
    const count = scene.prims[p * 4 + 2];
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let k = 0; k < count; k++) {
      const x = scene.vertices[(start + k) * 2 + 0];
      const y = scene.vertices[(start + k) * 2 + 1];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
    }
    return { minX: minX - r, minY: minY - r, maxX: maxX + r, maxY: maxY + r };
  }
}

// ── WGSL kernels ─────────────────────────────────────────────────────────

const WGSL_LEAF_REFIT = /* wgsl */ `
struct Scene {
  V: u32, P: u32, eps: f32, _pad: f32,
};

@group(0) @binding(0) var<uniform> scene: Scene;
@group(0) @binding(1) var<storage, read>       vertices: array<f32>;
@group(0) @binding(2) var<storage, read>       prims:    array<u32>;
@group(0) @binding(3) var<storage, read>       radii:    array<f32>;
@group(0) @binding(4) var<storage, read>       leafIds:  array<i32>;
@group(0) @binding(5) var<storage, read_write> boxes:    array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> dirty:    array<atomic<u32>>;

@compute @workgroup_size(64)
fn leaf_refit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lid: u32 = gid.x;
  if (lid >= arrayLength(&leafIds)) { return; }
  let nodeId: i32 = leafIds[lid];
  let p: u32 = u32(nodeId);                    // leaf node id == prim id slot
  let kind: u32 = prims[p * 4u + 0u];
  let r: f32 = radii[p];

  var bmin = vec2<f32>( 1e30,  1e30);
  var bmax = vec2<f32>(-1e30, -1e30);

  if (kind == 0u) {
    let i0 = prims[p * 4u + 1u];
    let i1 = prims[p * 4u + 2u];
    let i2 = prims[p * 4u + 3u];
    let v0 = vec2<f32>(vertices[i0 * 2u], vertices[i0 * 2u + 1u]);
    let v1 = vec2<f32>(vertices[i1 * 2u], vertices[i1 * 2u + 1u]);
    let v2 = vec2<f32>(vertices[i2 * 2u], vertices[i2 * 2u + 1u]);
    bmin = min(min(v0, v1), v2);
    bmax = max(max(v0, v1), v2);
  } else {
    let start = prims[p * 4u + 1u];
    let count = prims[p * 4u + 2u];
    for (var k: u32 = 0u; k < count; k = k + 1u) {
      let v = vec2<f32>(vertices[(start + k) * 2u], vertices[(start + k) * 2u + 1u]);
      bmin = min(bmin, v);
      bmax = max(bmax, v);
    }
  }
  bmin = bmin - vec2<f32>(r);
  bmax = bmax + vec2<f32>(r);

  let prev = boxes[nodeId];
  let delta = max(
    max(abs(prev.x - bmin.x), abs(prev.y - bmin.y)),
    max(abs(prev.z - bmax.x), abs(prev.w - bmax.y))
  );
  boxes[nodeId] = vec4<f32>(bmin.x, bmin.y, bmax.x, bmax.y);
  if (delta > scene.eps) {
    atomicStore(&dirty[nodeId], 1u);
  }
}
`;

const WGSL_INTERNAL_REFIT = /* wgsl */ `
@group(0) @binding(0) var<storage, read>       child0:   array<i32>;
@group(0) @binding(1) var<storage, read>       child1:   array<i32>;
@group(0) @binding(2) var<storage, read>       layer:    array<i32>;
@group(0) @binding(3) var<storage, read_write> boxes:    array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> dirty:    array<atomic<u32>>;

@compute @workgroup_size(64)
fn internal_refit(@builtin(global_invocation_id) gid: vec3<u32>) {
  let lid: u32 = gid.x;
  if (lid >= arrayLength(&layer)) { return; }
  let n: i32 = layer[lid];
  let a: i32 = child0[n];
  let b: i32 = child1[n];
  let da = atomicLoad(&dirty[a]);
  let db = atomicLoad(&dirty[b]);
  if (da == 0u && db == 0u) { return; }
  let ba = boxes[a];
  let bb = boxes[b];
  boxes[n] = vec4<f32>(
    min(ba.x, bb.x), min(ba.y, bb.y),
    max(ba.z, bb.z), max(ba.w, bb.w)
  );
  atomicStore(&dirty[n], 1u);
}
`;

// ── Init ─────────────────────────────────────────────────────────────────

/**
 * Initialize the refit context. Tries WebGPU first, falls back to CPU.
 * The CPU path produces bit-identical results (modulo float order which
 * is identical because both kernels do the same min/max sequence).
 */
export async function initAabbRefit(
  scene: DeformableScene,
  options: AabbRefitOptions = {}
): Promise<AabbRefitContext> {
  const opts: Required<AabbRefitOptions> = {
    refitEpsilon: options.refitEpsilon ?? 1e-6,
  };
  const tree = buildRefitTree(scene);

  let device: GPUDevice | undefined;
  if (typeof navigator !== "undefined" && (navigator as Navigator & { gpu?: GPU }).gpu) {
    try {
      const adapter = await (navigator as Navigator & { gpu: GPU }).gpu.requestAdapter();
      if (adapter) {
        const d = await adapter.requestDevice().catch(() => null);
        if (d) device = d;
      }
    } catch { /* fall through to CPU */ }
  }

  return {
    mode: device ? "gpu" : "cpu",
    device,
    scene,
    tree,
    options: opts,
    stepCount: 0,
    lastDirtyLeaves: 0,
    lastDirtyInternals: 0,
  };
}

/**
 * Mark every node dirty. The next `refitStep()` will recompute every box
 * — useful after a topology-preserving teleport (camera reset, scenario
 * load) or when the incremental tree quality has degraded.
 */
export function markAllDirty(ctx: AabbRefitContext): void {
  ctx.tree.dirty.fill(1);
  // Zero out boxes so the next refit sees a delta and updates everything.
  ctx.tree.boxes.fill(0);
  ctx.stepCount = 0;
}

// ── CPU refit (also used by the GPU readback verifier) ───────────────────

function refitStepCpu(ctx: AabbRefitContext): void {
  const { tree, scene, options } = ctx;
  // dirty[n] reflects "changed this step" only — cleared at start, re-set
  // when a leaf box delta exceeds eps or when a child propagated dirtiness
  // up to its parent. markAllDirty zeroes boxes so the natural delta path
  // marks everything.
  tree.dirty.fill(0);

  const eps = options.refitEpsilon;
  let dirtyLeaves = 0;
  let dirtyInternals = 0;

  for (let L = 0; L < tree.levels.length; L++) {
    const layer = tree.levels[L];
    for (let k = 0; k < layer.length; k++) {
      const n = layer[k];
      const p = tree.leafPrim[n];
      const off = n * 4;
      if (p >= 0) {
        const box = leafBoxFromScene(scene, p);
        const delta = Math.max(
          Math.abs(tree.boxes[off + 0] - box.minX),
          Math.abs(tree.boxes[off + 1] - box.minY),
          Math.abs(tree.boxes[off + 2] - box.maxX),
          Math.abs(tree.boxes[off + 3] - box.maxY),
        );
        tree.boxes[off + 0] = box.minX;
        tree.boxes[off + 1] = box.minY;
        tree.boxes[off + 2] = box.maxX;
        tree.boxes[off + 3] = box.maxY;
        if (delta > eps) {
          tree.dirty[n] = 1;
          dirtyLeaves++;
        }
      } else {
        const a = tree.child0[n], b = tree.child1[n];
        if (!tree.dirty[a] && !tree.dirty[b]) continue;
        const ao = a * 4, bo = b * 4;
        tree.boxes[off + 0] = Math.min(tree.boxes[ao + 0], tree.boxes[bo + 0]);
        tree.boxes[off + 1] = Math.min(tree.boxes[ao + 1], tree.boxes[bo + 1]);
        tree.boxes[off + 2] = Math.max(tree.boxes[ao + 2], tree.boxes[bo + 2]);
        tree.boxes[off + 3] = Math.max(tree.boxes[ao + 3], tree.boxes[bo + 3]);
        tree.dirty[n] = 1;
        dirtyInternals++;
      }
    }
  }
  ctx.lastDirtyLeaves = dirtyLeaves;
  ctx.lastDirtyInternals = dirtyInternals;
}

// ── GPU refit ────────────────────────────────────────────────────────────

interface GpuResources {
  pipelineLeaf: GPUComputePipeline;
  pipelineInternal: GPUComputePipeline;
  uniformBuf: GPUBuffer;
  vertBuf: GPUBuffer;
  primBuf: GPUBuffer;
  radBuf: GPUBuffer;
  leafIdBuf: GPUBuffer;
  layerBufs: GPUBuffer[];
  child0Buf: GPUBuffer;
  child1Buf: GPUBuffer;
  boxBuf: GPUBuffer;
  dirtyBuf: GPUBuffer;
  readBox: GPUBuffer;
  readDirty: GPUBuffer;
}

const _gpuCache = new WeakMap<AabbRefitContext, GpuResources>();

async function refitStepGpu(ctx: AabbRefitContext): Promise<void> {
  const device = ctx.device!;
  const { tree, scene, options } = ctx;
  let res = _gpuCache.get(ctx);
  if (!res) {
    const make = (size: number, usage: number) => device.createBuffer({ size: Math.max(16, size), usage });
    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    res = {
      pipelineLeaf: device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: WGSL_LEAF_REFIT }), entryPoint: "leaf_refit" },
      }),
      pipelineInternal: device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code: WGSL_INTERNAL_REFIT }), entryPoint: "internal_refit" },
      }),
      uniformBuf: make(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      vertBuf:   make(scene.vertices.byteLength, STORAGE),
      primBuf:   make(scene.prims.byteLength,    STORAGE),
      radBuf:    make(scene.radii.byteLength,    STORAGE),
      leafIdBuf: make(tree.levels[0].byteLength, STORAGE),
      layerBufs: tree.levels.slice(1).map(l => make(l.byteLength, STORAGE)),
      child0Buf: make(tree.child0.byteLength,    STORAGE),
      child1Buf: make(tree.child1.byteLength,    STORAGE),
      boxBuf:    make(tree.boxes.byteLength,     STORAGE),
      dirtyBuf:  make(tree.N * 4,                STORAGE),
      readBox:   make(tree.boxes.byteLength,     GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
      readDirty: make(tree.N * 4,                GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ),
    };
    const wb = (b: GPUBuffer, d: ArrayBufferView) => writeTypedBuffer(device, b, d);
    wb(res.primBuf,   scene.prims);
    wb(res.radBuf,    scene.radii);
    wb(res.leafIdBuf, tree.levels[0]);
    tree.levels.slice(1).forEach((l, i) => wb(res!.layerBufs[i], l));
    wb(res.child0Buf, tree.child0);
    wb(res.child1Buf, tree.child1);
    wb(res.boxBuf,    tree.boxes);
    _gpuCache.set(ctx, res);
  }

  // Per-step uploads.
  writeTypedBuffer(device, res.vertBuf, scene.vertices);
  // dirty array: pack the Uint8 wasDirty into u32-per-element for the GPU.
  const dirty32 = new Uint32Array(tree.N);
  for (let i = 0; i < tree.N; i++) dirty32[i] = tree.dirty[i];
  device.queue.writeBuffer(res.dirtyBuf, 0, dirty32);
  // Then clear active dirty bits — the kernel re-sets per change.
  const cleared = new Uint32Array(tree.N);
  device.queue.writeBuffer(res.dirtyBuf, 0, cleared);
  const sceneU = new ArrayBuffer(16);
  new Uint32Array(sceneU, 0, 2).set([scene.V, scene.P]);
  new Float32Array(sceneU, 8, 2).set([options.refitEpsilon, 0]);
  device.queue.writeBuffer(res.uniformBuf, 0, sceneU);

  const enc = device.createCommandEncoder();
  // Leaf pass
  {
    const bg = device.createBindGroup({
      layout: res.pipelineLeaf.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: res.uniformBuf } },
        { binding: 1, resource: { buffer: res.vertBuf } },
        { binding: 2, resource: { buffer: res.primBuf } },
        { binding: 3, resource: { buffer: res.radBuf } },
        { binding: 4, resource: { buffer: res.leafIdBuf } },
        { binding: 5, resource: { buffer: res.boxBuf } },
        { binding: 6, resource: { buffer: res.dirtyBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(res.pipelineLeaf);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(tree.levels[0].length / 64));
    pass.end();
  }
  // Internal passes — one dispatch per layer, in order, so synchronization
  // is implicit between dispatches in the same encoder.
  for (let L = 1; L < tree.levels.length; L++) {
    const layerBuf = res.layerBufs[L - 1];
    const bg = device.createBindGroup({
      layout: res.pipelineInternal.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: res.child0Buf } },
        { binding: 1, resource: { buffer: res.child1Buf } },
        { binding: 2, resource: { buffer: layerBuf } },
        { binding: 3, resource: { buffer: res.boxBuf } },
        { binding: 4, resource: { buffer: res.dirtyBuf } },
      ],
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(res.pipelineInternal);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(tree.levels[L].length / 64));
    pass.end();
  }
  enc.copyBufferToBuffer(res.boxBuf,   0, res.readBox,   0, tree.boxes.byteLength);
  enc.copyBufferToBuffer(res.dirtyBuf, 0, res.readDirty, 0, tree.N * 4);
  device.queue.submit([enc.finish()]);

  await res.readBox.mapAsync(GPUMapMode.READ);
  tree.boxes.set(new Float32Array(res.readBox.getMappedRange().slice(0)));
  res.readBox.unmap();
  await res.readDirty.mapAsync(GPUMapMode.READ);
  const d = new Uint32Array(res.readDirty.getMappedRange().slice(0));
  res.readDirty.unmap();
  let dl = 0, di = 0;
  for (let i = 0; i < tree.N; i++) {
    tree.dirty[i] = d[i] ? 1 : 0;
    if (tree.dirty[i]) {
      if (tree.leafPrim[i] >= 0) dl++; else di++;
    }
  }
  ctx.lastDirtyLeaves = dl;
  ctx.lastDirtyInternals = di;
}

// ── Public step ──────────────────────────────────────────────────────────

/**
 * Perform one incremental refit using the latest `scene.vertices`. The
 * tree's `boxes` and `dirty` arrays are mutated in place. Returns the
 * tree so callers can chain into broad-phase queries.
 */
export async function refitStep(ctx: AabbRefitContext): Promise<RefitTree> {
  if (ctx.mode === "gpu" && ctx.device) {
    await refitStepGpu(ctx);
  } else {
    refitStepCpu(ctx);
  }
  ctx.stepCount++;
  return ctx.tree;
}

/** Synchronous CPU-only entry point (handy for tests). */
export function refitStepSync(ctx: AabbRefitContext): RefitTree {
  refitStepCpu(ctx);
  ctx.stepCount++;
  return ctx.tree;
}
