/**
 * GPU-accelerated traversal for the CSR topology graph.
 *
 *   kHopCpu / kHopGpu       — multi-seed bounded BFS (label + distance per node)
 *   computeHalosCpu / Gpu   — per-partition halo node lists from owners[]
 *
 * The WebGPU backend uploads the CSR (`neighborOffsets` + `neighborIdx`) once
 * and reuses the buffers across many traversal calls — the typical access
 * pattern when iterating partitions or running multiple frontier expansions.
 *
 * CPU paths are exposed so the rest of the engine can call them without a
 * GPU; the GPU paths are wire-compatible (same outputs, just faster on big
 * graphs) so the topology panel can A/B them.
 */

import type { TopologyGraph } from "./graph";

const WG = 64;

/** Distance value meaning "not reached within budget". */
export const KHOP_INF: number = 0xff;

export interface KHopResult {
  /** Per node: 0 if a seed, k if reached via k hops, KHOP_INF if unreached. */
  dist: Uint8Array;
  /** Per node: 0 if unreached, otherwise (seedLabel + 1). */
  label: Uint32Array;
  /** Number of relaxation iterations actually run. */
  iters: number;
  ms: number;
}

/** Multi-source bounded BFS — `seedLabel[i] === 0` means "not a seed". */
export function kHopCpu(g: TopologyGraph, seedLabel: Uint32Array, k: number): KHopResult {
  const t0 = performance.now();
  const N = g.nodes.length;
  let dist = new Uint8Array(N).fill(KHOP_INF);
  let label = new Uint32Array(N);
  for (let i = 0; i < N; i++) {
    if (seedLabel[i] !== 0) { dist[i] = 0; label[i] = seedLabel[i]; }
  }
  const off = g.neighborOffsets;
  const nbr = g.neighborIdx;
  let iters = 0;
  // Strict synchronous relaxation (ping-pong) — wire-compatible with the GPU
  // pass which always reads from the previous frame's buffers.
  for (; iters < k; iters++) {
    const nextDist = new Uint8Array(dist);
    const nextLabel = new Uint32Array(label);
    let changed = false;
    for (let i = 0; i < N; i++) {
      if (dist[i] === 0) continue;
      const a = off[i], b = off[i + 1];
      let bestD = dist[i], bestL = label[i];
      for (let p = a; p < b; p++) {
        const j = nbr[p];
        const dj = dist[j];
        if (dj >= KHOP_INF - 1) continue;
        const cand = dj + 1;
        if (cand <= k && cand < bestD) { bestD = cand; bestL = label[j]; }
      }
      if (bestD !== dist[i]) { nextDist[i] = bestD; nextLabel[i] = bestL; changed = true; }
    }
    dist = nextDist; label = nextLabel;
    if (!changed) break;
  }
  return { dist, label, iters, ms: performance.now() - t0 };
}

export interface HaloResult {
  /** Bitmask per node — bit p set ⇒ node neighbors a partition-p resident. */
  mask: Uint32Array;
  /** Per-partition halo node lists (compatible with TopologyPartitionPlan.halos). */
  halos: number[][];
  ms: number;
}

/** Per-partition halo computation. P must be ≤ 32 (mask is a u32). */
export function computeHalosCpu(g: TopologyGraph, owners: Uint32Array, P: number): HaloResult {
  const t0 = performance.now();
  if (P > 32) throw new Error(`computeHalosCpu: P=${P} exceeds 32-bit mask`);
  const N = g.nodes.length;
  const mask = new Uint32Array(N);
  const off = g.neighborOffsets;
  const nbr = g.neighborIdx;
  for (let i = 0; i < N; i++) {
    const myP = owners[i];
    let m = 0;
    const a = off[i], b = off[i + 1];
    for (let p = a; p < b; p++) {
      const np = owners[nbr[p]];
      if (np !== myP) m |= (1 << np);
    }
    mask[i] = m >>> 0;
  }
  const halos: number[][] = Array.from({ length: P }, () => []);
  for (let i = 0; i < N; i++) {
    const m = mask[i];
    if (!m) continue;
    const myP = owners[i];
    for (let p = 0; p < P; p++) {
      if (p === myP) continue;
      // node i is a halo of partition p iff p ∈ neighbors-of-i partitions
      // AND owners[i] != p (already guaranteed by the loop guard).
      if ((m >>> p) & 1) halos[p].push(i);
    }
  }
  return { mask, halos, ms: performance.now() - t0 };
}

// ─── WebGPU backend ──────────────────────────────────────────────────────

export interface CsrGpuBackend {
  available: true;
  device: GPUDevice;
  N: number;
  E: number;
  kHop(seedLabel: Uint32Array, k: number): Promise<KHopResult>;
  computeHalos(owners: Uint32Array, P: number): Promise<HaloResult>;
  destroy(): void;
}

export interface CsrGpuUnavailable { available: false; reason: string }

const KHOP_WGSL = /* wgsl */ `
struct Ctl { n: u32, inf: u32, _pad0: u32, _pad1: u32, };
@group(0) @binding(0) var<uniform> ctl: Ctl;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read> distIn: array<u32>;
@group(0) @binding(4) var<storage, read> labelIn: array<u32>;
@group(0) @binding(5) var<storage, read_write> distOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> labelOut: array<u32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ctl.n) { return; }
  var bestD: u32 = distIn[i];
  var bestL: u32 = labelIn[i];
  if (bestD == 0u) { distOut[i] = 0u; labelOut[i] = bestL; return; }
  let a = off[i];
  let b = off[i + 1u];
  for (var p: u32 = a; p < b; p = p + 1u) {
    let j = nbr[p];
    let dj = distIn[j];
    if (dj + 1u < bestD && dj < ctl.inf) {
      bestD = dj + 1u;
      bestL = labelIn[j];
    }
  }
  distOut[i] = bestD;
  labelOut[i] = bestL;
}
`;

const HALO_WGSL = /* wgsl */ `
struct Ctl { n: u32, p: u32, _pad0: u32, _pad1: u32, };
@group(0) @binding(0) var<uniform> ctl: Ctl;
@group(0) @binding(1) var<storage, read> off: array<u32>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read> owners: array<u32>;
@group(0) @binding(4) var<storage, read_write> mask: array<u32>;

@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= ctl.n) { return; }
  let myP = owners[i];
  var m: u32 = 0u;
  let a = off[i];
  let b = off[i + 1u];
  for (var p: u32 = a; p < b; p = p + 1u) {
    let np = owners[nbr[p]];
    if (np != myP) {
      m = m | (1u << np);
    }
  }
  mask[i] = m;
}
`;

export async function createCsrGpuBackend(g: TopologyGraph): Promise<CsrGpuBackend | CsrGpuUnavailable> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu unavailable" };
  }
  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) return { available: false, reason: "no GPU adapter" };
  let device: GPUDevice;
  try { device = await adapter.requestDevice(); }
  catch (e) { return { available: false, reason: `requestDevice failed: ${(e as Error).message}` }; }

  const N = g.nodes.length;
  const E = g.neighborIdx.length;

  const offBuf = device.createBuffer({ size: g.neighborOffsets.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(offBuf, 0, g.neighborOffsets as BufferSource);
  const nbrBuf = device.createBuffer({ size: Math.max(4, g.neighborIdx.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  if (g.neighborIdx.byteLength > 0) device.queue.writeBuffer(nbrBuf, 0, g.neighborIdx as BufferSource);

  // Pipelines
  const khopMod = device.createShaderModule({ code: KHOP_WGSL });
  const khopPipe = device.createComputePipeline({ layout: "auto", compute: { module: khopMod, entryPoint: "main" } });
  const haloMod = device.createShaderModule({ code: HALO_WGSL });
  const haloPipe = device.createComputePipeline({ layout: "auto", compute: { module: haloMod, entryPoint: "main" } });

  // Reusable per-node u32 buffers
  const u32Bytes = N * 4;
  const distA = device.createBuffer({ size: u32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const distB = device.createBuffer({ size: u32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const labelA = device.createBuffer({ size: u32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const labelB = device.createBuffer({ size: u32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const ownerBuf = device.createBuffer({ size: u32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const maskBuf = device.createBuffer({ size: u32Bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: u32Bytes * 2, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

  const ctlKhopBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const ctlHaloBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const dispatch = Math.ceil(N / WG);

  async function kHop(seedLabel: Uint32Array, k: number): Promise<KHopResult> {
    if (seedLabel.length !== N) throw new Error(`kHop: seedLabel length ${seedLabel.length} ≠ N=${N}`);
    const t0 = performance.now();
    const INF = 0xff;
    // Init dist0/label0
    const distInit = new Uint32Array(N);
    for (let i = 0; i < N; i++) distInit[i] = seedLabel[i] !== 0 ? 0 : INF;
    device.queue.writeBuffer(distA, 0, distInit);
    device.queue.writeBuffer(labelA, 0, seedLabel as BufferSource);
    device.queue.writeBuffer(ctlKhopBuf, 0, new Uint32Array([N, INF, 0, 0]));

    let inDist = distA, outDist = distB;
    let inLab = labelA, outLab = labelB;

    for (let it = 0; it < k; it++) {
      const bg = device.createBindGroup({
        layout: khopPipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: ctlKhopBuf } },
          { binding: 1, resource: { buffer: offBuf } },
          { binding: 2, resource: { buffer: nbrBuf } },
          { binding: 3, resource: { buffer: inDist } },
          { binding: 4, resource: { buffer: inLab } },
          { binding: 5, resource: { buffer: outDist } },
          { binding: 6, resource: { buffer: outLab } },
        ],
      });
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(khopPipe);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(dispatch);
      pass.end();
      device.queue.submit([enc.finish()]);
      [inDist, outDist] = [outDist, inDist];
      [inLab, outLab] = [outLab, inLab];
    }

    // Readback inDist + inLab
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(inDist, 0, readback, 0, u32Bytes);
    enc.copyBufferToBuffer(inLab, 0, readback, u32Bytes, u32Bytes);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const view = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();

    const dist = new Uint8Array(N);
    const label = new Uint32Array(N);
    for (let i = 0; i < N; i++) {
      const d = view[i];
      dist[i] = d >= INF ? KHOP_INF : d;
      label[i] = view[N + i];
    }
    return { dist, label, iters: k, ms: performance.now() - t0 };
  }

  async function computeHalos(owners: Uint32Array, P: number): Promise<HaloResult> {
    if (owners.length !== N) throw new Error(`computeHalos: owners length ${owners.length} ≠ N=${N}`);
    if (P > 32) throw new Error(`computeHalos: P=${P} exceeds 32-bit mask`);
    const t0 = performance.now();
    device.queue.writeBuffer(ownerBuf, 0, owners as BufferSource);
    device.queue.writeBuffer(ctlHaloBuf, 0, new Uint32Array([N, P, 0, 0]));
    const bg = device.createBindGroup({
      layout: haloPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: ctlHaloBuf } },
        { binding: 1, resource: { buffer: offBuf } },
        { binding: 2, resource: { buffer: nbrBuf } },
        { binding: 3, resource: { buffer: ownerBuf } },
        { binding: 4, resource: { buffer: maskBuf } },
      ],
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(haloPipe);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(dispatch);
    pass.end();
    enc.copyBufferToBuffer(maskBuf, 0, readback, 0, u32Bytes);
    device.queue.submit([enc.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const mask = new Uint32Array(readback.getMappedRange().slice(0, u32Bytes));
    readback.unmap();
    const halos: number[][] = Array.from({ length: P }, () => []);
    for (let i = 0; i < N; i++) {
      const m = mask[i];
      if (!m) continue;
      const myP = owners[i];
      for (let p = 0; p < P; p++) {
        if (p === myP) continue;
        if ((m >>> p) & 1) halos[p].push(i);
      }
    }
    return { mask, halos, ms: performance.now() - t0 };
  }

  return {
    available: true, device, N, E,
    kHop, computeHalos,
    destroy() {
      offBuf.destroy(); nbrBuf.destroy();
      distA.destroy(); distB.destroy(); labelA.destroy(); labelB.destroy();
      ownerBuf.destroy(); maskBuf.destroy();
      readback.destroy(); ctlKhopBuf.destroy(); ctlHaloBuf.destroy();
    },
  };
}
