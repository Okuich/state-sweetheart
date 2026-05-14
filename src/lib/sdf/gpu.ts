/**
 * WebGPU backend for SparseSDF queries.
 *
 * Pack:
 *   - voxels:   flat Float32Array, BRICK^3 floats per brick.
 *   - meta:     vec4<f32> per brick = (originX, originY, originZ, voxelSize).
 *   - lookup:   per-level dense Int32Array of brick indices (-1 if empty),
 *               concatenated; per-level offsets+dims live in uniforms.
 *
 * One compute shader runs in 4 modes (distance / gradient / collide / nearest).
 * Output is `vec4<f32>` per query so all modes share the same bind layout.
 */

import { BRICK, type SparseSDF } from "./sparseField";

export type GpuMode = "distance" | "gradient" | "collide" | "nearest";

export interface GpuBackend {
  available: true;
  device: GPUDevice;
  brickCount: number;
  levelCount: number;
  /** Run N queries. For "collide", each query.w = sphere radius. Returns vec4 per query. */
  run(mode: GpuMode, points: Float32Array, opts?: { radius?: number; iters?: number }): Promise<{
    out: Float32Array;
    gpuMs: number;
    totalMs: number;
  }>;
  destroy(): void;
}

export interface GpuUnavailable { available: false; reason: string }

const MAX_LEVELS = 4;
const WG = 64;

export async function createGpuSdfBackend(sdf: SparseSDF): Promise<GpuBackend | GpuUnavailable> {
  if (typeof navigator === "undefined" || !("gpu" in navigator)) {
    return { available: false, reason: "navigator.gpu unavailable" };
  }
  const adapter = await (navigator as any).gpu.requestAdapter();
  if (!adapter) return { available: false, reason: "no GPU adapter" };
  const device: GPUDevice = await adapter.requestDevice();

  const levelCount = Math.min(MAX_LEVELS, sdf.stats.levels);
  const vs0 = sdf.options.voxelSize;
  const dims0 = sdf.stats.rootBrickDims;

  // Per-level dense lookup tables.
  const lookups: Int32Array[] = [];
  const dimsPerLevel: [number, number, number][] = [];
  for (let lvl = 0; lvl < levelCount; lvl++) {
    const m = 1 << lvl;
    const dx = dims0[0] * m, dy = dims0[1] * m, dz = dims0[2] * m;
    dimsPerLevel.push([dx, dy, dz]);
    lookups.push(new Int32Array(dx * dy * dz).fill(-1));
  }
  const meta = new Float32Array(sdf.bricks.length * 4);
  const voxels = new Float32Array(sdf.bricks.length * BRICK * BRICK * BRICK);
  const stride = BRICK * BRICK * BRICK;

  for (let i = 0; i < sdf.bricks.length; i++) {
    const b = sdf.bricks[i];
    if (b.level >= levelCount) continue;
    const [cx, cy, cz] = b.coord;
    const [dx, dy] = dimsPerLevel[b.level];
    lookups[b.level][(cz * dy + cy) * dx + cx] = i;
    meta[i * 4 + 0] = b.origin[0];
    meta[i * 4 + 1] = b.origin[1];
    meta[i * 4 + 2] = b.origin[2];
    meta[i * 4 + 3] = b.voxelSize;
    voxels.set(b.voxels, i * stride);
  }

  // Concat lookups + offsets.
  let totalLookup = 0;
  const offsets: number[] = [];
  for (const lk of lookups) { offsets.push(totalLookup); totalLookup += lk.length; }
  const lookupAll = new Int32Array(totalLookup);
  for (let i = 0; i < lookups.length; i++) lookupAll.set(lookups[i], offsets[i]);

  const voxelBuf = device.createBuffer({ size: voxels.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(voxelBuf, 0, voxels);
  const metaBuf = device.createBuffer({ size: meta.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(metaBuf, 0, meta);
  const lookupBuf = device.createBuffer({ size: lookupAll.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(lookupBuf, 0, lookupAll);

  const shader = /* wgsl */ `
    const BRICK : u32 = ${BRICK}u;
    const BRICK3: u32 = ${BRICK * BRICK * BRICK}u;
    const MAX_LEVELS: u32 = ${MAX_LEVELS}u;

    struct LevelInfo { dims: vec4<u32> }; // xyz=dims, w=offset into lookup

    struct Params {
      bboxMin: vec4<f32>,        // xyz, w=vs0
      cfg: vec4<f32>,            // x=band, y=mode, z=eps, w=radius
      counts: vec4<u32>,         // x=N, y=levelCount, z=iters, w=0
      levels: array<LevelInfo, ${MAX_LEVELS}>,
    };

    @group(0) @binding(0) var<storage, read> voxels : array<f32>;
    @group(0) @binding(1) var<storage, read> meta   : array<vec4<f32>>;
    @group(0) @binding(2) var<storage, read> lookup : array<i32>;
    @group(0) @binding(3) var<uniform> P : Params;
    @group(0) @binding(4) var<storage, read> queries : array<vec4<f32>>;
    @group(0) @binding(5) var<storage, read_write> outBuf : array<vec4<f32>>;

    fn trilinear(brick: u32, p: vec3<f32>) -> f32 {
      let m = meta[brick];
      let vs = m.w;
      let l = (p - m.xyz) / vs;
      let bi = f32(BRICK - 1u);
      let lc = clamp(l, vec3<f32>(0.0), vec3<f32>(bi));
      let f0 = floor(lc);
      let i0 = vec3<u32>(u32(f0.x), u32(f0.y), u32(f0.z));
      let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(BRICK - 1u));
      let f = lc - f0;
      let base = brick * BRICK3;
      let c000 = voxels[base + (i0.z * BRICK + i0.y) * BRICK + i0.x];
      let c100 = voxels[base + (i0.z * BRICK + i0.y) * BRICK + i1.x];
      let c010 = voxels[base + (i0.z * BRICK + i1.y) * BRICK + i0.x];
      let c110 = voxels[base + (i0.z * BRICK + i1.y) * BRICK + i1.x];
      let c001 = voxels[base + (i1.z * BRICK + i0.y) * BRICK + i0.x];
      let c101 = voxels[base + (i1.z * BRICK + i0.y) * BRICK + i1.x];
      let c011 = voxels[base + (i1.z * BRICK + i1.y) * BRICK + i0.x];
      let c111 = voxels[base + (i1.z * BRICK + i1.y) * BRICK + i1.x];
      let c00 = mix(c000, c100, f.x);
      let c10 = mix(c010, c110, f.x);
      let c01 = mix(c001, c101, f.x);
      let c11 = mix(c011, c111, f.x);
      let c0  = mix(c00, c10, f.y);
      let c1  = mix(c01, c11, f.y);
      return mix(c0, c1, f.z);
    }

    fn sampleSDF(p: vec3<f32>) -> f32 {
      let band = P.cfg.x;
      let vs0 = P.bboxMin.w;
      let lc = P.counts.y;
      for (var k: u32 = 0u; k < lc; k = k + 1u) {
        let level = lc - 1u - k;
        let li = P.levels[level];
        let scale = f32(1u << level);
        let vs = vs0 / scale;
        let bw = vs * f32(BRICK);
        let rel = (p - P.bboxMin.xyz) / bw;
        if (rel.x < 0.0 || rel.y < 0.0 || rel.z < 0.0) { continue; }
        let cx = u32(floor(rel.x));
        let cy = u32(floor(rel.y));
        let cz = u32(floor(rel.z));
        if (cx >= li.dims.x || cy >= li.dims.y || cz >= li.dims.z) { continue; }
        let idx = lookup[li.dims.w + (cz * li.dims.y + cy) * li.dims.x + cx];
        if (idx < 0) { continue; }
        return trilinear(u32(idx), p);
      }
      return band;
    }

    fn gradAt(p: vec3<f32>) -> vec3<f32> {
      let e = P.cfg.z;
      let dx = sampleSDF(p + vec3<f32>(e,0.0,0.0)) - sampleSDF(p - vec3<f32>(e,0.0,0.0));
      let dy = sampleSDF(p + vec3<f32>(0.0,e,0.0)) - sampleSDF(p - vec3<f32>(0.0,e,0.0));
      let dz = sampleSDF(p + vec3<f32>(0.0,0.0,e)) - sampleSDF(p - vec3<f32>(0.0,0.0,e));
      let inv = 1.0 / (2.0 * e);
      return vec3<f32>(dx, dy, dz) * inv;
    }

    @compute @workgroup_size(${WG})
    fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
      let i = gid.x;
      if (i >= P.counts.x) { return; }
      let q = queries[i];
      let mode = u32(P.cfg.y);
      let p = q.xyz;

      if (mode == 0u) {
        outBuf[i] = vec4<f32>(sampleSDF(p), 0.0, 0.0, 0.0);
      } else if (mode == 1u) {
        let g = gradAt(p);
        outBuf[i] = vec4<f32>(g.x, g.y, g.z, sampleSDF(p));
      } else if (mode == 2u) {
        let r = P.cfg.w;
        let d = sampleSDF(p);
        let depth = r - d;
        if (depth <= 0.0) {
          outBuf[i] = vec4<f32>(0.0);
        } else {
          var n = gradAt(p);
          let ln = max(length(n), 1e-6);
          n = n / ln;
          outBuf[i] = vec4<f32>(depth, n.x, n.y, n.z);
        }
      } else {
        var qp = p;
        var d = sampleSDF(qp);
        let iters = P.counts.z;
        for (var k: u32 = 0u; k < iters; k = k + 1u) {
          var n = gradAt(qp);
          let ln = max(length(n), 1e-6);
          n = n / ln;
          qp = qp - d * n;
          d = sampleSDF(qp);
        }
        outBuf[i] = vec4<f32>(qp.x, qp.y, qp.z, d);
      }
    }
  `;

  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });

  const paramSize = 16 + 16 + 16 + MAX_LEVELS * 16; // bytes
  const paramBuf = device.createBuffer({ size: paramSize, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const bandWorld = sdf.options.bandWidth * vs0;

  return {
    available: true,
    device,
    brickCount: sdf.bricks.length,
    levelCount,
    async run(mode, points, opts) {
      const t0 = performance.now();
      const N = points.length / 3;
      const q = new Float32Array(N * 4);
      for (let i = 0; i < N; i++) {
        q[i * 4] = points[i * 3];
        q[i * 4 + 1] = points[i * 3 + 1];
        q[i * 4 + 2] = points[i * 3 + 2];
      }
      const queryBuf = device.createBuffer({ size: q.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(queryBuf, 0, q);

      const outSize = N * 16;
      const outBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      const readBuf = device.createBuffer({ size: outSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

      const params = new ArrayBuffer(paramSize);
      const f = new Float32Array(params);
      const u = new Uint32Array(params);
      f[0] = sdf.bbox.min[0]; f[1] = sdf.bbox.min[1]; f[2] = sdf.bbox.min[2]; f[3] = vs0;
      const modeIdx = mode === "distance" ? 0 : mode === "gradient" ? 1 : mode === "collide" ? 2 : 3;
      f[4] = bandWorld;
      f[5] = modeIdx;
      f[6] = vs0 * 0.5; // eps
      f[7] = opts?.radius ?? 0.05;
      u[8] = N;
      u[9] = levelCount;
      u[10] = opts?.iters ?? 6;
      u[11] = 0;
      for (let lvl = 0; lvl < MAX_LEVELS; lvl++) {
        const base = 12 + lvl * 4;
        if (lvl < levelCount) {
          u[base] = dimsPerLevel[lvl][0];
          u[base + 1] = dimsPerLevel[lvl][1];
          u[base + 2] = dimsPerLevel[lvl][2];
          u[base + 3] = offsets[lvl];
        } else {
          u[base] = 0; u[base + 1] = 0; u[base + 2] = 0; u[base + 3] = 0;
        }
      }
      device.queue.writeBuffer(paramBuf, 0, params);

      const bg = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: voxelBuf } },
          { binding: 1, resource: { buffer: metaBuf } },
          { binding: 2, resource: { buffer: lookupBuf } },
          { binding: 3, resource: { buffer: paramBuf } },
          { binding: 4, resource: { buffer: queryBuf } },
          { binding: 5, resource: { buffer: outBuf } },
        ],
      });

      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(N / WG));
      pass.end();
      enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, outSize);
      const tg0 = performance.now();
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      const gpuMs = performance.now() - tg0;

      await readBuf.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(readBuf.getMappedRange().slice(0));
      readBuf.unmap();

      queryBuf.destroy();
      outBuf.destroy();
      readBuf.destroy();

      return { out, gpuMs, totalMs: performance.now() - t0 };
    },
    destroy() {
      voxelBuf.destroy();
      metaBuf.destroy();
      lookupBuf.destroy();
      paramBuf.destroy();
    },
  };
}
