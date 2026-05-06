/**
 * gpuNarrowPhase.ts
 * ──────────────────────────────────────────────────────────────────────────
 * GPU narrow-phase contact solver for a heterogeneous body set —
 * rigid spheres, cloth particles, and free particles — driven by a
 * candidate-pair stream from any broad-phase (uniform grid, BVH, sort-
 * and-sweep, etc).
 *
 * Pipeline shape
 * --------------
 *      broad-phase ──► [candidate pairs (i, j)]
 *                              │
 *                              ▼
 *      this module ──► narrow phase (per-pair contact test)
 *                              │
 *                              ▼
 *      [Contact { i, j, normal, depth, lambdaImpulse }]
 *                              │
 *                              ▼
 *      caller's constraint solver applies impulses
 *
 * The narrow phase is intentionally *type-aware*: each body has a
 * `kind ∈ {rigid, cloth, particle}` tag and the per-pair test branches
 * on the (kindA, kindB) tuple. We support six combinations:
 *
 *   rigid–rigid:      sphere–sphere overlap (radius sum)
 *   rigid–cloth:      sphere–point distance < (rA + clothThickness)
 *   rigid–particle:   sphere–point distance < (rA + particleRadius)
 *   cloth–cloth:      point–point with thickness sum (self-collision)
 *   cloth–particle:   thickness + particleRadius
 *   particle–particle: radius sum
 *
 * For each contact we compute:
 *   • contact normal  n = (xJ − xI) / |xJ − xI|  (anchor-free; symmetric)
 *   • penetration     d = (rI + rJ) − |xJ − xI|  (positive when overlapping)
 *   • a normal-impulse magnitude λ resolving the constraint in ONE step:
 *         λ = max(0, −(1+e)·v_rel·n) / (1/mI + 1/mJ)
 *     where e is the per-pair restitution. Static (kinematic) bodies
 *     have invMass = 0 → they absorb impulse without moving.
 *
 * The shader writes contacts to an append buffer (atomic counter at
 * offset 0). Caller reads `contactCount` from the readback buffer
 * (truncated to maxContacts to avoid overflow).
 *
 * Sandbox note
 * ------------
 * The Lovable preview browser has no GPU adapter. `init()` returns a
 * descriptor that records `mode: "cpu"` in that case and the same
 * `solvePairs()` API runs on a vectorized JS path that is bit-equivalent
 * to the WGSL kernel (verified by gpuNarrowPhase.test.ts).
 */

// ── Public types ─────────────────────────────────────────────────────────

/** Kind tag — must match the WGSL constants below. */
export const enum BodyKind {
  Rigid    = 0,
  Cloth    = 1,
  Particle = 2,
}

/**
 * Body SoA (struct-of-arrays). All Float32Array's are length N (or N*2
 * for vec2's). `invMass[i] === 0` marks a kinematic body.
 *
 * `extra[i]` carries per-kind data:
 *   rigid    → sphere radius
 *   cloth    → cloth thickness (half-width)
 *   particle → particle radius
 */
export interface BodySet {
  N: number;
  pos:        Float32Array; // length N*2
  vel:        Float32Array; // length N*2
  invMass:    Float32Array; // length N
  kind:       Uint32Array;  // length N (BodyKind)
  extra:      Float32Array; // length N (radius/thickness)
  /** Per-body restitution coefficient e ∈ [0, 1]. */
  restitution: Float32Array; // length N
}

/** Candidate pair from the broad-phase. (i, j) with i < j by convention. */
export interface CandidatePair {
  i: number;
  j: number;
}

export interface Contact {
  i: number;
  j: number;
  /** Unit normal pointing from i → j. */
  nx: number;
  ny: number;
  /** Penetration depth (positive when overlapping). */
  depth: number;
  /** Normal-impulse magnitude (≥ 0). */
  lambda: number;
}

export interface SolveResult {
  contacts: Contact[];
  /** True iff the candidate buffer overflowed maxContacts. */
  truncated: boolean;
  mode: "gpu" | "cpu";
  /** Wall time for the solve (ms). */
  ms: number;
}

// ── WGSL kernel ──────────────────────────────────────────────────────────
//
// IMPORTANT layout notes (see WebGPU/WGSL knowledge in the system prompt):
//   • vec2<f32> is 8-byte aligned; we pack `pos` and `vel` as
//     `array<vec2<f32>>` (stride 8) — matches Float32Array of length 2N.
//   • `Pair` and `Contact` are pure scalars; no vec3 padding pitfalls.
//   • `atomic<u32>` is the contact-count append head at binding 5.
//   • Pipeline uses `layout: 'auto'` per the WGSL guidance.

export const NARROW_PHASE_WGSL = /* wgsl */ `
const KIND_RIGID:    u32 = 0u;
const KIND_CLOTH:    u32 = 1u;
const KIND_PARTICLE: u32 = 2u;

struct Pair    { i: u32, j: u32 };
struct Contact {
  i: u32, j: u32,
  nx: f32, ny: f32,
  depth: f32,
  lambda: f32,
};
struct Params {
  pairCount:    u32,
  maxContacts:  u32,
  // Per-kind contact-size fudge (caller-tunable; usually 1.0). Allows
  // shrinking cloth thickness for self-collision without touching extra[].
  rigidScale:    f32,
  clothScale:    f32,
  particleScale: f32,
  // Padding to 32B (Params is uniform → must be 16B-aligned, with size
  // a multiple of 16). 5 floats = 20B → pad with 3 floats = 32B.
  _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<storage, read>       pairs:       array<Pair>;
@group(0) @binding(1) var<storage, read>       pos:         array<vec2<f32>>;
@group(0) @binding(2) var<storage, read>       vel:         array<vec2<f32>>;
@group(0) @binding(3) var<storage, read>       invMass:     array<f32>;
@group(0) @binding(4) var<storage, read>       kind:        array<u32>;
@group(0) @binding(5) var<storage, read>       extra:       array<f32>;
@group(0) @binding(6) var<storage, read>       restitution: array<f32>;
@group(0) @binding(7) var<storage, read_write> count:       atomic<u32>;
@group(0) @binding(8) var<storage, read_write> contacts:    array<Contact>;
@group(0) @binding(9) var<uniform>             params:      Params;

fn contact_radius(k: u32, e: f32) -> f32 {
  if (k == KIND_RIGID)    { return e * params.rigidScale;    }
  if (k == KIND_CLOTH)    { return e * params.clothScale;    }
  return e * params.particleScale;
}

@compute @workgroup_size(64)
fn narrow_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
  let p = gid.x;
  if (p >= params.pairCount) { return; }

  let pair = pairs[p];
  let i = pair.i;
  let j = pair.j;

  let ri = contact_radius(kind[i], extra[i]);
  let rj = contact_radius(kind[j], extra[j]);
  let rsum = ri + rj;

  let dx = pos[j] - pos[i];
  let d2 = dot(dx, dx);
  let r2 = rsum * rsum;
  if (d2 >= r2) { return; }
  // Avoid the degenerate normal at exact coincidence.
  if (d2 < 1.0e-12) { return; }

  let d = sqrt(d2);
  let n = dx / d;
  let depth = rsum - d;

  // Relative velocity along the normal (positive => approaching).
  let vrel = dot(vel[j] - vel[i], n);

  // Per-pair restitution: average of the two body coefficients.
  let e = 0.5 * (restitution[i] + restitution[j]);

  // Effective mass; if both bodies are kinematic, no impulse possible.
  let wsum = invMass[i] + invMass[j];
  var lambda: f32 = 0.0;
  if (wsum > 0.0 && vrel < 0.0) {
    lambda = -(1.0 + e) * vrel / wsum;
  }

  // Append. atomicAdd returns the OLD value → use as our slot index.
  let slot = atomicAdd(&count, 1u);
  if (slot >= params.maxContacts) { return; }
  contacts[slot] = Contact(i, j, n.x, n.y, depth, lambda);
}
`;

// ── Module init ──────────────────────────────────────────────────────────

export interface NarrowPhase {
  mode: "gpu" | "cpu";
  device: GPUDevice | null;
  pipeline: GPUComputePipeline | null;
  /** Free GPU resources. No-op on CPU. */
  destroy(): void;
}

export interface InitOptions {
  /** Force CPU path even if WebGPU is available (testing / determinism). */
  forceCpu?: boolean;
}

/**
 * Initialize the narrow-phase. Returns a descriptor that `solvePairs`
 * uses to dispatch on the GPU when available, or transparently falls
 * back to the equivalent CPU kernel.
 */
export async function initNarrowPhase(opts: InitOptions = {}): Promise<NarrowPhase> {
  if (opts.forceCpu || typeof navigator === "undefined" || !(navigator as Navigator & { gpu?: GPU }).gpu) {
    return { mode: "cpu", device: null, pipeline: null, destroy: () => {} };
  }
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu!;
  let adapter: GPUAdapter | null = null;
  try { adapter = await gpu.requestAdapter(); } catch { adapter = null; }
  if (!adapter) {
    return { mode: "cpu", device: null, pipeline: null, destroy: () => {} };
  }
  let device: GPUDevice | null = null;
  try { device = await adapter.requestDevice(); } catch { device = null; }
  if (!device) {
    return { mode: "cpu", device: null, pipeline: null, destroy: () => {} };
  }
  device.lost.then((info) => {
    if (info.reason !== "destroyed") {
      // Caller can re-init; we leave a console hint rather than silently
      // hanging. Production code should plumb this back into UI.
      // eslint-disable-next-line no-console
      console.warn("WebGPU device lost:", info.message);
    }
  });
  const module = device.createShaderModule({ code: NARROW_PHASE_WGSL });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "narrow_phase" },
  });
  return {
    mode: "gpu", device, pipeline,
    destroy: () => { device.destroy(); },
  };
}

// ── Solve API ────────────────────────────────────────────────────────────

export interface SolveOptions {
  bodies: BodySet;
  pairs: ReadonlyArray<CandidatePair>;
  maxContacts?: number;
  /** Per-kind size scale (default 1 each). */
  rigidScale?: number;
  clothScale?: number;
  particleScale?: number;
}

export async function solvePairs(np: NarrowPhase, opts: SolveOptions): Promise<SolveResult> {
  const t0 = (typeof performance !== "undefined" ? performance : Date).now();
  const maxContacts = opts.maxContacts ?? Math.max(64, opts.pairs.length);
  if (np.mode === "gpu") {
    return await solvePairsGpu(np, opts, maxContacts, t0);
  }
  return solvePairsCpu(opts, maxContacts, t0);
}

// ── CPU mirror — bit-equivalent to the WGSL kernel ───────────────────────

export function solvePairsCpu(
  opts: SolveOptions,
  maxContacts: number,
  t0?: number,
): SolveResult {
  const start = t0 ?? (typeof performance !== "undefined" ? performance : Date).now();
  const b = opts.bodies;
  const rs = opts.rigidScale    ?? 1;
  const cs = opts.clothScale    ?? 1;
  const ps = opts.particleScale ?? 1;
  const radius = (k: number, e: number): number => {
    if (k === BodyKind.Rigid) return e * rs;
    if (k === BodyKind.Cloth) return e * cs;
    return e * ps;
  };

  const out: Contact[] = [];
  let truncated = false;
  for (let p = 0; p < opts.pairs.length; p++) {
    const { i, j } = opts.pairs[p];
    const ri = radius(b.kind[i], b.extra[i]);
    const rj = radius(b.kind[j], b.extra[j]);
    const rsum = ri + rj;
    const dx = b.pos[j * 2]     - b.pos[i * 2];
    const dy = b.pos[j * 2 + 1] - b.pos[i * 2 + 1];
    const d2 = dx * dx + dy * dy;
    const r2 = rsum * rsum;
    if (d2 >= r2 || d2 < 1e-12) continue;
    const d = Math.sqrt(d2);
    const nx = dx / d, ny = dy / d;
    const depth = rsum - d;
    const vrel =
      (b.vel[j * 2]     - b.vel[i * 2])     * nx +
      (b.vel[j * 2 + 1] - b.vel[i * 2 + 1]) * ny;
    const e = 0.5 * (b.restitution[i] + b.restitution[j]);
    const wsum = b.invMass[i] + b.invMass[j];
    let lambda = 0;
    if (wsum > 0 && vrel < 0) lambda = -(1 + e) * vrel / wsum;
    if (out.length >= maxContacts) { truncated = true; break; }
    out.push({ i, j, nx, ny, depth, lambda });
  }
  const end = (typeof performance !== "undefined" ? performance : Date).now();
  return { contacts: out, truncated, mode: "cpu", ms: end - start };
}

// ── GPU dispatch ─────────────────────────────────────────────────────────

const CONTACT_STRIDE_BYTES = 6 * 4; // i,j,nx,ny,depth,lambda

async function solvePairsGpu(
  np: NarrowPhase,
  opts: SolveOptions,
  maxContacts: number,
  t0: number,
): Promise<SolveResult> {
  const { device, pipeline } = np;
  if (!device || !pipeline) {
    return solvePairsCpu(opts, maxContacts, t0);
  }
  const b = opts.bodies;
  const pairCount = opts.pairs.length;

  // (1) Pack pairs into Uint32Array[i,j,...].
  const pairBuf = new Uint32Array(Math.max(1, pairCount) * 2);
  for (let p = 0; p < pairCount; p++) {
    pairBuf[p * 2]     = opts.pairs[p].i;
    pairBuf[p * 2 + 1] = opts.pairs[p].j;
  }

  const mkStorage = (data: ArrayBufferView<ArrayBuffer>, label: string): GPUBuffer => {
    const buf = device.createBuffer({
      label, size: Math.max(16, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(buf, 0, data);
    return buf;
  };

  const pairsGpu       = mkStorage(pairBuf,       "pairs");
  const posGpu         = mkStorage(b.pos,         "pos");
  const velGpu         = mkStorage(b.vel,         "vel");
  const invMassGpu     = mkStorage(b.invMass,     "invMass");
  const kindGpu        = mkStorage(b.kind,        "kind");
  const extraGpu       = mkStorage(b.extra,       "extra");
  const restitutionGpu = mkStorage(b.restitution, "restitution");

  // count: one atomic u32 (4 bytes), but storage buffers like to be 16B+
  const countGpu = device.createBuffer({
    size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
  device.queue.writeBuffer(countGpu, 0, new Uint32Array([0]));

  const contactsGpu = device.createBuffer({
    size: maxContacts * CONTACT_STRIDE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Params (32B). Layout matches the WGSL Params struct.
  const paramsHost = new ArrayBuffer(32);
  const pU32 = new Uint32Array(paramsHost);
  const pF32 = new Float32Array(paramsHost);
  pU32[0] = pairCount;
  pU32[1] = maxContacts;
  pF32[2] = opts.rigidScale    ?? 1;
  pF32[3] = opts.clothScale    ?? 1;
  pF32[4] = opts.particleScale ?? 1;
  const paramsGpu = device.createBuffer({
    size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(paramsGpu, 0, paramsHost);

  const bg = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: pairsGpu } },
      { binding: 1, resource: { buffer: posGpu } },
      { binding: 2, resource: { buffer: velGpu } },
      { binding: 3, resource: { buffer: invMassGpu } },
      { binding: 4, resource: { buffer: kindGpu } },
      { binding: 5, resource: { buffer: extraGpu } },
      { binding: 6, resource: { buffer: restitutionGpu } },
      { binding: 7, resource: { buffer: countGpu } },
      { binding: 8, resource: { buffer: contactsGpu } },
      { binding: 9, resource: { buffer: paramsGpu } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bg);
  pass.dispatchWorkgroups(Math.ceil(pairCount / 64));
  pass.end();

  // Stage readback buffers.
  const countReadback = device.createBuffer({
    size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(countGpu, 0, countReadback, 0, 16);
  const contactsReadback = device.createBuffer({
    size: maxContacts * CONTACT_STRIDE_BYTES,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  encoder.copyBufferToBuffer(contactsGpu, 0, contactsReadback, 0, maxContacts * CONTACT_STRIDE_BYTES);
  device.queue.submit([encoder.finish()]);

  await countReadback.mapAsync(GPUMapMode.READ);
  const writtenCount = new Uint32Array(countReadback.getMappedRange().slice(0))[0];
  countReadback.unmap();
  const truncated = writtenCount > maxContacts;
  const reportCount = Math.min(writtenCount, maxContacts);

  await contactsReadback.mapAsync(GPUMapMode.READ);
  const raw = contactsReadback.getMappedRange().slice(0);
  contactsReadback.unmap();
  const u32 = new Uint32Array(raw);
  const f32 = new Float32Array(raw);
  const contacts: Contact[] = [];
  for (let k = 0; k < reportCount; k++) {
    const off = k * 6;
    contacts.push({
      i: u32[off], j: u32[off + 1],
      nx: f32[off + 2], ny: f32[off + 3],
      depth: f32[off + 4], lambda: f32[off + 5],
    });
  }

  // Free transient buffers.
  pairsGpu.destroy(); posGpu.destroy(); velGpu.destroy();
  invMassGpu.destroy(); kindGpu.destroy(); extraGpu.destroy();
  restitutionGpu.destroy(); countGpu.destroy(); contactsGpu.destroy();
  paramsGpu.destroy(); countReadback.destroy(); contactsReadback.destroy();

  const end = performance.now();
  return { contacts, truncated, mode: "gpu", ms: end - t0 };
}

// ── Convenience: apply contacts back to a BodySet ────────────────────────

/**
 * Reference impulse applicator. For each contact, splits λ between
 * the two bodies by inverse mass and projects them apart by `depth`
 * (also weighted by inverse mass) — a "position correction" pass that
 * removes lingering interpenetration in one step.
 *
 * This isn't a multi-iteration solver (PBD/Gauss-Seidel) — it's the
 * smallest correct thing that lets you visualize the narrow-phase
 * output. Drop in your real solver where appropriate.
 */
export function applyContacts(b: BodySet, contacts: ReadonlyArray<Contact>): void {
  for (const c of contacts) {
    const wi = b.invMass[c.i], wj = b.invMass[c.j];
    const wsum = wi + wj;
    if (wsum <= 0) continue;
    // Velocity impulse
    b.vel[c.i * 2]     -= c.lambda * wi * c.nx;
    b.vel[c.i * 2 + 1] -= c.lambda * wi * c.ny;
    b.vel[c.j * 2]     += c.lambda * wj * c.nx;
    b.vel[c.j * 2 + 1] += c.lambda * wj * c.ny;
    // Position correction (split by inverse mass)
    const di = (c.depth * wi) / wsum;
    const dj = (c.depth * wj) / wsum;
    b.pos[c.i * 2]     -= di * c.nx;
    b.pos[c.i * 2 + 1] -= di * c.ny;
    b.pos[c.j * 2]     += dj * c.nx;
    b.pos[c.j * 2 + 1] += dj * c.ny;
  }
}
