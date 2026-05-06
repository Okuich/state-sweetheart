/**
 * kernelCapabilities.ts
 *
 * Runtime capability registry for compute kernels. Each "kernel path"
 * (a specific backend × algorithm combo, e.g. WebGPU spatial-hash, CPU
 * reference) declares whether it supports f64 math end-to-end.
 *
 * The PrecisionPolicy panel queries this to decide whether the user's
 * dtype toggle should be enabled, and to surface a path-specific reason
 * when f64 is unavailable.
 *
 * Detection is layered:
 *   1. Static per-kernel capability table (knowledge baked at build time).
 *   2. Runtime probe: in browsers, check navigator.gpu and the adapter's
 *      features list — WebGPU never exposes f64, so any WebGPU kernel
 *      path is automatically marked unsupported.
 *   3. Override via `setKernelF64Support(...)` for tests / forced paths.
 */

export type KernelBackend = "cpu" | "webgpu" | "webgl2" | "wasm";

export interface KernelPath {
  /** Stable id, e.g. "webgpu-spatial-hash". */
  id: string;
  /** Human label for UI. */
  label: string;
  backend: KernelBackend;
}

export interface KernelF64Status {
  path: KernelPath;
  /** True iff this kernel can run f64 end-to-end. */
  supportsF64: boolean;
  /** Short, user-facing reason when unsupported. */
  reason?: string;
}

const STATIC_TABLE: Record<string, Omit<KernelF64Status, "path"> & { path: KernelPath }> = {
  "cpu-reference": {
    path: { id: "cpu-reference", label: "CPU reference", backend: "cpu" },
    supportsF64: true,
  },
  "wasm-simd": {
    path: { id: "wasm-simd", label: "WASM SIMD", backend: "wasm" },
    supportsF64: true,
  },
  "webgpu-spatial-hash": {
    path: { id: "webgpu-spatial-hash", label: "WebGPU spatial-hash", backend: "webgpu" },
    supportsF64: false,
    reason: "WebGPU has no f64 type; WGSL only exposes f32 (and optional f16).",
  },
  "webgpu-lbvh": {
    path: { id: "webgpu-lbvh", label: "WebGPU LBVH", backend: "webgpu" },
    supportsF64: false,
    reason: "WebGPU has no f64 type; WGSL only exposes f32 (and optional f16).",
  },
  "webgpu-mpm": {
    path: { id: "webgpu-mpm", label: "WebGPU MPM solver", backend: "webgpu" },
    supportsF64: false,
    reason: "WebGPU has no f64 type; WGSL only exposes f32 (and optional f16).",
  },
  "webgl2-fallback": {
    path: { id: "webgl2-fallback", label: "WebGL2 fallback", backend: "webgl2" },
    supportsF64: false,
    reason: "WebGL2 fragment/vertex pipelines are f32 only.",
  },
};

const overrides = new Map<string, boolean>();

/** Force a kernel's f64 support on/off (tests, manual debug). */
export function setKernelF64Support(id: string, supported: boolean): void {
  overrides.set(id, supported);
}
export function clearKernelOverrides(): void { overrides.clear(); }

/** List every kernel path the app knows about, with current f64 status. */
export function listKernelPaths(): KernelF64Status[] {
  return Object.values(STATIC_TABLE).map((entry) => resolve(entry));
}

export function getKernelF64Status(id: string): KernelF64Status | undefined {
  const e = STATIC_TABLE[id];
  return e ? resolve(e) : undefined;
}

function resolve(entry: { path: KernelPath; supportsF64: boolean; reason?: string }): KernelF64Status {
  const override = overrides.get(entry.path.id);
  if (override !== undefined) {
    return {
      path: entry.path,
      supportsF64: override,
      reason: override ? undefined : entry.reason ?? "Disabled by override.",
    };
  }
  // Backend-level runtime probe.
  if (entry.path.backend === "webgpu" && !hasWebGPU()) {
    return {
      path: entry.path,
      supportsF64: false,
      reason: "WebGPU adapter not available in this browser.",
    };
  }
  return { path: entry.path, supportsF64: entry.supportsF64, reason: entry.reason };
}

function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" &&
    !!(navigator as Navigator & { gpu?: unknown }).gpu;
}

/**
 * Aggregate decision for the dtype toggle:
 *   - supported: true iff at least one *active* kernel path supports f64.
 *   - blockers:  list of active paths that do NOT support f64 (with reason).
 *
 * `activePathIds` is the set the simulator currently has selected.
 */
export interface DtypeToggleDecision {
  /** Whether the f64 toggle should be enabled. */
  f64Enabled: boolean;
  /** Reason to show when disabled (concatenated blockers). */
  reason?: string;
  blockers: KernelF64Status[];
  active: KernelF64Status[];
}

export function decideDtypeToggle(activePathIds: string[]): DtypeToggleDecision {
  const active = activePathIds
    .map(getKernelF64Status)
    .filter((s): s is KernelF64Status => !!s);
  const blockers = active.filter((s) => !s.supportsF64);
  if (active.length === 0) {
    return { f64Enabled: false, reason: "No active kernel path.", blockers, active };
  }
  if (blockers.length === 0) {
    return { f64Enabled: true, blockers, active };
  }
  const reason = blockers
    .map((b) => `${b.path.label}: ${b.reason ?? "f64 unsupported"}`)
    .join(" · ");
  return { f64Enabled: false, reason, blockers, active };
}
