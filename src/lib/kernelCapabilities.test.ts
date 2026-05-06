import { describe, it, expect, beforeEach } from "vitest";
import {
  listKernelPaths,
  getKernelF64Status,
  setKernelF64Support,
  clearKernelOverrides,
  decideDtypeToggle,
} from "./kernelCapabilities";

describe("kernelCapabilities", () => {
  beforeEach(() => clearKernelOverrides());

  it("lists known kernel paths", () => {
    const paths = listKernelPaths();
    const ids = paths.map((p) => p.path.id);
    expect(ids).toContain("cpu-reference");
    expect(ids).toContain("webgpu-spatial-hash");
  });

  it("CPU reference supports f64", () => {
    const s = getKernelF64Status("cpu-reference");
    expect(s?.supportsF64).toBe(true);
  });

  it("WebGPU paths never support f64", () => {
    const s = getKernelF64Status("webgpu-lbvh");
    expect(s?.supportsF64).toBe(false);
    expect(s?.reason).toMatch(/WebGPU|adapter/i);
  });

  it("override can force support on/off", () => {
    setKernelF64Support("webgpu-lbvh", true);
    expect(getKernelF64Status("webgpu-lbvh")?.supportsF64).toBe(true);
    setKernelF64Support("cpu-reference", false);
    expect(getKernelF64Status("cpu-reference")?.supportsF64).toBe(false);
  });

  it("decideDtypeToggle disables when any active path lacks f64", () => {
    const d = decideDtypeToggle(["cpu-reference", "webgpu-spatial-hash"]);
    expect(d.f64Enabled).toBe(false);
    expect(d.blockers.map((b) => b.path.id)).toContain("webgpu-spatial-hash");
    expect(d.reason).toMatch(/spatial-hash/);
  });

  it("decideDtypeToggle enables when all active paths support f64", () => {
    const d = decideDtypeToggle(["cpu-reference", "wasm-simd"]);
    expect(d.f64Enabled).toBe(true);
    expect(d.blockers).toHaveLength(0);
  });

  it("decideDtypeToggle handles empty active set", () => {
    const d = decideDtypeToggle([]);
    expect(d.f64Enabled).toBe(false);
    expect(d.reason).toMatch(/No active/);
  });
});
