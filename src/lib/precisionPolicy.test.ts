import { describe, it, expect } from "vitest";
import { PrecisionPolicy } from "./precisionPolicy";

describe("PrecisionPolicy", () => {
  it("passes Float32Array through unchanged", () => {
    const p = new PrecisionPolicy({ hostIsF32Only: true });
    const a = new Float32Array([1, 2, 3]);
    const b = p.toGpuFloat32(a, "v");
    expect(b).toBe(a); // same reference
    expect(p.snapshot().conversions).toBe(0);
  });

  it("auto-downgrade truncates Float64 and tracks loss", () => {
    const p = new PrecisionPolicy({ mode: "auto-downgrade", hostIsF32Only: true });
    // 1e-12 is well below f32 ULP at value ~1, so loss is significant.
    const src = new Float64Array([1 + 1e-12, 2.5, 1e10]);
    const out = p.toGpuFloat32(src, "vertices");
    expect(out).toBeInstanceOf(Float32Array);
    expect(out[1]).toBe(2.5);
    const s = p.snapshot();
    expect(s.conversions).toBe(1);
    expect(s.elementsConverted).toBe(3);
    expect(s.worstRelLoss).toBeGreaterThan(0);
    expect(s.events[0].label).toBe("vertices");
  });

  it("strict mode throws when downgrade would lose precision", () => {
    const p = new PrecisionPolicy({ mode: "strict", hostIsF32Only: true });
    const src = new Float64Array([1 + 1e-10]);
    expect(() => p.toGpuFloat32(src, "Fp")).toThrow(/strict/);
  });

  it("strict mode allows lossless f64 → f32 (small ints)", () => {
    const p = new PrecisionPolicy({ mode: "strict", hostIsF32Only: true });
    const src = new Float64Array([1, 2, 3]);
    const out = p.toGpuFloat32(src, "ok");
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });

  it("flags overflow above f32 max", () => {
    const p = new PrecisionPolicy({ mode: "auto-downgrade", hostIsF32Only: true });
    const src = new Float64Array([1e40]);
    p.toGpuFloat32(src, "huge");
    const s = p.snapshot();
    expect(s.anyOverflow).toBe(true);
    expect(s.events[0].overflow).toBe(true);
  });

  it("shouldShowBanner only true on f32-only host with real loss", () => {
    const lossy = new Float64Array([1 + 1e-10]);

    const cpuLike = new PrecisionPolicy({ mode: "auto-downgrade", hostIsF32Only: false });
    cpuLike.toGpuFloat32(lossy, "v");
    expect(cpuLike.shouldShowBanner()).toBe(false);

    const gpuLike = new PrecisionPolicy({ mode: "auto-downgrade", hostIsF32Only: true });
    expect(gpuLike.shouldShowBanner()).toBe(false); // no events yet
    gpuLike.toGpuFloat32(lossy, "v");
    expect(gpuLike.shouldShowBanner()).toBe(true);
  });

  it("history is bounded by historySize", () => {
    const p = new PrecisionPolicy({ historySize: 3, hostIsF32Only: true });
    for (let k = 0; k < 5; k++) p.toGpuFloat32(new Float64Array([1 + 1e-10]), `e${k}`);
    const s = p.snapshot();
    expect(s.events.length).toBe(3);
    expect(s.events.map((e) => e.label)).toEqual(["e2", "e3", "e4"]);
  });

  it("reset clears stats but keeps mode/host", () => {
    const p = new PrecisionPolicy({ hostIsF32Only: true });
    p.toGpuFloat32(new Float64Array([1 + 1e-10]), "v");
    p.reset();
    const s = p.snapshot();
    expect(s.conversions).toBe(0);
    expect(s.events.length).toBe(0);
    expect(s.worstRelLoss).toBe(0);
    expect(s.hostIsF32Only).toBe(true);
  });
});
