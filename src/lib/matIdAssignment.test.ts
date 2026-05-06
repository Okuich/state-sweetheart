import { describe, it, expect } from "vitest";
import {
  buildCubeMesh, DEFAULT_PALETTE, histogram, paintByBoxRegion,
  paintByIndices, paintBySphereRegion, restore, snapshot, tetCentroid,
} from "./matIdAssignment";

describe("buildCubeMesh", () => {
  it("creates 6·NX·NY·NZ tets", () => {
    const m = buildCubeMesh(2, 2, 2);
    expect(m.tets.length).toBe(48);
    expect(m.matId.length).toBe(48);
    expect(m.matId.every((v) => v === 0)).toBe(true);
  });
  it("centroids lie inside the unit cube", () => {
    const m = buildCubeMesh(3, 3, 3);
    for (let i = 0; i < m.tets.length; i++) {
      const [x, y, z] = tetCentroid(m, i);
      expect(x).toBeGreaterThanOrEqual(0); expect(x).toBeLessThanOrEqual(1);
      expect(y).toBeGreaterThanOrEqual(0); expect(y).toBeLessThanOrEqual(1);
      expect(z).toBeGreaterThanOrEqual(0); expect(z).toBeLessThanOrEqual(1);
    }
  });
});

describe("painters", () => {
  it("paintByBoxRegion only touches tets with centroid in the box", () => {
    const m = buildCubeMesh(4, 4, 4);
    const n = paintByBoxRegion(m, 1, [0, 0, 0], [0.5, 1, 1]);
    expect(n).toBeGreaterThan(0);
    for (let i = 0; i < m.tets.length; i++) {
      const [x] = tetCentroid(m, i);
      expect(m.matId[i]).toBe(x <= 0.5 ? 1 : 0);
    }
  });

  it("paintBySphereRegion respects the radius", () => {
    const m = buildCubeMesh(4, 4, 4);
    paintBySphereRegion(m, 2, [0.5, 0.5, 0.5], 0.2);
    for (let i = 0; i < m.tets.length; i++) {
      const [x, y, z] = tetCentroid(m, i);
      const d2 = (x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2;
      expect(m.matId[i]).toBe(d2 <= 0.04 ? 2 : 0);
    }
  });

  it("paintByIndices ignores OOR indices", () => {
    const m = buildCubeMesh(2, 2, 2);
    const n = paintByIndices(m, 3, [0, 5, 999, -1, 7]);
    expect(n).toBe(3);
    expect(m.matId[0]).toBe(3);
    expect(m.matId[5]).toBe(3);
    expect(m.matId[7]).toBe(3);
  });
});

describe("histogram", () => {
  it("counts every palette entry, defaults to zero", () => {
    const m = buildCubeMesh(2, 2, 2);
    paintByBoxRegion(m, 1, [0, 0, 0], [0.5, 1, 1]);
    const h = histogram(m, DEFAULT_PALETTE);
    expect(h.size).toBe(DEFAULT_PALETTE.length);
    let total = 0;
    h.forEach((v) => { total += v; });
    expect(total).toBe(m.tets.length);
  });
});

describe("snapshot / restore", () => {
  it("round-trips matId arrays", () => {
    const m = buildCubeMesh(2, 2, 2);
    paintByBoxRegion(m, 4, [0, 0, 0], [1, 1, 1]);
    const snap = snapshot(m);
    paintByBoxRegion(m, 0, [0, 0, 0], [1, 1, 1]);
    restore(m, snap);
    expect(m.matId.every((v) => v === 4)).toBe(true);
  });
  it("rejects mismatched snapshot length", () => {
    const m = buildCubeMesh(1, 1, 1);
    expect(() => restore(m, new Uint32Array(99))).toThrow();
  });
});
