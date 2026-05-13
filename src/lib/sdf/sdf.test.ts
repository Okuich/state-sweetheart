import { describe, it, expect } from "vitest";
import {
  buildSparseSDF, distance, gradient, nearestSurface, sphereCollide,
  partitionSDF, buildEmbedding,
  type SDFPrim, type AABB,
} from "./index";

const bbox: AABB = { min: [-1, -1, -1], max: [1, 1, 1] };

describe("sdf/sparseField", () => {
  it("collapses to a small sparse footprint vs dense", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.1, bandWidth: 3 });
    expect(sdf.stats.allocatedBricks).toBeGreaterThan(0);
    expect(sdf.stats.allocatedBricks).toBeLessThan(sdf.stats.rootBrickCount);
    expect(sdf.stats.sparsity).toBeGreaterThan(0);
  });

  it("distance is approximately correct for a sphere", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.05, bandWidth: 4 });
    const d = distance(sdf, [0.7, 0, 0]);
    expect(d).toBeGreaterThan(0.1);
    expect(d).toBeLessThan(0.25);
  });

  it("gradient points outward at the surface", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.05 });
    const g = gradient(sdf, [0.5, 0, 0]);
    expect(g[0]).toBeGreaterThan(0);
    expect(Math.abs(g[1])).toBeLessThan(0.5);
  });
});

describe("sdf/queries", () => {
  it("nearestSurface projects onto the sphere", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.05 });
    const r = nearestSurface(sdf, [0.8, 0, 0]);
    expect(Math.abs(Math.hypot(r.point[0], r.point[1], r.point[2]) - 0.5)).toBeLessThan(0.1);
  });

  it("sphere collision detects penetration and returns separation normal", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.05 });
    const c = sphereCollide(sdf, [0.4, 0, 0], 0.2);
    expect(c.hit).toBe(true);
    expect(c.depth).toBeGreaterThan(0);
    expect(c.normal[0]).toBeGreaterThan(0);
  });
});

describe("sdf/partition", () => {
  it("partitions bricks across slabs and produces halos", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.6 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.1 });
    const plan = partitionSDF(sdf, 4, 1);
    expect(plan.partitionCount).toBe(4);
    expect(plan.resident.reduce((a, b) => a + b.length, 0)).toBe(sdf.bricks.length);
    expect(plan.totalHalo).toBeGreaterThanOrEqual(0);
  });
});

describe("sdf/embeddings", () => {
  it("produces a 32-d geometry embedding and manufacturability score", () => {
    const prims: SDFPrim[] = [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }];
    const sdf = buildSparseSDF(bbox, prims, { voxelSize: 0.1 });
    const emb = buildEmbedding(sdf);
    expect(emb.geometry.length).toBe(32);
    expect(emb.manufacturability.score).toBeGreaterThanOrEqual(0);
    expect(emb.manufacturability.score).toBeLessThanOrEqual(1);
    expect(emb.topology.volumeFraction).toBeGreaterThan(0);
  });
});
