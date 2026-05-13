import { describe, it, expect } from "vitest";
import { morton3D, mortonCodes, mortonOrder } from "./morton";
import { buildBVH, bvhQueryAABB, bvhAllPairs, type BVH } from "./bvh";
import { buildKDTree, knn, radiusQuery } from "./kdtree";
import { buildSpatialHash, broadphasePairs } from "./spatialHash";
import { rayAABB, rayQuery, pointInSolid, contactCandidates } from "./queries";
import { analyzeFabrication } from "./fabIntel";
import { bvhFromOctree } from "./index";
import { buildOctreeMesh } from "../meshing/octree";
import { buildAdjacency } from "../meshing/adjacency";

function makePrims(n: number, jitter = 0): { lo: Float32Array; hi: Float32Array } {
  const lo = new Float32Array(n * 3);
  const hi = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const x = (i % 8) / 8 + jitter * Math.sin(i);
    const y = (Math.floor(i / 8) % 8) / 8 + jitter * Math.cos(i);
    const z = (i % 4) / 4;
    lo[i * 3] = x; lo[i * 3 + 1] = y; lo[i * 3 + 2] = z;
    hi[i * 3] = x + 0.05; hi[i * 3 + 1] = y + 0.05; hi[i * 3 + 2] = z + 0.05;
  }
  return { lo, hi };
}

describe("morton", () => {
  it("preserves Z-order locality", () => {
    expect(morton3D(0, 0, 0)).toBe(0);
    const a = morton3D(0.1, 0.1, 0.1);
    const b = morton3D(0.9, 0.9, 0.9);
    expect(b).toBeGreaterThan(a);
  });
  it("orders points coherently", () => {
    const pts = new Float32Array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5]);
    const codes = mortonCodes(pts, { min: [0, 0, 0], max: [1, 1, 1] });
    const order = mortonOrder(codes);
    expect(order[0]).toBe(1);
    expect(order[2]).toBe(0);
  });
});

describe("BVH", () => {
  let bvh: BVH;
  it("builds over 64 prims", () => {
    const { lo, hi } = makePrims(64);
    bvh = buildBVH(lo, hi);
    expect(bvh.primCount).toBe(64);
    expect(bvh.nodeCount).toBeGreaterThan(1);
    expect(bvh.maxDepth).toBeGreaterThan(0);
  });
  it("AABB query returns matching prims", () => {
    const { lo, hi } = makePrims(64);
    const tree = buildBVH(lo, hi);
    const hits = bvhQueryAABB(tree, [0, 0, 0], [0.2, 0.2, 1]);
    expect(hits.length).toBeGreaterThan(0);
    for (const id of hits) {
      expect(lo[id * 3]).toBeLessThanOrEqual(0.2);
    }
  });
  it("all-pairs returns symmetric overlap set", () => {
    const { lo, hi } = makePrims(32);
    const tree = buildBVH(lo, hi);
    const pairs = bvhAllPairs(tree);
    expect(pairs.length % 2).toBe(0);
    for (let i = 0; i < pairs.length; i += 2) expect(pairs[i]).toBeLessThan(pairs[i + 1]);
  });
});

describe("KDTree", () => {
  it("knn returns k closest", () => {
    const pts = new Float32Array([0, 0, 0,  1, 0, 0,  0, 1, 0,  0, 0, 1,  2, 2, 2]);
    const t = buildKDTree(pts);
    const r = knn(t, [0.1, 0.1, 0.1], 3);
    expect(r.length).toBe(3);
    expect(r[0].idx).toBe(0);
    expect(r.map((x) => x.idx).includes(4)).toBe(false);
  });
  it("radius query respects bound", () => {
    const pts = new Float32Array([0, 0, 0,  0.5, 0, 0,  10, 0, 0]);
    const t = buildKDTree(pts);
    const ids = radiusQuery(t, [0, 0, 0], 1);
    expect(ids.sort()).toEqual([0, 1]);
  });
});

describe("spatial hash", () => {
  it("buckets prims, generates broadphase pairs", () => {
    const { lo, hi } = makePrims(64);
    const h = buildSpatialHash(lo, hi, 0.2);
    expect(h.primCount).toBe(64);
    expect(h.maxBucket).toBeGreaterThan(0);
    const pairs = broadphasePairs(h, lo, hi);
    expect(pairs.length % 2).toBe(0);
  });
});

describe("queries", () => {
  it("rayAABB enters and exits", () => {
    const r = rayAABB([-1, 0.5, 0.5], [1, 0, 0], [0, 0, 0], [1, 1, 1])!;
    expect(r.tEnter).toBeCloseTo(1, 5);
    expect(r.tExit).toBeCloseTo(2, 5);
  });
  it("rayQuery hits sorted by tEnter", () => {
    const lo = new Float32Array([0, 0, 0,  2, 0, 0,  4, 0, 0]);
    const hi = new Float32Array([1, 1, 1,  3, 1, 1,  5, 1, 1]);
    const tree = buildBVH(lo, hi);
    const hits = rayQuery(tree, [-1, 0.5, 0.5], [1, 0, 0]);
    expect(hits.map((h) => h.primId)).toEqual([0, 1, 2]);
  });
  it("pointInSolid: parity test", () => {
    const lo = new Float32Array([0, 0, 0]);
    const hi = new Float32Array([1, 1, 1]);
    const tree = buildBVH(lo, hi);
    expect(pointInSolid(tree, [0.5, 0.5, 0.5])).toBe(true);
    expect(pointInSolid(tree, [2, 0.5, 0.5])).toBe(false);
  });
  it("contactCandidates between two BVHs", () => {
    const a = buildBVH(new Float32Array([0, 0, 0, 5, 5, 5]), new Float32Array([1, 1, 1, 6, 6, 6]));
    const b = buildBVH(new Float32Array([0.5, 0.5, 0.5]),    new Float32Array([1.5, 1.5, 1.5]));
    const c = contactCandidates(a, b);
    expect(c.length).toBe(2);
    expect(c[0]).toBe(0);
  });
});

describe("fabIntel", () => {
  it("classifies leaves and reports notes", () => {
    const mesh = buildOctreeMesh(
      { min: [0, 0, 0], max: [1, 1, 1] },
      [{ kind: "fillet", center: [0.5, 0.5, 0.5], radius: 0.3, weight: 1 }],
      { minDepth: 3, maxDepth: 3, refineThreshold: 1 },
    );
    const adj = buildAdjacency(mesh);
    const r = analyzeFabrication(mesh, adj);
    expect(r.classification.length).toBe(mesh.leaves.length);
    expect(r.notes.length).toBeGreaterThan(0);
    expect(r.ms).toBeGreaterThanOrEqual(0);
  });
  it("bvhFromOctree builds a queryable BVH", () => {
    const mesh = buildOctreeMesh(
      { min: [0, 0, 0], max: [1, 1, 1] },
      [],
      { minDepth: 2, maxDepth: 2, refineThreshold: 1 },
    );
    const tree = bvhFromOctree(mesh);
    expect(tree.primCount).toBe(mesh.leaves.length);
    const hits = bvhQueryAABB(tree, [0, 0, 0], [0.4, 0.4, 0.4]);
    expect(hits.length).toBeGreaterThan(0);
  });
});
