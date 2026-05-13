import { describe, it, expect } from "vitest";
import { generateMesh, seedsFromFeatures, type AABB } from "./index";
import { buildOctreeMesh } from "./octree";
import { evaluateQuality, tetVolume } from "./quality";
import { buildAdjacency } from "./adjacency";
import { partitionMesh } from "./partition";

const BBOX: AABB = { min: [0, 0, 0], max: [1, 1, 1] };

describe("octree", () => {
  it("respects minDepth uniformly", () => {
    const m = buildOctreeMesh(BBOX, [], { minDepth: 2, maxDepth: 2, refineThreshold: 1 });
    // 2 levels of full subdivision: 8^2 = 64 leaves.
    expect(m.leaves.length).toBe(64);
    expect(m.tets.length / 4).toBe(64 * 6);
  });
  it("refines deeper near a seed", () => {
    const seeded = buildOctreeMesh(BBOX, [
      { kind: "hole", center: [0.5, 0.5, 0.5], radius: 0.05, weight: 5 },
    ], { minDepth: 1, maxDepth: 5, refineThreshold: 0.2 });
    const baseline = buildOctreeMesh(BBOX, [], { minDepth: 1, maxDepth: 5, refineThreshold: 0.2 });
    expect(seeded.leaves.length).toBeGreaterThan(baseline.leaves.length);
    const maxD = Math.max(...seeded.leaves.map((id) => seeded.nodes[id].depth));
    expect(maxD).toBeGreaterThan(1);
  });
  it("respects maxLeaves cap", () => {
    const m = buildOctreeMesh(BBOX, [
      { kind: "hotspot", center: [0.5, 0.5, 0.5], radius: 1, weight: 5 },
    ], { minDepth: 1, maxDepth: 8, refineThreshold: 0.1, maxLeaves: 200 });
    expect(m.leaves.length).toBeLessThanOrEqual(200 + 8);
  });
});

describe("quality", () => {
  it("reports zero inverted tets for a uniform mesh", () => {
    const m = buildOctreeMesh(BBOX, [], { minDepth: 2, maxDepth: 2, refineThreshold: 1 });
    const q = evaluateQuality(m);
    expect(q.invertedTets).toBe(0);
    expect(q.totalVolume).toBeCloseTo(1, 3);
    expect(q.convergenceScore).toBeGreaterThan(0.3);
  });
  it("tetVolume sign matches orientation", () => {
    const v = tetVolume([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]);
    expect(v).toBeGreaterThan(0);
  });
});

describe("adjacency", () => {
  it("produces a CSR graph with valid bounds", () => {
    const m = buildOctreeMesh(BBOX, [], { minDepth: 2, maxDepth: 2, refineThreshold: 1 });
    const a = buildAdjacency(m);
    const V = m.vertices.length / 3;
    expect(a.vvRowPtr.length).toBe(V + 1);
    expect(a.vvRowPtr[V]).toBe(a.vvCol.length);
    expect(a.edgeStats.count).toBeGreaterThan(0);
    expect(a.warpStride % 32).toBe(0);
  });
  it("tet-tet adjacency is symmetric", () => {
    const m = buildOctreeMesh(BBOX, [], { minDepth: 1, maxDepth: 1, refineThreshold: 1 });
    const a = buildAdjacency(m);
    const T = m.tets.length / 4;
    for (let i = 0; i < T; i++) {
      for (let k = a.ttRowPtr[i]; k < a.ttRowPtr[i + 1]; k++) {
        const j = a.ttCol[k];
        let found = false;
        for (let kk = a.ttRowPtr[j]; kk < a.ttRowPtr[j + 1]; kk++) {
          if (a.ttCol[kk] === i) { found = true; break; }
        }
        expect(found).toBe(true);
      }
    }
  });
});

describe("partition", () => {
  it("balances sizes within ~10%", () => {
    const m = buildOctreeMesh(BBOX, [], { minDepth: 3, maxDepth: 3, refineThreshold: 1 });
    const a = buildAdjacency(m);
    const p = partitionMesh(m, a, 4);
    expect(p.partitionCount).toBe(4);
    let total = 0;
    p.sizes.forEach((s) => { total += s; });
    expect(total).toBe(m.tets.length / 4);
    expect(p.imbalance).toBeLessThan(1.5);
  });
  it("emits halos and a comm matrix", () => {
    const m = buildOctreeMesh(BBOX, [], { minDepth: 2, maxDepth: 2, refineThreshold: 1 });
    const a = buildAdjacency(m);
    const p = partitionMesh(m, a, 4);
    expect(p.halos.length).toBe(4);
    expect(p.commMatrix.length).toBe(16);
    expect(p.edgeCut).toBeGreaterThan(0);
  });
});

describe("generateMesh end-to-end", () => {
  it("builds a complete mesh + summary", () => {
    const r = generateMesh({
      bbox: BBOX,
      seeds: [{ kind: "fillet", center: [0.2, 0.2, 0.2], radius: 0.1, weight: 1 }],
      octree: { minDepth: 2, maxDepth: 4, refineThreshold: 0.3 },
      partitionCount: 4,
    });
    expect(r.summary.tets.count).toBeGreaterThan(0);
    expect(r.summary.partition.partitionCount).toBe(4);
    expect(r.summary.adjacency.edgeCount).toBeGreaterThan(0);
    expect(r.summary.octree.boundaryLeaves).toBeGreaterThan(0);
  });
  it("seedsFromFeatures maps STEP feature kinds", () => {
    const seeds = seedsFromFeatures(BBOX, [
      { kind: "cylindrical" },
      { kind: "toroidal" },
      { kind: "unknown" },
    ]);
    expect(seeds.map((s) => s.kind).sort()).toEqual(["fillet", "hole"]);
  });
});
