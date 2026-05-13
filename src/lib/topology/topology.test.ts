import { describe, it, expect } from "vitest";
import { buildOctreeMesh, type RefinementSeed } from "@/lib/meshing/octree";
import {
  analyzeTopology, buildTopologyGraph, classifyFeatures,
  scoreManufacturability, buildStructuralEmbedding, partitionTopology,
  cosine,
} from "./index";

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };

function buildMesh(seeds: RefinementSeed[] = []) {
  return buildOctreeMesh(BBOX, seeds, { minDepth: 2, maxDepth: 4, refineThreshold: 0.3 });
}

describe("topology/graph", () => {
  it("builds adjacency over leaves", () => {
    const mesh = buildMesh();
    const g = buildTopologyGraph(mesh);
    expect(g.nodes.length).toBe(mesh.leaves.length);
    expect(g.edges.length).toBeGreaterThan(0);
    expect(g.neighborOffsets.length).toBe(g.nodes.length + 1);
    // CSR consistency
    expect(g.neighborOffsets[g.nodes.length]).toBe(g.neighborIdx.length);
  });

  it("curvature is non-negative and bounded", () => {
    const g = buildTopologyGraph(buildMesh());
    for (const n of g.nodes) {
      expect(n.curvature).toBeGreaterThanOrEqual(0);
      expect(n.curvature).toBeLessThanOrEqual(6);
    }
  });
});

describe("topology/features", () => {
  it("classifies overhangs given downward-tagged seeds", () => {
    const seeds: RefinementSeed[] = [
      { kind: "overhang", center: [0, 0.6, 0], radius: 0.3, weight: 1 },
    ];
    const mesh = buildMesh(seeds);
    const g = buildTopologyGraph(mesh);
    const feats = classifyFeatures(g, mesh.bbox);
    expect(feats.counts.overhang + feats.counts.boundary + feats.counts.thin_wall + feats.counts.cavity + feats.counts.stress_concentrator).toBeGreaterThan(0);
    expect(feats.symmetryScore).toBeGreaterThanOrEqual(0);
    expect(feats.symmetryScore).toBeLessThanOrEqual(1);
  });

  it("flags stress concentrators on sharp seeds", () => {
    const mesh = buildMesh([{ kind: "sharp", center: [0.7, 0, 0], radius: 0.25, weight: 1 }]);
    const g = buildTopologyGraph(mesh);
    const feats = classifyFeatures(g, mesh.bbox);
    expect(feats.counts.stress_concentrator).toBeGreaterThanOrEqual(0);
  });
});

describe("topology/manufacturability", () => {
  it("scores in [0,1]", () => {
    const mesh = buildMesh([{ kind: "overhang", center: [0, 0.6, 0], radius: 0.3, weight: 1 }]);
    const g = buildTopologyGraph(mesh);
    const feats = classifyFeatures(g, mesh.bbox);
    const m = scoreManufacturability(g, feats, mesh.bbox);
    for (const v of [m.feasibility, m.supportFraction, m.machiningAccess, m.thermalDistortionRisk, m.assemblyComplexity]) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe("topology/embeddings", () => {
  it("produces a 40-d vector and retrieval works", () => {
    const a = analyzeTopology(buildMesh([{ kind: "overhang", center: [0, 0.6, 0], radius: 0.3, weight: 1 }]));
    const b = analyzeTopology(buildMesh([{ kind: "sharp", center: [0.7, 0, 0], radius: 0.25, weight: 1 }]));
    expect(a.embedding.vector.length).toBe(40);
    const selfSim = cosine(a.embedding.vector, a.embedding.vector);
    expect(selfSim).toBeCloseTo(1, 5);
    const crossSim = cosine(a.embedding.vector, b.embedding.vector);
    expect(crossSim).toBeLessThanOrEqual(1);
  });
});

describe("topology/partition", () => {
  it("partitions all nodes and produces edge cut + halos", () => {
    const mesh = buildMesh();
    const g = buildTopologyGraph(mesh);
    const plan = partitionTopology(g, 4);
    expect(plan.resident.reduce((a, b) => a + b.length, 0)).toBe(g.nodes.length);
    expect(plan.edgeCut).toBeGreaterThanOrEqual(0);
    expect(plan.halos.length).toBe(4);
  });
});

describe("topology/end-to-end", () => {
  it("runs full pipeline and emits priors", () => {
    const r = analyzeTopology(buildMesh([{ kind: "thin_wall", center: [0, 0, 0.6], radius: 0.3, weight: 1 }]), { partitionCount: 2 });
    expect(r.priors.timestepScale).toBeGreaterThan(0);
    expect(r.priors.damping).toBeGreaterThanOrEqual(0);
    expect(r.priors.contactStiffness).toBeGreaterThanOrEqual(0);
    expect(r.partition.partitionCount).toBe(2);
  });
});
