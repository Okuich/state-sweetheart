import { describe, expect, it } from "vitest";
import { buildOctreeMesh, type RefinementSeed } from "../meshing/octree";
import { buildAdjacency } from "../meshing/adjacency";
import {
  planDistributed,
  runAlgorithm,
  buildHaloPlan,
  planRebalance,
  buildCheckpoint,
  verifyReplay,
  applyMigration,
  type PartitionAlgorithm,
} from "./index";

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };
const seeds: RefinementSeed[] = [
  { kind: "sharp", center: [0.4, 0.4, 0], radius: 0.2, weight: 1 },
  { kind: "hotspot", center: [-0.4, 0, 0.3], radius: 0.25, weight: 0.8 },
];

function makeMesh() {
  const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
  const adj = buildAdjacency(mesh);
  return { mesh, adj };
}

describe("distpart algorithms", () => {
  const { mesh, adj } = makeMesh();
  const T = mesh.tets.length / 4;
  const P = 4;

  it.each<PartitionAlgorithm>(["regionGrow", "morton", "spectral", "kway"])(
    "%s produces complete, balanced assignment",
    (algo) => {
      const r = runAlgorithm(algo, mesh, adj, P);
      expect(r.tetPart.length).toBe(T);
      const sizes = new Uint32Array(P);
      for (let t = 0; t < T; t++) {
        expect(r.tetPart[t]).toBeLessThan(P);
        sizes[r.tetPart[t]]++;
      }
      const sum = Array.from(sizes).reduce((a, b) => a + b, 0);
      expect(sum).toBe(T);
      expect(r.stats.imbalance).toBeLessThan(1.5);
    },
  );

  it("kway refinement does not increase edge cut vs seed", () => {
    const seed = runAlgorithm("regionGrow", mesh, adj, P);
    const refined = runAlgorithm("kway", mesh, adj, P);
    expect(refined.stats.edgeCut).toBeLessThanOrEqual(seed.stats.edgeCut + 1);
  });
});

describe("comm graph + halos", () => {
  it("halo plan produces valid send/recv lists and rounds", () => {
    const { mesh, adj } = makeMesh();
    const P = 4;
    const r = runAlgorithm("regionGrow", mesh, adj, P);
    const halo = buildHaloPlan(r.tetPart, adj, P);
    expect(halo.haloRecv.length).toBe(P);
    expect(halo.commMatrix.length).toBe(P * P);
    // Diagonal must be zero.
    for (let p = 0; p < P; p++) expect(halo.commMatrix[p * P + p]).toBe(0);
    // Each round visits each rank at most once.
    for (const round of halo.rounds) {
      const seen = new Set<number>();
      for (const { src, dst } of round) {
        expect(seen.has(src)).toBe(false);
        expect(seen.has(dst)).toBe(false);
        seen.add(src); seen.add(dst);
      }
    }
  });
});

describe("rebalance", () => {
  it("reduces makespan under skewed weights", () => {
    const { mesh, adj } = makeMesh();
    const P = 4;
    const seed = runAlgorithm("regionGrow", mesh, adj, P);
    const T = seed.tetPart.length;
    const weights = new Float32Array(T).fill(1);
    // Pile 5x weight on partition 0's tets.
    for (let t = 0; t < T; t++) if (seed.tetPart[t] === 0) weights[t] = 5;
    const plan = planRebalance({
      tetPart: seed.tetPart, weights, partitionCount: P, adj, tolerance: 1.1,
    });
    expect(plan.afterMakespan).toBeLessThanOrEqual(plan.beforeMakespan);
    if (plan.moves.length) {
      const newPart = applyMigration(seed.tetPart, plan);
      expect(newPart.length).toBe(T);
    }
  });
});

describe("checkpoint + replay", () => {
  it("identical inputs yield identical digest", () => {
    const { mesh, adj } = makeMesh();
    const P = 4;
    const r = runAlgorithm("regionGrow", mesh, adj, P);
    const a = buildCheckpoint(r.tetPart, 42, P);
    const b = buildCheckpoint(r.tetPart, 42, P);
    expect(verifyReplay(a, b).ok).toBe(true);
  });

  it("different step produces different root digest", () => {
    const { mesh, adj } = makeMesh();
    const P = 4;
    const r = runAlgorithm("regionGrow", mesh, adj, P);
    const a = buildCheckpoint(r.tetPart, 1, P);
    const b = buildCheckpoint(r.tetPart, 2, P);
    expect(a.rootDigest).not.toBe(b.rootDigest);
  });
});

describe("planDistributed end-to-end", () => {
  it("produces halo + latency + checkpoint", () => {
    const { mesh, adj } = makeMesh();
    const r = planDistributed({
      mesh, adj, partitionCount: 4, algorithm: "kway",
      rebalance: true, checkpointStep: 7,
    });
    expect(r.halo.partitionCount).toBe(4);
    expect(r.latency.totalUs).toBeGreaterThanOrEqual(0);
    expect(r.checkpoint?.snapshots.length).toBe(4);
  });
});
