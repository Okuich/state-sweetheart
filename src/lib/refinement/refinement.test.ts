import { describe, expect, it } from "vitest";
import { buildOctreeMesh, type RefinementSeed } from "../meshing/octree";
import { partitionMesh } from "../meshing/partition";
import { buildAdjacency } from "../meshing/adjacency";
import {
  computeLeafError,
  planRefinement,
  applyRefinement,
  synthesizeFields,
  runAdaptivePass,
  RefinementPriorStore,
} from "./index";

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };

const seeds: RefinementSeed[] = [
  { kind: "sharp", center: [0.5, 0.5, 0], radius: 0.2, weight: 1 },
  { kind: "hotspot", center: [-0.4, 0, 0.3], radius: 0.25, weight: 0.8 },
];

describe("refinement", () => {
  it("computes leaf error and metric tensor", () => {
    const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
    const fields = synthesizeFields(mesh, seeds, 0);
    const err = computeLeafError(mesh, fields);
    expect(err.combined.length).toBe(mesh.leaves.length);
    expect(err.metric.length).toBe(mesh.leaves.length * 3);
    const max = Math.max(...err.combined);
    expect(max).toBeGreaterThan(0);
  });

  it("plans split + coarsen with hot-leaf cap", () => {
    const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
    const fields = synthesizeFields(mesh, seeds, 0);
    const err = computeLeafError(mesh, fields);
    const plan = planRefinement(mesh, err, { splitThreshold: 0.3, maxNewLeaves: 16 });
    expect(plan.splitLeaves.length).toBeLessThanOrEqual(16);
    expect(plan.derivedSeeds.length).toBe(plan.splitLeaves.length);
  });

  it("apply produces refined mesh with >= base leaves", () => {
    const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
    const fields = synthesizeFields(mesh, seeds, 0);
    const err = computeLeafError(mesh, fields);
    const plan = planRefinement(mesh, err, { splitThreshold: 0.3, extraDepth: 1 });
    const pass = applyRefinement(BBOX, seeds, mesh, plan, { minDepth: 2 });
    expect(pass.newMesh.leaves.length).toBeGreaterThanOrEqual(mesh.leaves.length);
  });

  it("end-to-end adaptive pass returns repartition hint", () => {
    const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
    const adj = buildAdjacency(mesh);
    const part = partitionMesh(mesh, adj, 4);
    const result = runAdaptivePass({
      bbox: BBOX,
      baseSeeds: seeds,
      baseMesh: mesh,
      basePartition: part,
      step: 0,
      options: { splitThreshold: 0.3, extraDepth: 1, maxNewLeaves: 50 },
      ingestPriors: false,
    });
    expect(result.pass.newMesh.leaves.length).toBeGreaterThan(0);
    expect(result.repartition.newSizes.length).toBe(4);
    expect(result.haloDelta.perPartition.length).toBe(4);
  });

  it("prior store dedupes by quantized cell + tag", () => {
    const mesh = buildOctreeMesh(BBOX, seeds, { maxDepth: 4, minDepth: 2 });
    const fields = synthesizeFields(mesh, seeds, 0);
    const err = computeLeafError(mesh, fields);
    const plan = planRefinement(mesh, err, { splitThreshold: 0.3 });
    const store = new RefinementPriorStore(0.1);
    const a = store.ingest(mesh, err, plan);
    const b = store.ingest(mesh, err, plan); // re-ingest should not add
    expect(a).toBeGreaterThan(0);
    expect(b).toBe(0);
    expect(store.topK(3).length).toBeLessThanOrEqual(3);
  });
});
