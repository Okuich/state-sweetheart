import { describe, expect, it } from "vitest";
import { buildOctreeMesh } from "../meshing/octree";
import { buildTopologyGraph } from "./graph";
import { partitionTopology } from "./partition";
import { kHopCpu, computeHalosCpu, KHOP_INF } from "./gpuTraversal";

const BBOX = { min: [-1, -1, -1] as const, max: [1, 1, 1] as const };

function buildGraph() {
  const mesh = buildOctreeMesh(BBOX, [{ kind: "overhang" as const, center: [0, 0.5, 0], radius: 0.4, weight: 1 }],
    { minDepth: 2, maxDepth: 4, refineThreshold: 0.3 });
  return buildTopologyGraph(mesh);
}

describe("kHopCpu", () => {
  it("seeds have distance 0 and label preserved", () => {
    const g = buildGraph();
    const seeds = new Uint32Array(g.nodes.length);
    seeds[0] = 1; seeds[5] = 2;
    const r = kHopCpu(g, seeds, 4);
    expect(r.dist[0]).toBe(0);
    expect(r.dist[5]).toBe(0);
    expect(r.label[0]).toBe(1);
    expect(r.label[5]).toBe(2);
  });

  it("k=1 only labels seeds + direct neighbors", () => {
    const g = buildGraph();
    const seeds = new Uint32Array(g.nodes.length);
    seeds[0] = 1;
    const r = kHopCpu(g, seeds, 1);
    const nbrs = new Set<number>();
    for (let p = g.neighborOffsets[0]; p < g.neighborOffsets[1]; p++) nbrs.add(g.neighborIdx[p]);
    for (let i = 0; i < g.nodes.length; i++) {
      if (i === 0) continue;
      if (nbrs.has(i)) expect(r.dist[i]).toBe(1);
      else expect(r.dist[i]).toBeGreaterThan(1);
    }
  });

  it("dist agrees with BFS reference", () => {
    const g = buildGraph();
    const seeds = new Uint32Array(g.nodes.length);
    seeds[3] = 1;
    const r = kHopCpu(g, seeds, 6);
    // Reference BFS from node 3
    const N = g.nodes.length;
    const ref = new Uint8Array(N).fill(KHOP_INF);
    ref[3] = 0;
    const queue = [3];
    while (queue.length) {
      const v = queue.shift()!;
      for (let p = g.neighborOffsets[v]; p < g.neighborOffsets[v + 1]; p++) {
        const n = g.neighborIdx[p];
        if (ref[n] === KHOP_INF && ref[v] < 6) { ref[n] = ref[v] + 1; queue.push(n); }
      }
    }
    for (let i = 0; i < N; i++) expect(r.dist[i]).toBe(ref[i]);
  });
});

describe("computeHalosCpu", () => {
  it("matches partitionTopology halos", () => {
    const g = buildGraph();
    const plan = partitionTopology(g, 4);
    const owners = new Uint32Array(plan.owners);
    const r = computeHalosCpu(g, owners, plan.partitionCount);
    for (let p = 0; p < plan.partitionCount; p++) {
      const ref = new Set(plan.halos[p]);
      const got = new Set(r.halos[p]);
      expect(got.size).toBe(ref.size);
      for (const x of ref) expect(got.has(x)).toBe(true);
    }
  });

  it("mask bit p set ⇒ at least one neighbor in partition p", () => {
    const g = buildGraph();
    const plan = partitionTopology(g, 4);
    const owners = new Uint32Array(plan.owners);
    const r = computeHalosCpu(g, owners, plan.partitionCount);
    for (let i = 0; i < g.nodes.length; i++) {
      const m = r.mask[i];
      for (let p = 0; p < plan.partitionCount; p++) {
        if ((m >>> p) & 1) {
          let found = false;
          for (let q = g.neighborOffsets[i]; q < g.neighborOffsets[i + 1]; q++) {
            if (owners[g.neighborIdx[q]] === p) { found = true; break; }
          }
          expect(found).toBe(true);
        }
      }
    }
  });
});
