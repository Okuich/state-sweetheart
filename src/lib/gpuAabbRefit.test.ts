import { describe, it, expect } from "vitest";
import {
  buildRefitTree,
  initAabbRefit,
  leafBoxFromScene,
  markAllDirty,
  PrimKind,
  refitStepSync,
  type DeformableScene,
} from "./gpuAabbRefit";

/** Helper: build a small mixed scene with `nTris` cloth triangles and one
 *  particle shape of `nParts` particles. */
function makeScene(nTris: number, nParts: number): DeformableScene {
  const V = nTris * 3 + nParts;
  const P = nTris + (nParts > 0 ? 1 : 0);
  const vertices = new Float32Array(V * 2);
  const prims    = new Uint32Array(P * 4);
  const radii    = new Float32Array(P);
  // Lay cloth triangles on a grid.
  for (let t = 0; t < nTris; t++) {
    const x = t * 2.0;
    const v0 = t * 3, v1 = t * 3 + 1, v2 = t * 3 + 2;
    vertices.set([x, 0, x + 1, 0, x + 0.5, 1], v0 * 2);
    prims.set([PrimKind.ClothTri, v0, v1, v2], t * 4);
    radii[t] = 0.05;
  }
  // Particle shape: line of points.
  if (nParts > 0) {
    const start = nTris * 3;
    for (let k = 0; k < nParts; k++) {
      vertices[(start + k) * 2 + 0] = k * 0.5;
      vertices[(start + k) * 2 + 1] = -2;
    }
    prims.set([PrimKind.ParticleShape, start, nParts, 0], nTris * 4);
    radii[nTris] = 0.1;
  }
  return { V, P, vertices, prims, radii };
}

describe("buildRefitTree", () => {
  it("builds a tree with P leaves", () => {
    const scene = makeScene(8, 4);
    const tree = buildRefitTree(scene);
    let leaves = 0;
    for (let i = 0; i < tree.N; i++) if (tree.leafPrim[i] >= 0) leaves++;
    expect(leaves).toBe(scene.P);
  });

  it("levels[0] are the leaves and root is alone in the last level", () => {
    const scene = makeScene(7, 3);
    const tree = buildRefitTree(scene);
    for (const id of tree.levels[0]) expect(tree.leafPrim[id]).toBeGreaterThanOrEqual(0);
    expect(tree.levels[tree.levels.length - 1].length).toBe(1);
    expect(tree.parent[tree.levels[tree.levels.length - 1][0]]).toBe(-1);
  });

  it("handles P=1 (root is a leaf)", () => {
    const scene = makeScene(1, 0);
    const tree = buildRefitTree(scene);
    expect(tree.N).toBe(1);
    expect(tree.leafPrim[0]).toBe(0);
  });

  it("handles P=0", () => {
    const empty: DeformableScene = { V: 0, P: 0, vertices: new Float32Array(), prims: new Uint32Array(), radii: new Float32Array() };
    const tree = buildRefitTree(empty);
    expect(tree.N).toBe(0);
  });
});

describe("refitStepSync — leaf boxes", () => {
  it("first refit produces tight leaf AABBs", async () => {
    const scene = makeScene(4, 5);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);
    for (let p = 0; p < scene.P; p++) {
      const ref = leafBoxFromScene(scene, p);
      // Find the leaf node id for prim p.
      const leafId = ctx.tree.leafPrim.indexOf(p);
      const off = leafId * 4;
      expect(ctx.tree.boxes[off + 0]).toBeCloseTo(ref.minX, 6);
      expect(ctx.tree.boxes[off + 1]).toBeCloseTo(ref.minY, 6);
      expect(ctx.tree.boxes[off + 2]).toBeCloseTo(ref.maxX, 6);
      expect(ctx.tree.boxes[off + 3]).toBeCloseTo(ref.maxY, 6);
    }
  });

  it("root box contains every leaf box", async () => {
    const scene = makeScene(6, 4);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);
    const rootId = ctx.tree.levels[ctx.tree.levels.length - 1][0];
    const rb = ctx.tree.boxes.subarray(rootId * 4, rootId * 4 + 4);
    for (let p = 0; p < scene.P; p++) {
      const ref = leafBoxFromScene(scene, p);
      expect(rb[0]).toBeLessThanOrEqual(ref.minX + 1e-6);
      expect(rb[1]).toBeLessThanOrEqual(ref.minY + 1e-6);
      expect(rb[2]).toBeGreaterThanOrEqual(ref.maxX - 1e-6);
      expect(rb[3]).toBeGreaterThanOrEqual(ref.maxY - 1e-6);
    }
  });
});

describe("refitStepSync — incremental dirty propagation", () => {
  it("after a stationary step, no leaf is dirty", async () => {
    const scene = makeScene(8, 0);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);             // initial recompute, everything dirty
    refitStepSync(ctx);             // nothing moved → no dirty
    expect(ctx.lastDirtyLeaves).toBe(0);
    expect(ctx.lastDirtyInternals).toBe(0);
  });

  it("moving one vertex marks its leaf and its ancestor chain dirty", async () => {
    const scene = makeScene(8, 0);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);
    refitStepSync(ctx);
    // Move vertex 0 (belongs to triangle 0).
    scene.vertices[0] += 5.0;
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(1);
    // Ancestor chain depth >= 1 internal for any P>=2 tree.
    expect(ctx.lastDirtyInternals).toBeGreaterThanOrEqual(1);
    // Verify root contains the moved triangle.
    const rootId = ctx.tree.levels[ctx.tree.levels.length - 1][0];
    expect(ctx.tree.boxes[rootId * 4 + 2]).toBeGreaterThanOrEqual(scene.vertices[0] - 0.05);
  });

  it("only the affected ancestor chain refits — siblings stay clean", async () => {
    const scene = makeScene(16, 0);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);
    refitStepSync(ctx);
    scene.vertices[0] += 0.5; // move tri 0
    refitStepSync(ctx);
    const totalInternals = ctx.tree.N - scene.P;
    // Strict O(log N): far fewer internals refit than total.
    expect(ctx.lastDirtyInternals).toBeLessThan(totalInternals);
  });

  it("sub-epsilon motion does not mark a leaf dirty", async () => {
    const scene = makeScene(4, 0);
    const ctx = await initAabbRefit(scene, { refitEpsilon: 1e-3 });
    refitStepSync(ctx);
    refitStepSync(ctx);
    scene.vertices[0] += 1e-5; // way under epsilon
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(0);
  });

  it("markAllDirty forces a full refit on the next step", async () => {
    const scene = makeScene(6, 0);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(0);
    markAllDirty(ctx);
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(scene.P);
  });
});

describe("refitStepSync — particle shape", () => {
  it("particle shape AABB grows when one particle moves outward", async () => {
    const scene = makeScene(0, 6);
    const ctx = await initAabbRefit(scene);
    refitStepSync(ctx);
    const before = leafBoxFromScene(scene, 0);
    // Push the last particle far to the right.
    scene.vertices[(scene.V - 1) * 2 + 0] = 100;
    refitStepSync(ctx);
    const after = leafBoxFromScene(scene, 0);
    expect(after.maxX).toBeGreaterThan(before.maxX + 50);
    // The leaf node should match.
    const leafId = ctx.tree.leafPrim.indexOf(0);
    expect(ctx.tree.boxes[leafId * 4 + 2]).toBeCloseTo(after.maxX, 6);
  });
});

describe("refitStepSync — determinism", () => {
  it("two contexts on the same scene produce identical box arrays", async () => {
    const scene1 = makeScene(10, 4);
    const scene2 = makeScene(10, 4);
    const c1 = await initAabbRefit(scene1);
    const c2 = await initAabbRefit(scene2);
    refitStepSync(c1); refitStepSync(c1);
    refitStepSync(c2); refitStepSync(c2);
    scene1.vertices[2] += 0.3; scene2.vertices[2] += 0.3;
    refitStepSync(c1); refitStepSync(c2);
    expect(Array.from(c1.tree.boxes)).toEqual(Array.from(c2.tree.boxes));
    expect(c1.lastDirtyLeaves).toBe(c2.lastDirtyLeaves);
    expect(c1.lastDirtyInternals).toBe(c2.lastDirtyInternals);
  });
});
