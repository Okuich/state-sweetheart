/**
 * End-to-end runtime smoke test for gpuAabbRefit.
 *
 * Compile-time typechecking can't catch runtime regressions in the refit
 * kernel (CPU fallback path under Node — Vitest has no WebGPU adapter).
 * This test exercises the full lifecycle on a tiny representative scene:
 *
 *   build scene  →  initAabbRefit  →  refitStepSync (cold)
 *                →  mutate vertices →  refitStepSync (incremental)
 *                →  markAllDirty    →  refitStepSync (full rebuild)
 *
 * If any of those steps throw, return malformed boxes, or fail to propagate
 * dirty bits, the smoke test fails — surfacing breakage that `tsc --noEmit`
 * cannot.
 */
import { describe, it, expect } from "vitest";
import {
  initAabbRefit,
  refitStepSync,
  markAllDirty,
  leafBoxFromScene,
  PrimKind,
  type DeformableScene,
} from "./gpuAabbRefit";

function sampleScene(): DeformableScene {
  // 3 cloth triangles + 1 particle shape (4 particles).
  const nTris = 3;
  const nParts = 4;
  const V = nTris * 3 + nParts;
  const P = nTris + 1;
  const vertices = new Float32Array(V * 2);
  const prims = new Uint32Array(P * 4);
  const radii = new Float32Array(P);
  for (let t = 0; t < nTris; t++) {
    const x = t * 2.0;
    vertices.set([x, 0, x + 1, 0, x + 0.5, 1], t * 3 * 2);
    prims.set([PrimKind.ClothTri, t * 3, t * 3 + 1, t * 3 + 2], t * 4);
    radii[t] = 0.05;
  }
  const start = nTris * 3;
  for (let k = 0; k < nParts; k++) {
    vertices[(start + k) * 2 + 0] = k * 0.5;
    vertices[(start + k) * 2 + 1] = -2;
  }
  prims.set([PrimKind.ParticleShape, start, nParts, 0], nTris * 4);
  radii[nTris] = 0.1;
  return { V, P, vertices, prims, radii };
}

describe("gpuAabbRefit — runtime smoke test", () => {
  it("runs the full init → refit → mutate → refit → markAllDirty lifecycle", async () => {
    const scene = sampleScene();

    // Init — must not throw, must pick a mode (cpu under Node).
    const ctx = await initAabbRefit(scene);
    expect(ctx.mode === "cpu" || ctx.mode === "gpu").toBe(true);
    expect(ctx.tree.N).toBeGreaterThan(0);

    // Cold refit — every leaf should be tight against the scene reference.
    refitStepSync(ctx);
    for (let p = 0; p < scene.P; p++) {
      const ref = leafBoxFromScene(scene, p);
      const leafId = ctx.tree.leafPrim.indexOf(p);
      expect(leafId).toBeGreaterThanOrEqual(0);
      const off = leafId * 4;
      expect(ctx.tree.boxes[off + 0]).toBeCloseTo(ref.minX, 5);
      expect(ctx.tree.boxes[off + 1]).toBeCloseTo(ref.minY, 5);
      expect(ctx.tree.boxes[off + 2]).toBeCloseTo(ref.maxX, 5);
      expect(ctx.tree.boxes[off + 3]).toBeCloseTo(ref.maxY, 5);
    }

    // Stationary refit — nothing dirty.
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(0);

    // Mutate one vertex — incremental dirty propagation should fire.
    scene.vertices[0] += 3.0;
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(1);
    expect(ctx.lastDirtyInternals).toBeGreaterThanOrEqual(1);

    // Force-full refit — every leaf marked dirty.
    markAllDirty(ctx);
    refitStepSync(ctx);
    expect(ctx.lastDirtyLeaves).toBe(scene.P);

    // Root must enclose every leaf box.
    const rootId = ctx.tree.levels[ctx.tree.levels.length - 1][0];
    const rb = ctx.tree.boxes.subarray(rootId * 4, rootId * 4 + 4);
    for (let p = 0; p < scene.P; p++) {
      const ref = leafBoxFromScene(scene, p);
      expect(rb[0]).toBeLessThanOrEqual(ref.minX + 1e-5);
      expect(rb[1]).toBeLessThanOrEqual(ref.minY + 1e-5);
      expect(rb[2]).toBeGreaterThanOrEqual(ref.maxX - 1e-5);
      expect(rb[3]).toBeGreaterThanOrEqual(ref.maxY - 1e-5);
    }
  });
});
