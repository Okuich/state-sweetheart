/**
 * CPU vs GPU AABB refit correctness check.
 *
 * Builds a few pseudo-random scenes, runs the same mutation sequence on
 * two contexts (one CPU, one GPU when an adapter is available), and
 * verifies that every node's box matches within a tight epsilon.
 *
 * Under Node (Vitest default), `navigator.gpu` is absent so the GPU
 * context falls back to CPU; the test then degenerates into a
 * self-consistency check (still useful — it catches non-determinism in
 * the CPU path). When run in a browser with WebGPU, both paths execute
 * and the comparison becomes a real GPU↔CPU correctness gate.
 */
import { describe, it, expect } from "vitest";
import {
  initAabbRefit,
  refitStep,
  refitStepSync,
  PrimKind,
  type DeformableScene,
  type AabbRefitContext,
} from "./gpuAabbRefit";

// Tiny seeded PRNG so scenes are reproducible across runs.
function mulberry32(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6D2B79F5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function randomScene(seed: number, nTris: number, nParts: number): DeformableScene {
  const rng = mulberry32(seed);
  const V = nTris * 3 + nParts;
  const P = nTris + (nParts > 0 ? 1 : 0);
  const vertices = new Float32Array(V * 2);
  const prims = new Uint32Array(P * 4);
  const radii = new Float32Array(P);

  for (let t = 0; t < nTris; t++) {
    const cx = rng() * 20 - 10;
    const cy = rng() * 20 - 10;
    for (let k = 0; k < 3; k++) {
      vertices[(t * 3 + k) * 2 + 0] = cx + (rng() - 0.5) * 2;
      vertices[(t * 3 + k) * 2 + 1] = cy + (rng() - 0.5) * 2;
    }
    prims.set([PrimKind.ClothTri, t * 3, t * 3 + 1, t * 3 + 2], t * 4);
    radii[t] = 0.05 + rng() * 0.05;
  }
  if (nParts > 0) {
    const start = nTris * 3;
    for (let k = 0; k < nParts; k++) {
      vertices[(start + k) * 2 + 0] = rng() * 20 - 10;
      vertices[(start + k) * 2 + 1] = rng() * 20 - 10;
    }
    prims.set([PrimKind.ParticleShape, start, nParts, 0], nTris * 4);
    radii[nTris] = 0.1 + rng() * 0.1;
  }
  return { V, P, vertices, prims, radii };
}

/** Apply the same per-vertex jitter to two scenes that started identical. */
function jitterBoth(a: DeformableScene, b: DeformableScene, rng: () => number, amp: number) {
  for (let i = 0; i < a.vertices.length; i++) {
    const d = (rng() - 0.5) * amp;
    a.vertices[i] += d;
    b.vertices[i] += d;
  }
}

function expectBoxesMatch(cpu: AabbRefitContext, gpu: AabbRefitContext, eps = 1e-5) {
  expect(gpu.tree.N).toBe(cpu.tree.N);
  for (let i = 0; i < cpu.tree.N * 4; i++) {
    const dc = cpu.tree.boxes[i];
    const dg = gpu.tree.boxes[i];
    expect(Math.abs(dc - dg)).toBeLessThanOrEqual(eps);
  }
}

describe("gpuAabbRefit — CPU vs GPU correctness", () => {
  const cases = [
    { seed: 1,  nTris: 6,  nParts: 0  },
    { seed: 17, nTris: 12, nParts: 5  },
    { seed: 42, nTris: 24, nParts: 10 },
  ];

  for (const { seed, nTris, nParts } of cases) {
    it(`matches across CPU & GPU paths (seed=${seed}, tris=${nTris}, parts=${nParts})`, async () => {
      const sceneCpu = randomScene(seed, nTris, nParts);
      const sceneGpu = randomScene(seed, nTris, nParts);

      const ctxCpu = await initAabbRefit(sceneCpu);
      const ctxGpu = await initAabbRefit(sceneGpu);

      // Cold refit on both.
      refitStepSync(ctxCpu);
      await refitStep(ctxGpu);
      expectBoxesMatch(ctxCpu, ctxGpu);

      // Several mutation rounds — incremental dirty propagation must
      // produce identical boxes on both backends.
      const rng = mulberry32(seed ^ 0xA5A5A5A5);
      for (let round = 0; round < 4; round++) {
        jitterBoth(sceneCpu, sceneGpu, rng, 0.5);
        refitStepSync(ctxCpu);
        await refitStep(ctxGpu);
        expectBoxesMatch(ctxCpu, ctxGpu);
      }

      // Note when the GPU path was actually exercised vs falling back.
      // In Node (no navigator.gpu) both are "cpu"; that's expected.
      expect(ctxCpu.mode).toBe("cpu");
      expect(["cpu", "gpu"]).toContain(ctxGpu.mode);
    });
  }
});
