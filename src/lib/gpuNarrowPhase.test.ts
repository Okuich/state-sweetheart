import { describe, it, expect } from "vitest";
import {
  initNarrowPhase,
  solvePairs,
  solvePairsCpu,
  applyContacts,
  BodyKind,
  type BodySet,
  type CandidatePair,
} from "./gpuNarrowPhase";

function makeBodies(specs: Array<{
  x: number; y: number; vx?: number; vy?: number;
  invMass?: number; kind: BodyKind; extra: number; e?: number;
}>): BodySet {
  const N = specs.length;
  const b: BodySet = {
    N,
    pos:        new Float32Array(N * 2),
    vel:        new Float32Array(N * 2),
    invMass:    new Float32Array(N),
    kind:       new Uint32Array(N),
    extra:      new Float32Array(N),
    restitution:new Float32Array(N),
  };
  specs.forEach((s, i) => {
    b.pos[i * 2]     = s.x; b.pos[i * 2 + 1]     = s.y;
    b.vel[i * 2]     = s.vx ?? 0; b.vel[i * 2 + 1] = s.vy ?? 0;
    b.invMass[i]     = s.invMass ?? 1;
    b.kind[i]        = s.kind;
    b.extra[i]       = s.extra;
    b.restitution[i] = s.e ?? 0;
  });
  return b;
}

describe("narrow-phase contact solver (CPU path)", () => {
  it("returns no contacts when bodies are far apart", () => {
    const b = makeBodies([
      { x: 0, y: 0, kind: BodyKind.Particle, extra: 1 },
      { x: 10, y: 0, kind: BodyKind.Particle, extra: 1 },
    ]);
    const pairs: CandidatePair[] = [{ i: 0, j: 1 }];
    const r = solvePairsCpu({ bodies: b, pairs }, 16);
    expect(r.contacts).toHaveLength(0);
  });

  it("particle–particle: detects overlap, computes normal/depth/lambda", () => {
    // Two unit particles centered 1.5 apart → overlap depth = 0.5
    // Approaching at vrel = -2 along +x; e=0; wsum = 2; λ = -(1)·(-2)/2 = 1
    const b = makeBodies([
      { x: 0, y: 0, vx: +1, kind: BodyKind.Particle, extra: 1, e: 0 },
      { x: 1.5, y: 0, vx: -1, kind: BodyKind.Particle, extra: 1, e: 0 },
    ]);
    const r = solvePairsCpu({ bodies: b, pairs: [{ i: 0, j: 1 }] }, 16);
    expect(r.contacts).toHaveLength(1);
    const c = r.contacts[0];
    expect(c.nx).toBeCloseTo(1, 6);
    expect(c.ny).toBeCloseTo(0, 6);
    expect(c.depth).toBeCloseTo(0.5, 6);
    expect(c.lambda).toBeCloseTo(1, 6);
  });

  it("respects per-pair restitution (perfectly elastic doubles impulse)", () => {
    const b = makeBodies([
      { x: 0, y: 0, vx: +1, kind: BodyKind.Particle, extra: 1, e: 1 },
      { x: 1.5, y: 0, vx: -1, kind: BodyKind.Particle, extra: 1, e: 1 },
    ]);
    const r = solvePairsCpu({ bodies: b, pairs: [{ i: 0, j: 1 }] }, 16);
    expect(r.contacts[0].lambda).toBeCloseTo(2, 6);
  });

  it("kinematic body (invMass=0) absorbs impulse but doesn't contribute mass", () => {
    const b = makeBodies([
      { x: 0, y: 0, vx: +1, invMass: 1, kind: BodyKind.Particle, extra: 1, e: 0 },
      { x: 1.5, y: 0, vx: 0, invMass: 0, kind: BodyKind.Rigid, extra: 1, e: 0 },
    ]);
    const r = solvePairsCpu({ bodies: b, pairs: [{ i: 0, j: 1 }] }, 16);
    // wsum = 1; vrel = -1; λ = 1
    expect(r.contacts[0].lambda).toBeCloseTo(1, 6);
    applyContacts(b, r.contacts);
    // Kinematic body did not move
    expect(b.pos[2]).toBeCloseTo(1.5, 6);
    expect(b.vel[2]).toBeCloseTo(0, 6);
    // Dynamic body got the full position correction + impulse
    expect(b.pos[0]).toBeLessThan(0);
    expect(b.vel[0]).toBeLessThanOrEqual(0);
  });

  it("two static bodies produce a contact but no impulse", () => {
    const b = makeBodies([
      { x: 0, y: 0, invMass: 0, kind: BodyKind.Rigid, extra: 1 },
      { x: 1, y: 0, invMass: 0, kind: BodyKind.Rigid, extra: 1 },
    ]);
    const r = solvePairsCpu({ bodies: b, pairs: [{ i: 0, j: 1 }] }, 16);
    expect(r.contacts).toHaveLength(1);
    expect(r.contacts[0].lambda).toBe(0);
  });

  it("rigid–cloth–particle mix: type-aware contact radii", () => {
    const b = makeBodies([
      { x: 0,   y: 0, kind: BodyKind.Rigid,    extra: 1.0 }, // sphere r=1
      { x: 1.2, y: 0, kind: BodyKind.Cloth,    extra: 0.3 }, // thickness 0.3
      { x: 4.0, y: 0, kind: BodyKind.Particle, extra: 0.2 }, // r=0.2
    ]);
    const r = solvePairsCpu({
      bodies: b,
      pairs: [{ i: 0, j: 1 }, { i: 1, j: 2 }, { i: 0, j: 2 }],
    }, 16);
    // 0–1 overlap: 1.0 + 0.3 = 1.3 > 1.2 → contact
    // 1–2 overlap: 0.3 + 0.2 = 0.5  < 2.8 → no contact
    // 0–2 overlap: 1.0 + 0.2 = 1.2  < 4.0 → no contact
    expect(r.contacts).toHaveLength(1);
    expect(r.contacts[0].i).toBe(0);
    expect(r.contacts[0].j).toBe(1);
    expect(r.contacts[0].depth).toBeCloseTo(0.1, 6);
  });

  it("clothScale shrinks effective cloth thickness", () => {
    const b = makeBodies([
      { x: 0, y: 0, kind: BodyKind.Cloth, extra: 1 },
      { x: 1.5, y: 0, kind: BodyKind.Cloth, extra: 1 },
    ]);
    const pairs = [{ i: 0, j: 1 }];
    const full = solvePairsCpu({ bodies: b, pairs }, 16);
    const half = solvePairsCpu({ bodies: b, pairs, clothScale: 0.5 }, 16);
    expect(full.contacts).toHaveLength(1); // 2 > 1.5
    expect(half.contacts).toHaveLength(0); // 1 < 1.5
  });

  it("truncates when contact count exceeds maxContacts", () => {
    // 5 overlapping particles at the same point — every i<j pair contacts
    const b = makeBodies(Array.from({ length: 5 }, (_, k) => ({
      x: k * 0.01, y: 0, kind: BodyKind.Particle, extra: 1,
    })));
    const pairs: CandidatePair[] = [];
    for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) pairs.push({ i, j });
    const r = solvePairsCpu({ bodies: b, pairs }, 3);
    expect(r.contacts).toHaveLength(3);
    expect(r.truncated).toBe(true);
  });

  it("skips degenerate coincident-point pairs (avoids NaN normal)", () => {
    const b = makeBodies([
      { x: 1, y: 1, kind: BodyKind.Particle, extra: 1 },
      { x: 1, y: 1, kind: BodyKind.Particle, extra: 1 },
    ]);
    const r = solvePairsCpu({ bodies: b, pairs: [{ i: 0, j: 1 }] }, 16);
    expect(r.contacts).toHaveLength(0);
  });
});

describe("initNarrowPhase / solvePairs (auto path, sandbox = CPU)", () => {
  it("falls back to CPU when WebGPU is unavailable", async () => {
    const np = await initNarrowPhase();
    expect(np.mode).toBe("cpu");
    const b = makeBodies([
      { x: 0, y: 0, vx: +1, kind: BodyKind.Particle, extra: 1, e: 0 },
      { x: 1.5, y: 0, vx: -1, kind: BodyKind.Particle, extra: 1, e: 0 },
    ]);
    const r = await solvePairs(np, { bodies: b, pairs: [{ i: 0, j: 1 }] });
    expect(r.mode).toBe("cpu");
    expect(r.contacts).toHaveLength(1);
    expect(r.contacts[0].depth).toBeCloseTo(0.5, 6);
    np.destroy();
  });

  it("forceCpu option works", async () => {
    const np = await initNarrowPhase({ forceCpu: true });
    expect(np.mode).toBe("cpu");
  });
});

describe("applyContacts", () => {
  it("conserves momentum on equal-mass head-on collision", () => {
    const b = makeBodies([
      { x: 0, y: 0, vx: +2, kind: BodyKind.Particle, extra: 1, e: 1 },
      { x: 1.5, y: 0, vx: -2, kind: BodyKind.Particle, extra: 1, e: 1 },
    ]);
    const pBefore = (b.vel[0] / 1) + (b.vel[2] / 1); // p = m·v, m=1 for both
    const r = solvePairsCpu({ bodies: b, pairs: [{ i: 0, j: 1 }] }, 16);
    applyContacts(b, r.contacts);
    const pAfter = b.vel[0] + b.vel[2];
    expect(pAfter).toBeCloseTo(pBefore, 5);
    // Elastic: speeds swap (here, both flip sign since masses equal)
    expect(b.vel[0]).toBeCloseTo(-2, 5);
    expect(b.vel[2]).toBeCloseTo(+2, 5);
  });
});
