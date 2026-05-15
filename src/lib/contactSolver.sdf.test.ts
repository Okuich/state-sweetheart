/**
 * Additional contact-solver tests focused on the static-SDF path:
 *   1. Gradient projection — corrected position lies along the
 *      radial outward normal for a sphere SDF.
 *   2. Purely-Z normals are skipped (no in-plane motion) when the
 *      SDF surface is a horizontal plane at the simulation slice.
 *   3. Kinematic particles (mass = +∞ → invMass = 0) are not
 *      resolved against the SDF, even when penetrating.
 */
import { describe, it, expect } from "vitest";
import { resolveContacts, type ContactState } from "./contactSolver";
import { buildSparseSDF } from "./sdf/sparseField";

function stateAt(x: number, y: number, mass = 1): ContactState {
  return {
    N: 1,
    x: new Float32Array([x, y]),
    v: new Float32Array([0, 0]),
    m: new Float32Array([mass]),
  };
}

describe("resolveSDFContacts — gradient projection", () => {
  // Sphere of radius 0.5 centred at origin; particle radius 0.05.
  // The SDF needs a bbox that fully contains the sphere so the band
  // is populated everywhere we sample.
  const sdf = buildSparseSDF(
    { min: [-1, -1, -1], max: [1, 1, 1] },
    [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }],
  );

  it("pushes a penetrating particle along the outward radial normal", () => {
    // Place a particle deep inside the sphere off-axis.
    const px = 0.2, py = 0.1;
    const inLen = Math.hypot(px, py);
    const s = stateAt(px, py);
    const stats = resolveContacts(s, {
      radius: 0.05,
      beta: 1, slop: 0,
      staticSDF: { sdf },
    });
    expect(stats.sdfContacts).toBe(1);

    // Displacement direction must align with the radial unit vector.
    const dx = s.x[0] - px;
    const dy = s.x[1] - py;
    const dlen = Math.hypot(dx, dy);
    expect(dlen).toBeGreaterThan(0);
    const ux = dx / dlen, uy = dy / dlen;
    const rx = px / inLen, ry = py / inLen;
    // dot(displacement_dir, radial_dir) ≈ 1 → motion is purely outward.
    expect(ux * rx + uy * ry).toBeGreaterThan(0.98);
    // Particle should now be at or outside the surface.
    expect(Math.hypot(s.x[0], s.x[1])).toBeGreaterThanOrEqual(0.5 - 1e-3);
  });
});

describe("resolveSDFContacts — purely-Z normals are skipped", () => {
  // Horizontal plane at z=0 → gradient = (0, 0, 1) at the sim slice.
  const sdf = buildSparseSDF(
    { min: [-1, -1, -0.1], max: [1, 1, 0.1] },
    [{ kind: "plane", point: [0, 0, 0], normal: [0, 0, 1] }],
  );

  it("does not move the particle in XY when the SDF normal is along Z", () => {
    const s = stateAt(0.3, -0.4);
    const x0 = s.x[0], y0 = s.x[1];
    resolveContacts(s, {
      radius: 0.05,
      beta: 1, slop: 0,
      staticSDF: { sdf },
    });
    // In-plane normal length is < 1e-6 → solver early-outs.
    expect(s.x[0]).toBeCloseTo(x0, 6);
    expect(s.x[1]).toBeCloseTo(y0, 6);
    expect(s.v[0]).toBeCloseTo(0, 6);
    expect(s.v[1]).toBeCloseTo(0, 6);
  });
});

describe("resolveSDFContacts — kinematic (invMass=0) particles", () => {
  const sdf = buildSparseSDF(
    { min: [-1, -1, -0.1], max: [1, 1, 0.1] },
    [{ kind: "sphere", center: [0, 0, 0], radius: 0.5 }],
  );

  it("does not resolve particles whose mass equals pinnedMass (∞)", () => {
    const s = stateAt(0.2, 0.1, Infinity);
    s.v[0] = -0.5;
    const x0 = s.x[0], y0 = s.x[1], vx0 = s.v[0], vy0 = s.v[1];
    const stats = resolveContacts(s, {
      radius: 0.05,
      beta: 1, slop: 0,
      staticSDF: { sdf },
      // pinnedMass defaults to Infinity — be explicit for clarity.
      pinnedMass: Infinity,
    });
    // Particle is pinned → wi = 0 → loop skips both detection AND resolve.
    expect(stats.sdfContacts).toBe(0);
    expect(s.x[0]).toBe(x0);
    expect(s.x[1]).toBe(y0);
    expect(s.v[0]).toBe(vx0);
    expect(s.v[1]).toBe(vy0);
  });

  it("resolves a normal-mass particle in the same scene (control)", () => {
    const s = stateAt(0.2, 0.1, 1);
    const stats = resolveContacts(s, {
      radius: 0.05,
      beta: 1, slop: 0,
      staticSDF: { sdf },
    });
    expect(stats.sdfContacts).toBe(1);
    expect(Math.hypot(s.x[0] - 0.2, s.x[1] - 0.1)).toBeGreaterThan(0);
  });
});
