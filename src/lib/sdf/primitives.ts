import type { SDFPrim, Vec3 } from "./types";

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const scale = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
const normalize = (a: Vec3): Vec3 => {
  const l = len(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};

function sdSphere(p: Vec3, c: Vec3, r: number) {
  return len(sub(p, c)) - r;
}
function sdBox(p: Vec3, c: Vec3, h: Vec3) {
  const d: Vec3 = [Math.abs(p[0] - c[0]) - h[0], Math.abs(p[1] - c[1]) - h[1], Math.abs(p[2] - c[2]) - h[2]];
  const outside = Math.hypot(Math.max(d[0], 0), Math.max(d[1], 0), Math.max(d[2], 0));
  const inside = Math.min(Math.max(d[0], Math.max(d[1], d[2])), 0);
  return outside + inside;
}
function sdCylinder(p: Vec3, c: Vec3, axis: Vec3, r: number, h: number) {
  const a = normalize(axis);
  const rel = sub(p, c);
  const t = dot(rel, a);
  const proj = scale(a, t);
  const radial = sub(rel, proj);
  const dr = len(radial) - r;
  const dh = Math.abs(t) - h * 0.5;
  const outside = Math.hypot(Math.max(dr, 0), Math.max(dh, 0));
  const inside = Math.min(Math.max(dr, dh), 0);
  return outside + inside;
}
function sdPlane(p: Vec3, q: Vec3, n: Vec3) {
  const nn = normalize(n);
  return dot(sub(p, q), nn);
}
function sdTorus(p: Vec3, c: Vec3, R: number, r: number) {
  const rel = sub(p, c);
  const q = Math.hypot(rel[0], rel[2]) - R;
  return Math.hypot(q, rel[1]) - r;
}

export function evalPrim(prim: SDFPrim, p: Vec3): number {
  switch (prim.kind) {
    case "sphere":   return sdSphere(p, prim.center, prim.radius);
    case "box":      return sdBox(p, prim.center, prim.half);
    case "cylinder": return sdCylinder(p, prim.center, prim.axis, prim.radius, prim.height);
    case "plane":    return sdPlane(p, prim.point, prim.normal);
    case "torus":    return sdTorus(p, prim.center, prim.major, prim.minor);
  }
}

/** Smooth union over many primitives. k=0 → hard min. */
export function evalScene(prims: SDFPrim[], p: Vec3, k = 0): number {
  if (prims.length === 0) return Number.POSITIVE_INFINITY;
  let d = evalPrim(prims[0], p);
  for (let i = 1; i < prims.length; i++) {
    const di = evalPrim(prims[i], p);
    if (k <= 0) {
      if (di < d) d = di;
    } else {
      const h = Math.max(k - Math.abs(d - di), 0) / k;
      d = Math.min(d, di) - h * h * k * 0.25;
    }
  }
  return d;
}
