/**
 * 30-bit Morton (Z-order) encoding for 3D points.
 * Used to produce GPU-friendly cache-local geometry layouts and to seed
 * the LBVH builder.
 */

export type Vec3 = readonly [number, number, number];

/** Spread a 10-bit integer into every 3rd bit (bits 0..29). */
export function expandBits(v: number): number {
  v = (v * 0x00010001) & 0xff0000ff;
  v = (v * 0x00000101) & 0x0f00f00f;
  v = (v * 0x00000011) & 0xc30c30c3;
  v = (v * 0x00000005) & 0x49249249;
  return v >>> 0;
}

/** Encode normalized point in [0,1]^3 to 30-bit Morton code. */
export function morton3D(x: number, y: number, z: number): number {
  const xx = expandBits(Math.min(1023, Math.max(0, Math.floor(x * 1024))));
  const yy = expandBits(Math.min(1023, Math.max(0, Math.floor(y * 1024))));
  const zz = expandBits(Math.min(1023, Math.max(0, Math.floor(z * 1024))));
  return (xx * 4 + yy * 2 + zz) >>> 0;
}

/** Compute Morton codes for a set of points relative to bbox. */
export function mortonCodes(
  points: Float32Array,
  bbox: { min: Vec3; max: Vec3 },
): Uint32Array {
  const n = points.length / 3;
  const out = new Uint32Array(n);
  const sx = 1 / Math.max(1e-12, bbox.max[0] - bbox.min[0]);
  const sy = 1 / Math.max(1e-12, bbox.max[1] - bbox.min[1]);
  const sz = 1 / Math.max(1e-12, bbox.max[2] - bbox.min[2]);
  for (let i = 0; i < n; i++) {
    const nx = (points[i * 3] - bbox.min[0]) * sx;
    const ny = (points[i * 3 + 1] - bbox.min[1]) * sy;
    const nz = (points[i * 3 + 2] - bbox.min[2]) * sz;
    out[i] = morton3D(nx, ny, nz);
  }
  return out;
}

/**
 * Argsort indices by Morton code. Returns the permutation that orders
 * primitives along the Z-order curve (warp-coalesced layout).
 */
export function mortonOrder(codes: Uint32Array): Uint32Array {
  const n = codes.length;
  const idx = new Uint32Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  const arr = Array.from(idx);
  arr.sort((a, b) => codes[a] - codes[b]);
  return Uint32Array.from(arr);
}
