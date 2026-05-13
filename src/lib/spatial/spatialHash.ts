/**
 * Uniform-grid spatial hash for broadphase collision pair generation.
 *
 * Grid cells are addressed via Morton-mixed hash for cache-locality;
 * each prim is bucketed into every cell its AABB overlaps, then pairs
 * within a cell become collision candidates (de-duplicated by id pair).
 *
 * Designed for the sweet spot of broadphase (10²–10⁵ prims, ~uniform sizes).
 */

export type Vec3 = readonly [number, number, number];

export interface SpatialHash {
  cellSize: number;
  origin: Vec3;
  /** CSR-style buckets: bucketStart[H+1], bucketPrim[totalEntries]. */
  bucketStart: Uint32Array;
  bucketPrim: Uint32Array;
  hashSize: number;
  primCount: number;
  buildMs: number;
  meanBucket: number;
  maxBucket: number;
}

const HASH_PRIMES: [number, number, number] = [73856093, 19349663, 83492791];

function hashCell(ix: number, iy: number, iz: number, hashSize: number): number {
  const h = ((ix | 0) * HASH_PRIMES[0]) ^ ((iy | 0) * HASH_PRIMES[1]) ^ ((iz | 0) * HASH_PRIMES[2]);
  return ((h >>> 0) % hashSize) | 0;
}

export function buildSpatialHash(
  primMin: Float32Array,
  primMax: Float32Array,
  cellSize: number,
  origin: Vec3 = [0, 0, 0],
  hashSize?: number,
): SpatialHash {
  const t0 = Date.now();
  const P = primMin.length / 3;
  const H = hashSize ?? Math.max(64, 1 << Math.ceil(Math.log2(Math.max(64, P * 2))));
  const inv = 1 / cellSize;

  // First pass: count per-bucket entries.
  const counts = new Uint32Array(H);
  const cellRanges: Int32Array = new Int32Array(P * 6); // ix0,iy0,iz0,ix1,iy1,iz1
  for (let i = 0; i < P; i++) {
    const ix0 = Math.floor((primMin[i * 3]     - origin[0]) * inv);
    const iy0 = Math.floor((primMin[i * 3 + 1] - origin[1]) * inv);
    const iz0 = Math.floor((primMin[i * 3 + 2] - origin[2]) * inv);
    const ix1 = Math.floor((primMax[i * 3]     - origin[0]) * inv);
    const iy1 = Math.floor((primMax[i * 3 + 1] - origin[1]) * inv);
    const iz1 = Math.floor((primMax[i * 3 + 2] - origin[2]) * inv);
    cellRanges[i * 6]     = ix0; cellRanges[i * 6 + 1] = iy0; cellRanges[i * 6 + 2] = iz0;
    cellRanges[i * 6 + 3] = ix1; cellRanges[i * 6 + 4] = iy1; cellRanges[i * 6 + 5] = iz1;
    for (let z = iz0; z <= iz1; z++)
      for (let y = iy0; y <= iy1; y++)
        for (let x = ix0; x <= ix1; x++)
          counts[hashCell(x, y, z, H)]++;
  }

  const bucketStart = new Uint32Array(H + 1);
  let total = 0;
  let maxB = 0;
  for (let h = 0; h < H; h++) {
    bucketStart[h] = total;
    total += counts[h];
    if (counts[h] > maxB) maxB = counts[h];
  }
  bucketStart[H] = total;
  const bucketPrim = new Uint32Array(total);
  const cursor = new Uint32Array(H);

  // Second pass: scatter prim ids into buckets.
  for (let i = 0; i < P; i++) {
    const ix0 = cellRanges[i * 6],     iy0 = cellRanges[i * 6 + 1], iz0 = cellRanges[i * 6 + 2];
    const ix1 = cellRanges[i * 6 + 3], iy1 = cellRanges[i * 6 + 4], iz1 = cellRanges[i * 6 + 5];
    for (let z = iz0; z <= iz1; z++)
      for (let y = iy0; y <= iy1; y++)
        for (let x = ix0; x <= ix1; x++) {
          const h = hashCell(x, y, z, H);
          bucketPrim[bucketStart[h] + cursor[h]++] = i;
        }
  }

  return {
    cellSize,
    origin,
    bucketStart,
    bucketPrim,
    hashSize: H,
    primCount: P,
    buildMs: Date.now() - t0,
    meanBucket: H ? total / H : 0,
    maxBucket: maxB,
  };
}

/** Generate unique broadphase candidate pairs with AABB overlap pre-test. */
export function broadphasePairs(
  hash: SpatialHash,
  primMin: Float32Array,
  primMax: Float32Array,
  max = 1_000_000,
): Uint32Array {
  const seen = new Set<number>();
  const out: number[] = [];
  for (let h = 0; h < hash.hashSize; h++) {
    const a = hash.bucketStart[h], b = hash.bucketStart[h + 1];
    for (let i = a; i < b; i++)
      for (let j = i + 1; j < b; j++) {
        let pi = hash.bucketPrim[i], pj = hash.bucketPrim[j];
        if (pi === pj) continue;
        if (pi > pj) { const t = pi; pi = pj; pj = t; }
        const key = pi * 0x100000 + pj;
        if (seen.has(key)) continue;
        // Conservative AABB overlap.
        if (
          primMax[pi * 3]     >= primMin[pj * 3]     && primMin[pi * 3]     <= primMax[pj * 3] &&
          primMax[pi * 3 + 1] >= primMin[pj * 3 + 1] && primMin[pi * 3 + 1] <= primMax[pj * 3 + 1] &&
          primMax[pi * 3 + 2] >= primMin[pj * 3 + 2] && primMin[pi * 3 + 2] <= primMax[pj * 3 + 2]
        ) {
          seen.add(key);
          out.push(pi, pj);
          if (out.length / 2 >= max) return new Uint32Array(out);
        }
      }
  }
  return new Uint32Array(out);
}
