/**
 * Mesh quality metrics for tetrahedral meshes.
 * - Aspect ratio (max edge / min altitude proxy)
 * - Signed volume / inverted-element count
 * - Min/mean dihedral approximation via edge ratio
 * - Manifold consistency (face-occurrence parity)
 */

import type { OctreeMesh } from "./octree";

export interface QualityReport {
  tetCount: number;
  vertexCount: number;
  invertedTets: number;
  /** Aspect-ratio histogram, 10 bins on [1, 50+]. */
  aspectHist: number[];
  meanAspect: number;
  worstAspect: number;
  /** Volume stats. */
  totalVolume: number;
  minVolume: number;
  meanVolume: number;
  /** Manifold check: number of internal faces shared by != 2 tets. */
  nonManifoldFaces: number;
  /** Convergence proxy in [0,1] — higher is more solver-friendly. */
  convergenceScore: number;
  /** Computed in ms. */
  ms: number;
}

function v(vs: Float32Array, i: number): [number, number, number] {
  return [vs[i * 3], vs[i * 3 + 1], vs[i * 3 + 2]];
}

function sub(a: number[], b: number[]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function cross(a: number[], b: number[]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function len(a: number[]): number {
  return Math.sqrt(dot(a, a));
}

export function tetVolume(p0: number[], p1: number[], p2: number[], p3: number[]): number {
  return dot(sub(p1, p0), cross(sub(p2, p0), sub(p3, p0))) / 6;
}

export function tetAspectRatio(p0: number[], p1: number[], p2: number[], p3: number[]): number {
  const edges = [
    len(sub(p1, p0)), len(sub(p2, p0)), len(sub(p3, p0)),
    len(sub(p2, p1)), len(sub(p3, p1)), len(sub(p3, p2)),
  ];
  const longest = Math.max(...edges);
  const shortest = Math.min(...edges);
  if (shortest < 1e-12) return 1e6;
  return longest / shortest;
}

function faceKey(a: number, b: number, c: number): string {
  const s = [a, b, c].sort((x, y) => x - y);
  return `${s[0]},${s[1]},${s[2]}`;
}

export function evaluateQuality(mesh: OctreeMesh): QualityReport {
  const t0 = Date.now();
  const tets = mesh.tets;
  const verts = mesh.vertices;
  const tCount = tets.length / 4;

  const aspectHist = new Array(10).fill(0);
  let aspectSum = 0;
  let worst = 0;
  let totalVol = 0;
  let minVol = Infinity;
  let inverted = 0;

  const faceCount = new Map<string, number>();

  for (let t = 0; t < tCount; t++) {
    const i0 = tets[t * 4 + 0];
    const i1 = tets[t * 4 + 1];
    const i2 = tets[t * 4 + 2];
    const i3 = tets[t * 4 + 3];
    const p0 = v(verts, i0);
    const p1 = v(verts, i1);
    const p2 = v(verts, i2);
    const p3 = v(verts, i3);
    const vol = tetVolume(p0, p1, p2, p3);
    if (vol <= 0) inverted++;
    const absVol = Math.abs(vol);
    totalVol += absVol;
    if (absVol < minVol) minVol = absVol;

    const ar = tetAspectRatio(p0, p1, p2, p3);
    aspectSum += ar;
    if (ar > worst) worst = ar;
    const bin = Math.min(9, Math.floor(Math.log2(Math.max(1, ar))));
    aspectHist[bin]++;

    for (const [a, b, c] of [
      [i0, i1, i2], [i0, i1, i3], [i0, i2, i3], [i1, i2, i3],
    ]) {
      const k = faceKey(a, b, c);
      faceCount.set(k, (faceCount.get(k) ?? 0) + 1);
    }
  }

  let nonManifold = 0;
  faceCount.forEach((cnt) => { if (cnt !== 1 && cnt !== 2) nonManifold++; });

  const meanAspect = tCount ? aspectSum / tCount : 0;
  const meanVol = tCount ? totalVol / tCount : 0;

  // Convergence proxy: penalize bad aspect, inverted tets, non-manifold faces.
  const arPenalty = Math.max(0, 1 - (meanAspect - 1) / 20);
  const invPenalty = 1 - inverted / Math.max(1, tCount);
  const manPenalty = 1 - nonManifold / Math.max(1, faceCount.size);
  const convergenceScore = Math.max(0, Math.min(1, arPenalty * invPenalty * manPenalty));

  return {
    tetCount: tCount,
    vertexCount: verts.length / 3,
    invertedTets: inverted,
    aspectHist,
    meanAspect,
    worstAspect: worst,
    totalVolume: totalVol,
    minVolume: tCount ? minVol : 0,
    meanVolume: meanVol,
    nonManifoldFaces: nonManifold,
    convergenceScore,
    ms: Date.now() - t0,
  };
}
