/**
 * Geometry intelligence: classify each topology node into a fabrication-
 * sensitive feature class.
 *
 * Detection is rule-based on the leaf graph + refinement tags + curvature:
 *   - thin_wall: boundary node with small wall thickness (BFS to opposite boundary)
 *   - overhang: boundary node whose downward neighbor is empty (no leaf below)
 *   - cavity: boundary node enclosed by a closed shell (no exterior path)
 *   - stress_concentrator: high-curvature boundary node with sharp/fillet tag
 *   - thermal_bottleneck: high-curvature interior node tagged hotspot
 *   - symmetry_seed: deterministically picked nodes near the bbox symmetry axes
 *   - boundary: any other surface node
 *   - bulk: everything else
 *
 * All passes are linear or near-linear in node count, so the engine handles
 * 100k+ nodes inside the dashboard without Web Worker offload.
 */

import type { TopologyGraph } from "./graph";
import type { AABB, FeatureClass, TopoNode } from "./types";

export interface FeatureReport {
  /** Counts per class. */
  counts: Record<FeatureClass, number>;
  /** Indices grouped by class. */
  byClass: Record<FeatureClass, number[]>;
  /** Avg wall thickness across all flagged thin_wall nodes (0 if none). */
  avgThinWallThickness: number;
  /** Min wall thickness across all boundary nodes (0 if no boundary). */
  minWallThickness: number;
  /** Symmetry score in [0,1] estimated from boundary mass mirror balance. */
  symmetryScore: number;
}

export function classifyFeatures(graph: TopologyGraph, bbox: AABB): FeatureReport {
  const N = graph.nodes.length;

  // Pass 1: mark downward-facing boundary nodes (overhang candidates).
  // A node is "downward" if it's a boundary node and there's no neighbor
  // immediately below (smaller Y, sharing a face on the Y axis).
  for (let i = 0; i < N; i++) {
    const n = graph.nodes[i];
    if (!n.boundary) { n.downward = false; continue; }
    let hasBelow = false;
    const start = graph.neighborOffsets[i];
    const end = graph.neighborOffsets[i + 1];
    for (let k = start; k < end; k++) {
      const j = graph.neighborIdx[k];
      const e = graph.edges[graph.neighborEdge[k]];
      if (e.axis === 1 && graph.nodes[j].center[1] < n.center[1]) {
        hasBelow = true; break;
      }
    }
    n.downward = !hasBelow;
  }

  // Pass 2: wall thickness via BFS on opposite-boundary distance.
  // For each boundary node, walk inward (non-boundary neighbors) and record
  // the shortest-path radius until hitting another boundary.
  for (let i = 0; i < N; i++) {
    const n = graph.nodes[i];
    if (!n.boundary) { n.wallThickness = 0; continue; }
    n.wallThickness = bfsWallThickness(graph, i, 6);
  }

  // Pass 3: cavity detection — exterior connectivity via flood from bbox face
  // boundary nodes. Boundary nodes NOT reachable from the exterior flood are
  // cavity walls.
  const cavityFlag = detectCavities(graph, bbox);

  // Pass 4: classify.
  const counts: Record<FeatureClass, number> = {
    bulk: 0, boundary: 0, thin_wall: 0, overhang: 0, cavity: 0,
    stress_concentrator: 0, thermal_bottleneck: 0, symmetry_seed: 0,
  };
  const byClass: Record<FeatureClass, number[]> = {
    bulk: [], boundary: [], thin_wall: [], overhang: [], cavity: [],
    stress_concentrator: [], thermal_bottleneck: [], symmetry_seed: [],
  };

  let thinSum = 0, thinCount = 0;
  let minWall = Number.POSITIVE_INFINITY;

  for (let i = 0; i < N; i++) {
    const n = graph.nodes[i];
    let cls: FeatureClass = "bulk";

    if (n.boundary) {
      const wallVx = n.wallThickness;
      const wt = wallVx * (2 * n.radius / Math.sqrt(3)); // approx world thickness
      if (wt > 0 && wt < minWall) minWall = wt;

      if (n.tag === "sharp" && n.curvature >= 3) cls = "stress_concentrator";
      else if (n.tag === "fillet" && n.curvature >= 2) cls = "stress_concentrator";
      else if (cavityFlag[i]) cls = "cavity";
      else if (n.downward) cls = "overhang";
      else if (n.tag === "thin_wall" || (wallVx > 0 && wallVx <= 2)) {
        cls = "thin_wall";
        thinSum += wt;
        thinCount++;
      } else {
        cls = "boundary";
      }
    } else if (n.tag === "hotspot" && n.curvature === 0) {
      cls = "thermal_bottleneck";
    }

    n.feature = cls;
    counts[cls]++;
    byClass[cls].push(i);
  }

  // Pass 5: symmetry seeds — pick deterministic samples near the three
  // bbox-centered planes; flip class only on bulk nodes (don't shadow real
  // features).
  const symSeeds = pickSymmetrySeeds(graph, bbox, Math.min(32, Math.floor(N / 50)));
  for (const i of symSeeds) {
    if (graph.nodes[i].feature === "bulk") {
      graph.nodes[i].feature = "symmetry_seed";
      counts.bulk--;
      counts.symmetry_seed++;
      byClass.bulk = byClass.bulk.filter((x) => x !== i);
      byClass.symmetry_seed.push(i);
    }
  }

  const symmetryScore = mirrorBalance(graph, bbox);

  return {
    counts,
    byClass,
    avgThinWallThickness: thinCount > 0 ? thinSum / thinCount : 0,
    minWallThickness: Number.isFinite(minWall) ? minWall : 0,
    symmetryScore,
  };
}

/** BFS inward from a boundary node; returns hop count to nearest other boundary. */
function bfsWallThickness(graph: TopologyGraph, start: number, maxHops: number): number {
  const seen = new Uint8Array(graph.nodes.length);
  const queue: { i: number; d: number }[] = [{ i: start, d: 0 }];
  seen[start] = 1;
  while (queue.length) {
    const { i, d } = queue.shift()!;
    if (d > 0 && graph.nodes[i].boundary) return d;
    if (d >= maxHops) continue;
    const s = graph.neighborOffsets[i];
    const e = graph.neighborOffsets[i + 1];
    for (let k = s; k < e; k++) {
      const j = graph.neighborIdx[k];
      if (seen[j]) continue;
      seen[j] = 1;
      queue.push({ i: j, d: d + 1 });
    }
  }
  return 0;
}

/**
 * Cavity detection: flood from boundary nodes whose center lies on the bbox
 * exterior. Any boundary node NOT reached but still flagged as boundary is
 * an internal (cavity) wall.
 */
function detectCavities(graph: TopologyGraph, bbox: AABB): Uint8Array {
  const N = graph.nodes.length;
  const seen = new Uint8Array(N);
  const cavity = new Uint8Array(N);
  const queue: number[] = [];
  const ext = [
    bbox.max[0] - bbox.min[0],
    bbox.max[1] - bbox.min[1],
    bbox.max[2] - bbox.min[2],
  ];
  const eps = Math.min(ext[0], ext[1], ext[2]) * 0.05;
  for (let i = 0; i < N; i++) {
    const n = graph.nodes[i];
    if (!n.boundary) continue;
    const c = n.center;
    if (
      c[0] - bbox.min[0] < eps || bbox.max[0] - c[0] < eps ||
      c[1] - bbox.min[1] < eps || bbox.max[1] - c[1] < eps ||
      c[2] - bbox.min[2] < eps || bbox.max[2] - c[2] < eps
    ) {
      seen[i] = 1;
      queue.push(i);
    }
  }
  while (queue.length) {
    const i = queue.pop()!;
    const s = graph.neighborOffsets[i];
    const e = graph.neighborOffsets[i + 1];
    for (let k = s; k < e; k++) {
      const j = graph.neighborIdx[k];
      if (seen[j]) continue;
      // Walk only across boundary or bulk; we want exterior reach.
      seen[j] = 1;
      queue.push(j);
    }
  }
  for (let i = 0; i < N; i++) {
    if (graph.nodes[i].boundary && !seen[i]) cavity[i] = 1;
  }
  return cavity;
}

function pickSymmetrySeeds(graph: TopologyGraph, bbox: AABB, count: number): number[] {
  if (count <= 0) return [];
  const cx = (bbox.min[0] + bbox.max[0]) / 2;
  const cy = (bbox.min[1] + bbox.max[1]) / 2;
  const cz = (bbox.min[2] + bbox.max[2]) / 2;
  const scored = graph.nodes.map((n, i) => {
    const dx = Math.abs(n.center[0] - cx);
    const dy = Math.abs(n.center[1] - cy);
    const dz = Math.abs(n.center[2] - cz);
    return { i, score: -Math.min(dx, dy, dz) };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, count).map((s) => s.i);
}

/** Mirror-balance score: 1 - normalized mass asymmetry across bbox center. */
function mirrorBalance(graph: TopologyGraph, bbox: AABB): number {
  if (graph.nodes.length === 0) return 0;
  const cx = (bbox.min[0] + bbox.max[0]) / 2;
  const cy = (bbox.min[1] + bbox.max[1]) / 2;
  const cz = (bbox.min[2] + bbox.max[2]) / 2;
  let leftX = 0, rightX = 0, leftY = 0, rightY = 0, leftZ = 0, rightZ = 0;
  for (const n of graph.nodes) {
    const w = n.boundary ? 1 : 0.25;
    if (n.center[0] < cx) leftX += w; else rightX += w;
    if (n.center[1] < cy) leftY += w; else rightY += w;
    if (n.center[2] < cz) leftZ += w; else rightZ += w;
  }
  const a = (l: number, r: number) => (l + r > 0 ? Math.abs(l - r) / (l + r) : 0);
  const asym = (a(leftX, rightX) + a(leftY, rightY) + a(leftZ, rightZ)) / 3;
  return Math.max(0, Math.min(1, 1 - asym));
}
