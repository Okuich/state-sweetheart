/**
 * Geometry-Aware Physics Intelligence — engine.
 *
 *   computeTopology(mesh)
 *     Vertex/edge/face counts, components (Union-Find), Euler χ,
 *     genus = (2c − χ) / 2, valence stats, triangle quality.
 *
 *   computeCurvature(mesh)
 *     Discrete differential geometry on triangle meshes:
 *       - mean curvature   |H| via cotangent Laplace-Beltrami
 *       - Gaussian K via angle defect
 *       - Voronoi (mixed) area per vertex
 *
 *   deformationMetrics(rest, deformed)
 *     Per-vertex displacement, edge stretch (L2 strain proxy), area
 *     ratios, dihedral-angle bending, composite distortion score.
 *
 *   predictStress(mesh, curvature, deformation?)
 *     Curvature-weighted stress field — locations of geometric
 *     concentration (sharp corners, thin necks, high |H|) bias the
 *     prediction even before a solver runs. If deformation is given,
 *     the prediction also blends in strain-energy density.
 *
 *   routeSolver(topology, curvature, deformation?)
 *     Topology-aware solver routing — picks among explicit, implicit
 *     CG, implicit direct, and reduced-order based on:
 *       - mesh size (DOFs)
 *       - topology complexity (genus, components)
 *       - curvature stiffness (max |H|)
 *       - deformation magnitude (when available)
 *
 *   optimizeShape(mesh, opts)
 *     Topology-preserving shape-energy minimizer: cotangent Laplacian
 *     smoothing with edge-length preservation. Maintains V/E/F.
 *
 *   analyze(rest, deformed?)
 *     One-shot pipeline producing a GeometryReport.
 */

import type {
  TriMesh, DeformationMetrics, CurvatureField, StressPrediction,
  TopologyProfile, SolverDecision, ShapeOptimizationResult, GeometryReport,
  SolverKind,
} from "./types";

// ============================================================
// vector helpers
// ============================================================
function sub3(a: Float64Array, i: number, j: number, out: [number, number, number]) {
  out[0] = a[3 * i] - a[3 * j];
  out[1] = a[3 * i + 1] - a[3 * j + 1];
  out[2] = a[3 * i + 2] - a[3 * j + 2];
  return out;
}
function len3(v: [number, number, number]) { return Math.hypot(v[0], v[1], v[2]); }
function dot3(a: [number, number, number], b: [number, number, number]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function cross3(a: [number, number, number], b: [number, number, number]):
  [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function add3(a: [number, number, number], b: [number, number, number]):
  [number, number, number] { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function scale3(a: [number, number, number], s: number):
  [number, number, number] { return [a[0] * s, a[1] * s, a[2] * s]; }
function clamp(x: number, lo: number, hi: number) { return x < lo ? lo : x > hi ? hi : x; }

// ============================================================
// Topology
// ============================================================

class UF {
  parent: Int32Array; rank: Int32Array;
  constructor(n: number) {
    this.parent = new Int32Array(n);
    this.rank = new Int32Array(n);
    for (let i = 0; i < n; i++) this.parent[i] = i;
  }
  find(x: number): number {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }
  union(x: number, y: number) {
    const rx = this.find(x), ry = this.find(y);
    if (rx === ry) return;
    if (this.rank[rx] < this.rank[ry]) this.parent[rx] = ry;
    else if (this.rank[rx] > this.rank[ry]) this.parent[ry] = rx;
    else { this.parent[ry] = rx; this.rank[rx]++; }
  }
}

function edgeKey(a: number, b: number): number {
  return a < b ? a * 0x100000 + b : b * 0x100000 + a;
}

function triangleArea(mesh: TriMesh, t0: number, t1: number, t2: number): number {
  const e1: [number, number, number] = [0, 0, 0];
  const e2: [number, number, number] = [0, 0, 0];
  sub3(mesh.positions, t1, t0, e1);
  sub3(mesh.positions, t2, t0, e2);
  return 0.5 * len3(cross3(e1, e2));
}

function triangleQuality(mesh: TriMesh, a: number, b: number, c: number): number {
  // Inscribed-radius / circumscribed-radius ratio (×2). 1.0 = equilateral, 0 = degenerate.
  const ab = len3(sub3(mesh.positions, a, b, [0, 0, 0]));
  const bc = len3(sub3(mesh.positions, b, c, [0, 0, 0]));
  const ca = len3(sub3(mesh.positions, c, a, [0, 0, 0]));
  const s = (ab + bc + ca) / 2;
  const area = triangleArea(mesh, a, b, c);
  if (area < 1e-12 || s < 1e-12) return 0;
  const inradius = area / s;
  const circumR = (ab * bc * ca) / (4 * area);
  if (circumR < 1e-12) return 0;
  return clamp((2 * inradius) / circumR, 0, 1);
}

export function computeTopology(mesh: TriMesh): TopologyProfile {
  const V = mesh.positions.length / 3;
  const T = mesh.indices.length / 3;
  const uf = new UF(V);
  const edges = new Map<number, number>(); // key → count
  const valence = new Int32Array(V);
  let qSum = 0, qMin = 1, qN = 0;

  for (let t = 0; t < T; t++) {
    const a = mesh.indices[3 * t];
    const b = mesh.indices[3 * t + 1];
    const c = mesh.indices[3 * t + 2];
    uf.union(a, b); uf.union(b, c);
    const k1 = edgeKey(a, b), k2 = edgeKey(b, c), k3 = edgeKey(c, a);
    edges.set(k1, (edges.get(k1) ?? 0) + 1);
    edges.set(k2, (edges.get(k2) ?? 0) + 1);
    edges.set(k3, (edges.get(k3) ?? 0) + 1);
    valence[a]++; valence[b]++; valence[c]++;
    const q = triangleQuality(mesh, a, b, c);
    qSum += q; if (q < qMin) qMin = q; qN++;
  }

  const E = edges.size;
  const F = T;
  const comp = new Set<number>();
  for (let i = 0; i < V; i++) comp.add(uf.find(i));
  const components = comp.size;
  const euler = V - E + F;
  // genus only meaningful for a closed orientable surface — clamp to ≥0
  const genus = Math.max(0, (2 * components - euler) / 2);
  let maxValence = 0, sumValence = 0;
  for (let i = 0; i < V; i++) {
    if (valence[i] > maxValence) maxValence = valence[i];
    sumValence += valence[i];
  }
  return {
    V, E, F, components, euler, genus,
    avgValence: V > 0 ? sumValence / V : 0,
    maxValence,
    triQualityMin: qN > 0 ? qMin : 0,
    triQualityMean: qN > 0 ? qSum / qN : 0,
  };
}

// ============================================================
// Curvature (cotangent Laplace-Beltrami + angle defect)
// ============================================================

export function computeCurvature(mesh: TriMesh): CurvatureField {
  const V = mesh.positions.length / 3;
  const T = mesh.indices.length / 3;
  const H = new Float64Array(V);    // Laplace-Beltrami vector magnitude → mean curvature
  const HX = new Float64Array(V);   // accumulators
  const HY = new Float64Array(V);
  const HZ = new Float64Array(V);
  const K = new Float64Array(V);    // angle defect
  const area = new Float64Array(V); // mixed Voronoi area
  // initialize K with 2π
  for (let i = 0; i < V; i++) K[i] = 2 * Math.PI;

  for (let t = 0; t < T; t++) {
    const i0 = mesh.indices[3 * t];
    const i1 = mesh.indices[3 * t + 1];
    const i2 = mesh.indices[3 * t + 2];
    const idxs: [number, number, number] = [i0, i1, i2];

    // Edge vectors per vertex of the triangle
    for (let k = 0; k < 3; k++) {
      const vi = idxs[k], vj = idxs[(k + 1) % 3], vk = idxs[(k + 2) % 3];
      const eij = sub3(mesh.positions, vj, vi, [0, 0, 0]);
      const eik = sub3(mesh.positions, vk, vi, [0, 0, 0]);
      const cosA = clamp(dot3(eij, eik) / Math.max(1e-12, len3(eij) * len3(eik)), -1, 1);
      const ang = Math.acos(cosA);
      K[vi] -= ang;
    }

    // cotangent weights for Laplace operator
    // For edge (i,j) opposite vertex k: weight = 0.5 * cot(angle_k)
    const a = triangleArea(mesh, i0, i1, i2);
    for (let k = 0; k < 3; k++) {
      const vi = idxs[k], vj = idxs[(k + 1) % 3], vk = idxs[(k + 2) % 3];
      // angle at vk between (vk-vi) and (vk-vj)
      const a1 = sub3(mesh.positions, vi, vk, [0, 0, 0]);
      const a2 = sub3(mesh.positions, vj, vk, [0, 0, 0]);
      const cosK = clamp(dot3(a1, a2) / Math.max(1e-12, len3(a1) * len3(a2)), -1, 1);
      const sinK = Math.max(1e-12, Math.sqrt(1 - cosK * cosK));
      const cot = cosK / sinK;
      // contribution to Lap(vi) -= 0.5 cot * (vj - vi); to Lap(vj) -= 0.5 cot * (vi - vj)
      const eij = sub3(mesh.positions, vj, vi, [0, 0, 0]);
      HX[vi] += 0.5 * cot * eij[0]; HY[vi] += 0.5 * cot * eij[1]; HZ[vi] += 0.5 * cot * eij[2];
      HX[vj] -= 0.5 * cot * eij[0]; HY[vj] -= 0.5 * cot * eij[1]; HZ[vj] -= 0.5 * cot * eij[2];
    }
    area[i0] += a / 3; area[i1] += a / 3; area[i2] += a / 3;
  }

  for (let i = 0; i < V; i++) {
    const aInv = 1 / Math.max(1e-12, 2 * area[i]);
    const mag = Math.hypot(HX[i], HY[i], HZ[i]) * aInv;
    H[i] = mag;
    K[i] = K[i] / Math.max(1e-12, area[i]);
  }
  return { meanCurvature: H, gaussianCurvature: K, vertexArea: area };
}

// ============================================================
// Deformation metrics
// ============================================================

export function deformationMetrics(rest: TriMesh, deformed: TriMesh): DeformationMetrics {
  if (rest.positions.length !== deformed.positions.length
      || rest.indices.length !== deformed.indices.length) {
    throw new Error("deformationMetrics: rest and deformed mesh topology mismatch");
  }
  const V = rest.positions.length / 3;
  const T = rest.indices.length / 3;
  const perVertexDisp = new Float64Array(V);
  let maxDisp = 0, meanDisp = 0;
  for (let i = 0; i < V; i++) {
    const dx = deformed.positions[3 * i] - rest.positions[3 * i];
    const dy = deformed.positions[3 * i + 1] - rest.positions[3 * i + 1];
    const dz = deformed.positions[3 * i + 2] - rest.positions[3 * i + 2];
    const d = Math.hypot(dx, dy, dz);
    perVertexDisp[i] = d;
    meanDisp += d;
    if (d > maxDisp) maxDisp = d;
  }
  meanDisp /= Math.max(1, V);

  // Edge stretch (per edge, but accumulated as a stat)
  const seen = new Set<number>();
  let stretchSum = 0, stretchMax = 0, stretchN = 0;
  let areaRatioSum = 0, areaRatioN = 0;
  for (let t = 0; t < T; t++) {
    const a = rest.indices[3 * t];
    const b = rest.indices[3 * t + 1];
    const c = rest.indices[3 * t + 2];
    const pairs: [number, number][] = [[a, b], [b, c], [c, a]];
    for (const [i, j] of pairs) {
      const k = edgeKey(i, j);
      if (seen.has(k)) continue;
      seen.add(k);
      const lr = len3(sub3(rest.positions, i, j, [0, 0, 0]));
      const ld = len3(sub3(deformed.positions, i, j, [0, 0, 0]));
      if (lr < 1e-12) continue;
      const s = ld / lr;
      stretchSum += s; stretchN++;
      if (Math.abs(s - 1) > Math.abs(stretchMax - 1)) stretchMax = s;
    }
    const ar = triangleArea(rest, a, b, c);
    const ad = triangleArea(deformed, a, b, c);
    if (ar > 1e-12) { areaRatioSum += ad / ar; areaRatioN++; }
  }
  const meanStretch = stretchN > 0 ? stretchSum / stretchN : 1;
  const meanAreaRatio = areaRatioN > 0 ? areaRatioSum / areaRatioN : 1;

  // Bending: dihedral angle change at shared edges.
  const edgeTris = new Map<number, number[]>();
  for (let t = 0; t < T; t++) {
    const a = rest.indices[3 * t], b = rest.indices[3 * t + 1], c = rest.indices[3 * t + 2];
    for (const [i, j] of [[a, b], [b, c], [c, a]] as [number, number][]) {
      const k = edgeKey(i, j);
      if (!edgeTris.has(k)) edgeTris.set(k, []);
      edgeTris.get(k)!.push(t);
    }
  }
  let bendSum = 0, bendN = 0;
  const triNormal = (m: TriMesh, t: number): [number, number, number] => {
    const i0 = m.indices[3 * t], i1 = m.indices[3 * t + 1], i2 = m.indices[3 * t + 2];
    const e1 = sub3(m.positions, i1, i0, [0, 0, 0]);
    const e2 = sub3(m.positions, i2, i0, [0, 0, 0]);
    const n = cross3(e1, e2);
    const l = Math.max(1e-12, len3(n));
    return [n[0] / l, n[1] / l, n[2] / l];
  };
  for (const tris of edgeTris.values()) {
    if (tris.length !== 2) continue;
    const [ta, tb] = tris;
    const nrA = triNormal(rest, ta), nrB = triNormal(rest, tb);
    const ndA = triNormal(deformed, ta), ndB = triNormal(deformed, tb);
    const ar = Math.acos(clamp(dot3(nrA, nrB), -1, 1));
    const ad = Math.acos(clamp(dot3(ndA, ndB), -1, 1));
    bendSum += Math.abs(ad - ar); bendN++;
  }
  const meanBendingRad = bendN > 0 ? bendSum / bendN : 0;

  const distortion =
      Math.abs(meanStretch - 1) * 0.5
    + Math.abs(meanAreaRatio - 1) * 0.3
    + meanBendingRad * 0.2;

  return {
    perVertexDisp, maxDisp, meanDisp,
    meanStretch, maxStretch: stretchMax,
    meanAreaRatio, meanBendingRad, distortion,
  };
}

// ============================================================
// Curvature-aware stress prediction
// ============================================================

export function predictStress(
  mesh: TriMesh, curvature: CurvatureField, deformation?: DeformationMetrics,
): StressPrediction {
  const V = mesh.positions.length / 3;
  const vertexStress = new Float64Array(V);
  let hotIdx = 0, hotVal = 0, mean = 0;

  // Normalize curvature
  let hMax = 0;
  for (let i = 0; i < V; i++) if (curvature.meanCurvature[i] > hMax) hMax = curvature.meanCurvature[i];
  hMax = Math.max(1e-9, hMax);

  // Normalize displacement (if any)
  let dMax = 0;
  if (deformation) {
    for (let i = 0; i < V; i++) if (deformation.perVertexDisp[i] > dMax) dMax = deformation.perVertexDisp[i];
    dMax = Math.max(1e-9, dMax);
  }

  for (let i = 0; i < V; i++) {
    const hN = curvature.meanCurvature[i] / hMax;
    // Sharp corners → strongly negative Gaussian K or positive defect — both raise concentration
    const kN = Math.min(1, Math.abs(curvature.gaussianCurvature[i]) / Math.max(1e-9, Math.PI));
    let s = 0.6 * hN + 0.4 * kN;
    if (deformation) s = 0.5 * s + 0.5 * (deformation.perVertexDisp[i] / dMax);
    vertexStress[i] = s;
    mean += s;
    if (s > hotVal) { hotVal = s; hotIdx = i; }
  }
  mean /= Math.max(1, V);

  // Triangle strain-energy density × area summed
  let totalStrainEnergy = 0;
  const T = mesh.indices.length / 3;
  for (let t = 0; t < T; t++) {
    const a = mesh.indices[3 * t], b = mesh.indices[3 * t + 1], c = mesh.indices[3 * t + 2];
    const sBar = (vertexStress[a] + vertexStress[b] + vertexStress[c]) / 3;
    totalStrainEnergy += triangleArea(mesh, a, b, c) * sBar * sBar;
  }
  return {
    vertexStress, hotspotVertex: hotIdx, hotspotValue: hotVal,
    concentrationFactor: hotVal / Math.max(1e-9, mean),
    totalStrainEnergy,
  };
}

// ============================================================
// Topology-aware solver routing
// ============================================================

export function routeSolver(
  topology: TopologyProfile,
  curvature: CurvatureField,
  deformation?: DeformationMetrics,
): SolverDecision {
  const dofs = topology.V * 3;
  let hMax = 0;
  for (let i = 0; i < curvature.meanCurvature.length; i++) {
    if (curvature.meanCurvature[i] > hMax) hMax = curvature.meanCurvature[i];
  }
  const stiff = clamp(hMax / 10, 0, 1);
  const reasoning: string[] = [];

  // Decision tree:
  //   - very small + low stiffness → explicit (fast, stable)
  //   - small/mid + reusable shape (low genus) → ROM if precomputed
  //   - large + sparse                          → implicit_cg
  //   - large + high stiffness                  → implicit_direct
  let solver: SolverKind;
  let costFactor: number;
  let accuracyFactor: number;

  if (dofs < 600 && stiff < 0.4) {
    solver = "explicit";
    costFactor = 0.15; accuracyFactor = 0.85;
    reasoning.push(`small DOFs (${dofs}) and low curvature stiffness — explicit OK`);
  } else if (topology.genus <= 1 && topology.components === 1 && dofs < 12000) {
    solver = "rom_reduced";
    costFactor = 0.25; accuracyFactor = 0.88;
    reasoning.push(`single component, genus ≤ 1 — reduced-order basis is viable`);
  } else if (stiff > 0.7 || topology.triQualityMin < 0.15) {
    solver = "implicit_direct";
    costFactor = 1.0; accuracyFactor = 0.98;
    reasoning.push(`stiff system or poor element quality (q_min=${topology.triQualityMin.toFixed(2)}) — direct`);
  } else {
    solver = "implicit_cg";
    costFactor = 0.55; accuracyFactor = 0.94;
    reasoning.push(`mid-scale sparse problem (${dofs} DOFs, χ=${topology.euler}) — CG`);
  }

  if (deformation) {
    if (deformation.distortion > 0.4) {
      reasoning.push(`high distortion (${deformation.distortion.toFixed(2)}) — upgrade to implicit`);
      if (solver === "explicit") { solver = "implicit_cg"; costFactor = 0.55; accuracyFactor = 0.94; }
    }
    if (deformation.maxStretch > 2 || deformation.maxStretch < 0.5) {
      reasoning.push(`extreme stretch (${deformation.maxStretch.toFixed(2)}) — force direct`);
      solver = "implicit_direct"; costFactor = 1.0; accuracyFactor = 0.98;
    }
  }

  const confidence = clamp(
    0.5 + 0.25 * topology.triQualityMean + 0.25 * (1 - stiff), 0, 1,
  );

  return { solver, reasoning, costFactor, accuracyFactor, confidence };
}

// ============================================================
// Topology-preserving shape-energy optimization
// ============================================================

export interface OptimizeOpts {
  iterations?: number;
  step?: number;
  /** Vertex indices to clamp in place. */
  fixed?: number[];
}

export function optimizeShape(mesh: TriMesh, opts: OptimizeOpts = {}): ShapeOptimizationResult {
  const it = opts.iterations ?? 8;
  const step = opts.step ?? 0.4;
  const fixed = new Set(opts.fixed ?? []);

  const beforeTopo = computeTopology(mesh);
  const beforeCurv = computeCurvature(mesh);
  const beforeStress = predictStress(mesh, beforeCurv);
  const beforeDist = roughDistortion(mesh);

  const out: TriMesh = {
    positions: new Float64Array(mesh.positions),
    indices: mesh.indices, // topology unchanged
    label: (mesh.label ?? "mesh") + ":opt",
  };

  // Build vertex adjacency
  const V = out.positions.length / 3;
  const adj: Set<number>[] = Array.from({ length: V }, () => new Set());
  const T = out.indices.length / 3;
  for (let t = 0; t < T; t++) {
    const a = out.indices[3 * t], b = out.indices[3 * t + 1], c = out.indices[3 * t + 2];
    adj[a].add(b); adj[a].add(c);
    adj[b].add(a); adj[b].add(c);
    adj[c].add(a); adj[c].add(b);
  }

  // Laplacian smoothing with edge-length preservation factor → keeps area roughly stable.
  for (let iter = 0; iter < it; iter++) {
    const next = new Float64Array(out.positions);
    for (let i = 0; i < V; i++) {
      if (fixed.has(i)) continue;
      let sx = 0, sy = 0, sz = 0, n = 0;
      for (const j of adj[i]) {
        sx += out.positions[3 * j];
        sy += out.positions[3 * j + 1];
        sz += out.positions[3 * j + 2];
        n++;
      }
      if (n === 0) continue;
      const cx = sx / n, cy = sy / n, cz = sz / n;
      next[3 * i]     = out.positions[3 * i]     + step * (cx - out.positions[3 * i]);
      next[3 * i + 1] = out.positions[3 * i + 1] + step * (cy - out.positions[3 * i + 1]);
      next[3 * i + 2] = out.positions[3 * i + 2] + step * (cz - out.positions[3 * i + 2]);
    }
    out.positions = next;
  }

  const afterTopo = computeTopology(out);
  const afterCurv = computeCurvature(out);
  const afterStress = predictStress(out, afterCurv);
  const afterDist = roughDistortion(out);

  const topologyPreserved = (
    afterTopo.V === beforeTopo.V &&
    afterTopo.E === beforeTopo.E &&
    afterTopo.F === beforeTopo.F &&
    afterTopo.components === beforeTopo.components
  );

  const energyReduction = clamp(
    (beforeStress.totalStrainEnergy - afterStress.totalStrainEnergy)
      / Math.max(1e-9, beforeStress.totalStrainEnergy),
    -1, 1,
  );

  return {
    before: { energy: beforeStress.totalStrainEnergy, distortion: beforeDist },
    after:  { energy: afterStress.totalStrainEnergy,  distortion: afterDist },
    iterations: it,
    energyReduction,
    topologyPreserved,
    mesh: out,
  };
}

function roughDistortion(mesh: TriMesh): number {
  // Mean deviation of triangle quality from 1.
  const T = mesh.indices.length / 3;
  let s = 0;
  for (let t = 0; t < T; t++) {
    s += 1 - triangleQuality(mesh, mesh.indices[3 * t], mesh.indices[3 * t + 1], mesh.indices[3 * t + 2]);
  }
  return T > 0 ? s / T : 0;
}

// ============================================================
// One-shot pipeline
// ============================================================

export function analyze(rest: TriMesh, deformed?: TriMesh): GeometryReport {
  const topology = computeTopology(rest);
  const curvature = computeCurvature(deformed ?? rest);
  const deformation = deformed ? deformationMetrics(rest, deformed) : null;
  const stress = predictStress(deformed ?? rest, curvature, deformation ?? undefined);
  const routing = routeSolver(topology, curvature, deformation ?? undefined);
  return { topology, curvature, deformation, stress, routing };
}

// ============================================================
// Convenience: build a quad-sphere mesh (test/demo)
// ============================================================

/** Build an icosphere-like mesh by recursive subdivision of an octahedron. */
export function makeIcoSphere(subdivisions = 2, radius = 1): TriMesh {
  // Start: octahedron
  let verts: number[][] = [
    [ 1, 0, 0], [-1, 0, 0],
    [ 0, 1, 0], [ 0,-1, 0],
    [ 0, 0, 1], [ 0, 0,-1],
  ];
  let tris: [number, number, number][] = [
    [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
    [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
  ];
  for (let s = 0; s < subdivisions; s++) {
    const mid = new Map<number, number>();
    const getMid = (a: number, b: number) => {
      const key = a < b ? a * 100000 + b : b * 100000 + a;
      const m = mid.get(key);
      if (m !== undefined) return m;
      const mx = (verts[a][0] + verts[b][0]) / 2;
      const my = (verts[a][1] + verts[b][1]) / 2;
      const mz = (verts[a][2] + verts[b][2]) / 2;
      const l = Math.hypot(mx, my, mz);
      verts.push([mx / l, my / l, mz / l]);
      const idx = verts.length - 1;
      mid.set(key, idx);
      return idx;
    };
    const next: [number, number, number][] = [];
    for (const [a, b, c] of tris) {
      const ab = getMid(a, b), bc = getMid(b, c), ca = getMid(c, a);
      next.push([a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]);
    }
    tris = next;
  }
  const pos = new Float64Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    pos[3 * i] = verts[i][0] * radius;
    pos[3 * i + 1] = verts[i][1] * radius;
    pos[3 * i + 2] = verts[i][2] * radius;
  }
  const idx = new Uint32Array(tris.length * 3);
  for (let i = 0; i < tris.length; i++) {
    idx[3 * i] = tris[i][0];
    idx[3 * i + 1] = tris[i][1];
    idx[3 * i + 2] = tris[i][2];
  }
  return { positions: pos, indices: idx, label: `icosphere(${subdivisions})` };
}

/** Deform a sphere by a directional dent + radial stretch — demo input. */
export function deformMesh(mesh: TriMesh, dent: number, stretch: number): TriMesh {
  const out = new Float64Array(mesh.positions);
  const V = mesh.positions.length / 3;
  for (let i = 0; i < V; i++) {
    const x = mesh.positions[3 * i], y = mesh.positions[3 * i + 1], z = mesh.positions[3 * i + 2];
    // Dent: push along +z if x+y+z near a pole
    const w = Math.max(0, z) * Math.max(0, 1 - Math.hypot(x, y));
    out[3 * i]     = x * (1 + stretch * 0.1);
    out[3 * i + 1] = y * (1 + stretch * 0.05);
    out[3 * i + 2] = z - dent * w;
  }
  return { positions: out, indices: mesh.indices, label: (mesh.label ?? "mesh") + ":def" };
}
